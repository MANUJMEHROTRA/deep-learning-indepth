import os
import json
import torch
import time
import pickle
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from models import DistilBERTWithLoRA, LoRAMobileLLM

# Load model alias mapping
with open("model_dict.json", "r") as f:
    model_dict = json.load(f)

# FastAPI App
app = FastAPI()

# Model, tokenizer, and label encoder caches
loaded_models = {}
loaded_label_encoders = {}
loaded_tokenizers = {}
ort_sessions = {}

class ConversationInput(BaseModel):
    model_alias: str
    use_onnx: bool
    conversation: str

class BatchConversationInput(BaseModel):
    model_alias: str
    use_onnx: bool
    conversations: List[str]

def load_label_encoder(model_alias):
    """Loads the pre-trained label encoder from disk."""
    if model_alias not in loaded_label_encoders:
        label_encoder_path = f"model-metric/{model_alias}/label_encoder.pkl"
        if not os.path.exists(label_encoder_path):
            raise HTTPException(status_code=404, detail=f"Label encoder not found at {label_encoder_path}")

        with open(label_encoder_path, "rb") as f:
            loaded_label_encoders[model_alias] = pickle.load(f)

    return loaded_label_encoders[model_alias]

def load_tokenizer(model_alias):
    """Loads the tokenizer from disk or from huggingface."""
    if model_alias not in loaded_tokenizers:
        # Check if local tokenizer exists
        tokenizer_path = f"model-metric/{model_alias}/tokenizer/"
        
        if os.path.exists(tokenizer_path):
            # Load from local path
            try:
                # For MobileLLM, we need to set trust_remote_code=True
                if "mobileLLM-125M" in model_alias.lower():
                    loaded_tokenizers[model_alias] = AutoTokenizer.from_pretrained(
                        tokenizer_path, 
                        trust_remote_code=True
                    )
                else:
                    loaded_tokenizers[model_alias] = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                # Fall back to loading from model_dict if local loading fails
                model_name = model_dict.get(model_alias)
                if "mobileLLM-125M" in model_alias.lower():
                    loaded_tokenizers[model_alias] = AutoTokenizer.from_pretrained(
                        model_name, 
                        trust_remote_code=True
                    )
                else:
                    loaded_tokenizers[model_alias] = AutoTokenizer.from_pretrained(model_name)
        else:
            # If no local tokenizer, load from model name in model_dict
            model_name = model_dict.get(model_alias)
            if not model_name:
                raise HTTPException(status_code=400, detail=f"Unknown model alias: {model_alias}")
            
            # For MobileLLM, we need to set trust_remote_code=True
            if "mobileLLM-125M" in model_alias.lower():
                loaded_tokenizers[model_alias] = AutoTokenizer.from_pretrained(
                    model_name, 
                    trust_remote_code=True
                )
            else:
                loaded_tokenizers[model_alias] = AutoTokenizer.from_pretrained(model_name)
            
            # Save the tokenizer for future use
            os.makedirs(tokenizer_path, exist_ok=True)
            loaded_tokenizers[model_alias].save_pretrained(tokenizer_path)

    return loaded_tokenizers[model_alias]

def load_model(model_alias, num_labels):
    """Loads the model from disk, caching it if needed."""
    if model_alias not in loaded_models:
        model_name = model_dict.get(model_alias)
        if not model_name:
            raise HTTPException(status_code=400, detail=f"Unknown model alias: {model_alias}")
        
        model_path = f"model-metric/{model_alias}/{model_alias}.pth"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model weights not found at {model_path}")

        # Determine which model class to use based on the model alias
        if "distilbert-uncased" in model_alias:
            model = DistilBERTWithLoRA(num_labels=num_labels)
        elif "mobileLLM-125M" in model_alias:
            model = LoRAMobileLLM(num_labels=num_labels)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_alias}")
        
        # Load state dict
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model weights: {str(e)}")
        
        model.eval()
        loaded_models[model_alias] = model

    return loaded_models[model_alias]

def export_to_onnx(model, tokenizer, model_alias):
    """Exports the model to ONNX format."""
    model.eval().to("cpu")
    
    # Generate sample input based on model type
    sample_text = "This is a test input"
    sample_input = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
    
    input_ids = sample_input["input_ids"]
    attention_mask = sample_input["attention_mask"]
    
    onnx_path = f"model-metric/{model_alias}/{model_alias}.onnx"
    
    try:
        torch.onnx.export(
            model, 
            (input_ids, attention_mask), 
            onnx_path,
            input_names=["input_ids", "attention_mask"], 
            output_names=["output"], 
            dynamic_axes={"input_ids": {0: "batch", 1: "seq_length"}, 
                          "attention_mask": {0: "batch", 1: "seq_length"}, 
                          "output": {0: "batch"}}
        )
        return onnx_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting to ONNX: {str(e)}")

def run_onnx_inference(model_alias, input_ids, attention_mask):
    """Runs inference using ONNX Runtime."""
    onnx_path = f"model-metric/{model_alias}/{model_alias}.onnx"
    
    if not os.path.exists(onnx_path):
        raise HTTPException(status_code=404, detail=f"ONNX model not found at {onnx_path}")
    
    # Cache ONNX session to avoid reloading for each inference
    if onnx_path not in ort_sessions:
        try:
            ort_sessions[onnx_path] = ort.InferenceSession(onnx_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading ONNX session: {str(e)}")
    
    ort_session = ort_sessions[onnx_path]
    
    # Convert tensors to numpy arrays for ONNX Runtime with explicit dtype
    ort_inputs = {
        "input_ids": input_ids.cpu().numpy().astype(np.int64),
        "attention_mask": attention_mask.cpu().numpy().astype(np.int64)
    }
    
    try:
        ort_outputs = ort_session.run(None, ort_inputs)
        return torch.tensor(ort_outputs[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX inference error: {str(e)}")

@app.post("/predict_single")
async def predict_single(input_data: ConversationInput):
    """Predicts the issue area for a single conversation."""
    try:
        model_alias = input_data.model_alias
        
        # Load necessary components
        tokenizer = load_tokenizer(model_alias)
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Failed to load tokenizer")
            
        label_encoder = load_label_encoder(model_alias)
        num_labels = len(label_encoder.classes_)
        model = load_model(model_alias, num_labels)
        
        # Tokenize input
        start_time = time.time()
        try:
            encoded_input = tokenizer(input_data.conversation, 
                                     return_tensors="pt", 
                                     padding=True, 
                                     truncation=True)
        except Exception as e:
            raise HTTPException(status_code=500, 
                               detail=f"Tokenization error: {str(e)}. Check if the tokenizer is properly initialized.")
        
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        
        # Run inference
        if input_data.use_onnx:
            onnx_path = f"model-metric/{model_alias}/{model_alias}.onnx"
            if not os.path.exists(onnx_path):
                onnx_path = export_to_onnx(model, tokenizer, model_alias)
                
            # Reset timing after ONNX model is loaded/exported
            start_time = time.time()
            outputs = run_onnx_inference(model_alias, input_ids, attention_mask)
        else:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
        
        # Process outputs to get predictions
        predicted_label = torch.argmax(outputs, dim=1).item()
        predicted_issue_area = label_encoder.inverse_transform([predicted_label])[0]
        
        # Calculate latency
        latency = time.time() - start_time
        
        return {
            "predicted_issue_area": predicted_issue_area, 
            "latency": latency,
            "model_used": model_alias
        }
    except Exception as e:
        # Provide detailed error information
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(input_data: BatchConversationInput):
    """Predicts issue areas for a batch of conversations."""
    try:
        model_alias = input_data.model_alias
        
        # Load necessary components
        tokenizer = load_tokenizer(model_alias)
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Failed to load tokenizer")
            
        label_encoder = load_label_encoder(model_alias)
        num_labels = len(label_encoder.classes_)
        model = load_model(model_alias, num_labels)
        
        # Tokenize input batch
        start_time = time.time()
        try:
            encoded_inputs = tokenizer(input_data.conversations, 
                                      padding=True, 
                                      truncation=True, 
                                      return_tensors="pt")
        except Exception as e:
            raise HTTPException(status_code=500, 
                               detail=f"Batch tokenization error: {str(e)}. Check if the tokenizer is properly initialized.")
        
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]
        
        # Run inference
        if input_data.use_onnx:
            onnx_path = f"model-metric/{model_alias}/{model_alias}.onnx"
            if not os.path.exists(onnx_path):
                onnx_path = export_to_onnx(model, tokenizer, model_alias)
                
            # Reset timing after ONNX model is loaded/exported
            start_time = time.time()
            outputs = run_onnx_inference(model_alias, input_ids, attention_mask)
        else:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
        
        # Process outputs to get predictions
        predicted_labels = torch.argmax(outputs, dim=1).tolist()
        predicted_issue_areas = label_encoder.inverse_transform(predicted_labels).tolist()
        
        # Calculate metrics
        latency = time.time() - start_time
        throughput = len(input_data.conversations) / latency if latency > 0 else 0
        
        return {
            "predicted_issue_areas": predicted_issue_areas, 
            "latency": latency, 
            "throughput": throughput,
            "model_used": model_alias
        }
    except Exception as e:
        # Provide detailed error information
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/models")
async def list_models():
    """Lists all available models."""
    return {"available_models": list(model_dict.keys())}

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)