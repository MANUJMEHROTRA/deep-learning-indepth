from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tritonclient.http as httpclient
from transformers import AutoTokenizer
import numpy as np
import pickle
from typing import Dict, Any, List, Optional
import uvicorn
import time

app = FastAPI(title="DistilBERT Issue Classification API")

# Load the tokenizer and label encoder
tokenizer = AutoTokenizer.from_pretrained("model-metric/distilbert-uncased/tokenizer")

# Load the label encoder
with open("model-metric/distilbert-uncased/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Set up Triton client
client = httpclient.InferenceServerClient(url="localhost:8000")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    latency: float
    throughput: float
    inference_server: str



@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Tokenize input text
        inputs = tokenizer(
            request.text, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=512, 
            truncation=True
        )
        
        # Convert to numpy arrays
        input_ids = inputs["input_ids"].numpy().astype(np.int64)
        attention_mask = inputs["attention_mask"].numpy().astype(np.int64)
        
        # Create the inference inputs
        triton_inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
        ]
        
        # Set the data for each input
        triton_inputs[0].set_data_from_numpy(input_ids)
        triton_inputs[1].set_data_from_numpy(attention_mask)
        
        start_time = time.time()
        # Create the inference outputs
        outputs = [httpclient.InferRequestedOutput("output")]
        

        # Perform inference
        result = client.infer("distilbert_lora", inputs=triton_inputs, outputs=outputs)
        end_time = time.time()

        # Get the output data
        output_data = result.as_numpy("output")
        predicted_class_id = np.argmax(output_data, axis=1)[0]
        predicted_class = label_encoder.inverse_transform([predicted_class_id])[0]
        
        # Get probabilities
        probabilities = output_data[0]
        class_probs = {label_encoder.inverse_transform([i])[0]: float(prob) 
                      for i, prob in enumerate(probabilities)}
                # Calculate metrics
        latency = end_time - start_time
        throughput = 1 / latency if latency > 0 else 0

        return {
            "predicted_class": predicted_class,
            "confidence": float(probabilities[predicted_class_id]),
            "all_probabilities": class_probs,
            "latency": latency, 
            "throughput": throughput,
            "inference_server": "Nvidia Triton"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        client.is_server_live()
        return {"status": "ok", "message": "Server is live"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Server is not available: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)