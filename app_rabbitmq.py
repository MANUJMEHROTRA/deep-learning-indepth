import os
import json
import torch
import time
import pickle
import onnxruntime as ort
import numpy as np
import aio_pika
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from models import DistilBERTWithLoRA, LoRAMobileLLM
import httpx
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

# RabbitMQ connection
rabbitmq_connection = None
rabbitmq_channel = None

# RabbitMQ configuration
RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
PREDICTION_EXCHANGE = "predictions"
PREDICTION_QUEUE = "prediction_results"


class ConversationInput(BaseModel):
    model_alias: str
    use_onnx: bool
    conversation: str
    callback_queue: Optional[str] = None


class BatchConversationInput(BaseModel):
    model_alias: str
    use_onnx: bool
    conversations: List[str]
    callback_queue: Optional[str] = None

class QueueInfo(BaseModel):
    name: str
    messages: int

class QueueListResponse(BaseModel):
    queues: List[QueueInfo]


async def get_rabbitmq_channel():
    """Get or create RabbitMQ channel."""
    global rabbitmq_connection, rabbitmq_channel
    
    if rabbitmq_connection is None or rabbitmq_connection.is_closed:
        rabbitmq_connection = await aio_pika.connect_robust(RABBITMQ_URL)
        
    if rabbitmq_channel is None or rabbitmq_channel.is_closed:
        rabbitmq_channel = await rabbitmq_connection.channel()
        
        # Declare exchange
        await rabbitmq_channel.declare_exchange(
            PREDICTION_EXCHANGE, 
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        
        # Declare default queue
        await rabbitmq_channel.declare_queue(
            PREDICTION_QUEUE,
            durable=True
        )
    
    return rabbitmq_channel


async def publish_message(routing_key, message_data, channel=None):
    """Publish a message to RabbitMQ."""
    if channel is None:
        channel = await get_rabbitmq_channel()
    
    exchange = await channel.get_exchange(PREDICTION_EXCHANGE)
    
    message = aio_pika.Message(
        body=json.dumps(message_data).encode(),
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT
    )
    
    await exchange.publish(message, routing_key=routing_key)


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


async def process_prediction(input_data: ConversationInput, channel=None):
    """Process a single prediction and publish results to RabbitMQ."""
    try:
        model_alias = input_data.model_alias
        
        # Load necessary components
        tokenizer = load_tokenizer(model_alias)
        if tokenizer is None:
            raise Exception("Failed to load tokenizer")
            
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
            raise Exception(f"Tokenization error: {str(e)}. Check if the tokenizer is properly initialized.")
        
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
        
        result = {
            "predicted_issue_area": predicted_issue_area, 
            "latency": latency,
            "model_used": model_alias,
            "conversation": input_data.conversation[:100] + "..." if len(input_data.conversation) > 100 else input_data.conversation
        }
        
        # Publish to RabbitMQ
        routing_key = input_data.callback_queue or PREDICTION_QUEUE
        await publish_message(routing_key, result, channel)
        
        return result
    except Exception as e:
        error_msg = {"error": f"Prediction error: {str(e)}"}
        if input_data.callback_queue:
            await publish_message(input_data.callback_queue, error_msg, channel)
        raise Exception(f"Prediction error: {str(e)}")


async def process_batch_prediction(input_data: BatchConversationInput, channel=None):
    """Process a batch prediction and publish results to RabbitMQ."""
    try:
        model_alias = input_data.model_alias
        
        # Load necessary components
        tokenizer = load_tokenizer(model_alias)
        if tokenizer is None:
            raise Exception("Failed to load tokenizer")
            
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
            raise Exception(f"Batch tokenization error: {str(e)}. Check if the tokenizer is properly initialized.")
        
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
        
        result = {
            "predicted_issue_areas": predicted_issue_areas, 
            "latency": latency, 
            "throughput": throughput,
            "model_used": model_alias,
            "num_conversations": len(input_data.conversations)
        }
        
        # Publish to RabbitMQ
        routing_key = input_data.callback_queue or PREDICTION_QUEUE
        await publish_message(routing_key, result, channel)
        
        return result
    except Exception as e:
        error_msg = {"error": f"Batch prediction error: {str(e)}"}
        if input_data.callback_queue:
            await publish_message(input_data.callback_queue, error_msg, channel)
        raise Exception(f"Batch prediction error: {str(e)}")


@app.post("/predict_single")
async def predict_single(
    input_data: ConversationInput, 
    background_tasks: BackgroundTasks,
    async_mode: bool = False,
    channel: aio_pika.Channel = Depends(get_rabbitmq_channel)
):
    """Predicts the issue area for a single conversation."""
    try:
        # If async mode, process in background and return immediately
        if async_mode:
            background_tasks.add_task(process_prediction, input_data, channel)
            return {
                "status": "processing",
                "message": f"Prediction request received and being processed asynchronously. Results will be published to {input_data.callback_queue or PREDICTION_QUEUE}"
            }
        
        # Otherwise, process synchronously and return results
        return await process_prediction(input_data, channel)
    except Exception as e:
        # Provide detailed error information
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(
    input_data: BatchConversationInput, 
    background_tasks: BackgroundTasks,
    async_mode: bool = False,
    channel: aio_pika.Channel = Depends(get_rabbitmq_channel)
):
    """Predicts issue areas for a batch of conversations."""
    try:
        # If async mode, process in background and return immediately
        if async_mode:
            background_tasks.add_task(process_batch_prediction, input_data, channel)
            return {
                "status": "processing",
                "message": f"Batch prediction request received and being processed asynchronously. Results will be published to {input_data.callback_queue or PREDICTION_QUEUE}"
            }
        
        # Otherwise, process synchronously and return results
        return await process_batch_prediction(input_data, channel)
    except Exception as e:
        # Provide detailed error information
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/models")
async def list_models():
    """Lists all available models."""
    return {"available_models": list(model_dict.keys())}


@app.get("/queues", response_model=QueueListResponse)
async def list_queues():
    """Lists all queues from the RabbitMQ management API."""
    try:
        async with httpx.AsyncClient(auth=('guest', 'guest')) as client:
            response = await client.get("http://localhost:15672/api/queues")
            response.raise_for_status()
            queues_data = response.json()

            queue_list = [QueueInfo(name=q["name"], messages=q["messages"]) for q in queues_data]
            return QueueListResponse(queues=queue_list)

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching queues from RabbitMQ management API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving queues: {str(e)}")



@app.post("/create_queue")
async def create_queue(
    queue_name: str,
    channel: aio_pika.Channel = Depends(get_rabbitmq_channel)
):
    """Creates a new queue and binds it to the predictions exchange."""
    try:
        # Declare the queue
        queue = await channel.declare_queue(
            queue_name,
            durable=True
        )
        
        # Get the exchange
        exchange = await channel.get_exchange(PREDICTION_EXCHANGE)
        
        # Bind the queue to the exchange with the queue name as routing key
        await queue.bind(exchange, routing_key=queue_name)
        
        return {
            "status": "success",
            "message": f"Queue '{queue_name}' created and bound to exchange '{PREDICTION_EXCHANGE}'"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating queue: {str(e)}")


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    try:
        # Try to connect to RabbitMQ
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        await connection.close()
        
        return {
            "status": "healthy",
            "rabbitmq": "connected"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "rabbitmq": f"connection error: {str(e)}"
        }


@app.on_event("startup")
async def startup_event():
    """Initialize RabbitMQ connection on startup."""
    global rabbitmq_connection, rabbitmq_channel
    
    try:
        rabbitmq_connection = await aio_pika.connect_robust(RABBITMQ_URL)
        rabbitmq_channel = await rabbitmq_connection.channel()
        
        # Declare exchange
        await rabbitmq_channel.declare_exchange(
            PREDICTION_EXCHANGE, 
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        
        # Declare default queue
        queue = await rabbitmq_channel.declare_queue(
            PREDICTION_QUEUE,
            durable=True
        )
        
        # Bind queue to exchange
        await queue.bind(PREDICTION_EXCHANGE, routing_key=PREDICTION_QUEUE)
        
        print(f"Successfully connected to RabbitMQ and set up exchange '{PREDICTION_EXCHANGE}' and queue '{PREDICTION_QUEUE}'")
    except Exception as e:
        print(f"Error connecting to RabbitMQ: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Close RabbitMQ connection on shutdown."""
    global rabbitmq_connection
    
    if rabbitmq_connection is not None and not rabbitmq_connection.is_closed:
        await rabbitmq_connection.close()
        print("RabbitMQ connection closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)