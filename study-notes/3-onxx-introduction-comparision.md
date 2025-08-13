
---

## **ONNX with TensorRT: In-Depth Guide**

### **What is ONNX?**
ONNX (Open Neural Network Exchange) is an open standard format that allows machine learning models to be portable across different frameworks such as PyTorch, TensorFlow, and deployment runtimes like TensorRT and ONNX Runtime.

### **Why Use ONNX?**
- **Framework Interoperability**: Convert models from PyTorch, TensorFlow, etc., into a common format.
- **Optimized Performance**: Using ONNX Runtime or TensorRT can significantly speed up inference.
- **Deployment**: ONNX models can be deployed on various platforms, including cloud, edge, and mobile devices.

### **Why Use TensorRT?**
- TensorRT is NVIDIA's SDK for optimizing deep learning models.
- It provides **low latency** and **high throughput** inference, especially for **GPU-based** applications.
- Works best with **FP16/INT8 precision optimization**, reducing memory and computational cost.

---

## **Converting a PyTorch Model to ONNX**
Let's convert your PyTorch model (`lora_distilbert_trained.pth`) to ONNX.

### **1. Install Dependencies**
Ensure you have the required packages installed:

```bash
pip install torch onnx onnxruntime-gpu onnx-simplifier nvidia-pyindex nvidia-tensorrt
```

### **2. Load the Model and Convert to ONNX**
Modify your PyTorch code to export the model:

```python
import torch
import onnx
import onnxruntime as ort
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load fine-tuned PyTorch model
model_path = "./lora_distilbert_trained.pth"
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Define the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Sample input text
sample_text = "Agent: Thank you for calling BrownBox Customer Support. My name is Tom. How may I assist you today?"
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Export to ONNX
onnx_path = "distilbert.onnx"
torch.onnx.export(
    model,                         # Model to export
    (inputs["input_ids"], inputs["attention_mask"]),  # Example input
    onnx_path,                     # Output ONNX file
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},  # Support dynamic batching
    opset_version=11  # Compatible with TensorRT
)

print(f"ONNX model saved to {onnx_path}")
```

---

## **Checking and Simplifying the ONNX Model**
After exporting, verify that the ONNX model is valid:

```python
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")
```

Optionally, simplify the model for better optimization:

```python
from onnxsim import simplify

onnx_model, check = simplify(onnx_model)
assert check, "Simplified ONNX model is not valid!"
onnx.save(onnx_model, "distilbert_simplified.onnx")
print("ONNX model simplified and saved.")
```

---

## **Running ONNX Model Inference with ONNX Runtime**
Before using TensorRT, let's benchmark ONNX inference with ONNX Runtime:

```python
import numpy as np

# Load ONNX model
ort_session = ort.InferenceSession("distilbert_simplified.onnx", providers=["CUDAExecutionProvider"])

# Convert input tensors to NumPy arrays
onnx_inputs = {
    "input_ids": inputs["input_ids"].cpu().numpy(),
    "attention_mask": inputs["attention_mask"].cpu().numpy()
}

# Perform inference
onnx_outputs = ort_session.run(None, onnx_inputs)
predicted_label = np.argmax(onnx_outputs[0])

print("Predicted issue area (ONNX Runtime):", predicted_label)
```

---

## **Optimizing and Running ONNX Model with TensorRT**
### **1. Convert ONNX to TensorRT Engine**
TensorRT can compile ONNX models into an optimized inference engine:

```bash
trtexec --onnx=distilbert_simplified.onnx --saveEngine=distilbert.trt --fp16
```
This command:
- Converts `distilbert_simplified.onnx` into a TensorRT engine (`distilbert.trt`).
- Uses `--fp16` for half-precision optimization (optional).

### **2. Load and Run TensorRT Engine**
Once converted, load and run the model in Python:

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Load TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("distilbert.trt", "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

# Allocate memory for inputs/outputs
context = engine.create_execution_context()
input_ids = inputs["input_ids"].cuda().contiguous()
attention_mask = inputs["attention_mask"].cuda().contiguous()

# Run inference
bindings = [input_ids.data_ptr(), attention_mask.data_ptr()]
context.execute_v2(bindings)

# Process output
trt_output = torch.tensor(bindings[0]).cpu().numpy()
predicted_label = np.argmax(trt_output)

print("Predicted issue area (TensorRT):", predicted_label)
```

---

## **Benchmarking: ONNX vs Non-ONNX**
We will compare the inference speed of:
1. **PyTorch model** (native execution)
2. **ONNX Runtime** (optimized execution)
3. **TensorRT** (high-performance execution)

### **1. Benchmark PyTorch Model**
```python
import time

# Warm-up
for _ in range(5):
    with torch.no_grad():
        model(**inputs)

# Measure time
start = time.time()
for _ in range(100):
    with torch.no_grad():
        outputs = model(**inputs)
end = time.time()

print(f"PyTorch inference time: {(end - start) / 100:.6f} sec")
```

### **2. Benchmark ONNX Runtime**
```python
start = time.time()
for _ in range(100):
    ort_session.run(None, onnx_inputs)
end = time.time()

print(f"ONNX Runtime inference time: {(end - start) / 100:.6f} sec")
```

### **3. Benchmark TensorRT**
```python
start = time.time()
for _ in range(100):
    context.execute_v2(bindings)
end = time.time()

print(f"TensorRT inference time: {(end - start) / 100:.6f} sec")
```

---

## **Results and Expected Speedup**
| Model Type | Avg Inference Time (sec) | Speedup vs PyTorch |
|------------|------------------|-----------------|
| PyTorch | 0.XXX sec | 1.0x |
| ONNX Runtime | 0.XXX sec | ~1.5x - 3x |
| TensorRT | 0.XXX sec | ~3x - 10x |

- **ONNX Runtime is usually 1.5-3x faster** than PyTorch.
- **TensorRT provides a 3x-10x speedup**, especially with `FP16` or `INT8` precision.

---

## **Conclusion**
- **ONNX enables portability and optimization** across different platforms.
- **ONNX Runtime improves inference speed**, reducing CPU/GPU overhead.
- **TensorRT provides the best performance**, leveraging NVIDIA GPUs for optimized inference.

This setup makes **deployment faster** and **resource-efficient**, especially for real-time NLP applications.

---