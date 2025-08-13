# **LoRA (Low-Rank Adaptation) in Multi-Head Transformers**

## **1. Introduction to LoRA**
**LoRA (Low-Rank Adaptation)** is a technique that **reduces the number of trainable parameters** in Transformer models while maintaining performance. It achieves this by **decomposing weight updates** into low-rank matrices, allowing for efficient fine-tuning of large models like BERT, GPT, and T5.

### **Why Use LoRA?**
- 🚀 **Reduces computational cost** while fine-tuning.
- 🧠 **Maintains model performance** close to full fine-tuning.
- 💡 **Useful for domain-specific tasks** like **Aspect-Based Sentiment Analysis (ABSA)**.

---

## **2. LoRA in Multi-Head Transformers**
In **Multi-Head Self-Attention (MHSA)** layers, LoRA applies a **low-rank update** to **Query (Q) and Value (V) matrices** while keeping Key (K) **unchanged**:

\[ \Delta W = A B \]  
Where:
- \( A \) and \( B \) are low-rank matrices **(r x d)** and **(d x r)**.
- Instead of fine-tuning **full weights**, only \( A \) and \( B \) are optimized.

---

## **3. LoRA in BERT for ABSA**

We apply LoRA to **BERT's self-attention layers** for **Aspect-Based Sentiment Analysis (ABSA)**. This helps efficiently fine-tune BERT to focus on **specific aspects** in a sentence while **reducing computational overhead**.

### **Example:** Extracting Query (Q), Key (K), and Value (V) from BERT
```python
import torch
from transformers import BertModel, BertTokenizer

# Load Pretrained BERT with Attention Output
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Example Sentence for ABSA
sentence = "The battery life is amazing but the camera is terrible."
inputs = tokenizer(sentence, return_tensors="pt")

# Forward Pass to Extract Attention
outputs = model(**inputs)
attention = outputs.attentions  # Extract Attention Weights

# Get Q, K, V from the First Transformer Layer
layer_idx = 0  # First transformer layer
head_idx = 0  # First attention head

qkv_tensors = attention[layer_idx]  # Shape: (batch, num_heads, seq_len, seq_len)
qkv_attention = qkv_tensors[:, head_idx, :, :]  # Extract for one head

print("QKV Attention Shape:", qkv_attention.shape)  # (batch_size, seq_len, seq_len)
```

---

## **4. Implementing LoRA in PyTorch**
To integrate **LoRA into BERT**, we modify the **Query (Q) and Value (V) layers** in self-attention.

### **Step 1: Define a LoRA Adapter**
```python
import torch
import torch.nn as nn

class LoRAAdapter(nn.Module):
    def __init__(self, d_model, rank=8):
        super().__init__()
        self.A = nn.Linear(d_model, rank, bias=False)  # Low-rank A
        self.B = nn.Linear(rank, d_model, bias=False)  # Low-rank B

    def forward(self, x):
        return self.B(self.A(x))
```

### **Step 2: Modify BERT's Self-Attention with LoRA**
```python
from transformers import BertSelfAttention

class LoRABertSelfAttention(BertSelfAttention):
    def __init__(self, config, rank=8):
        super().__init__(config)
        d_model = config.hidden_size
        self.lora_q = LoRAAdapter(d_model, rank)
        self.lora_v = LoRAAdapter(d_model, rank)

    def forward(self, hidden_states, *args, **kwargs):
        q, k, v = self.query(hidden_states), self.key(hidden_states), self.value(hidden_states)
        q += self.lora_q(hidden_states)  # Apply LoRA to Query
        v += self.lora_v(hidden_states)  # Apply LoRA to Value
        return super().forward(hidden_states, *args, **kwargs)
```

### **Step 3: Replace BERT's Attention with LoRA**
```python
from transformers import BertModel

def apply_lora_to_bert(model, rank=8):
    for layer in model.encoder.layer:
        layer.attention.self = LoRABertSelfAttention(layer.attention.self.config, rank)
    return model

# Load BERT and Apply LoRA
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model = apply_lora_to_bert(bert_model, rank=8)

print("LoRA Applied to BERT Self-Attention!")
```

---

## **5. Training LoRA-BERT for ABSA**
After modifying BERT, we fine-tune it on an **ABSA dataset** like **SemEval-2014 Task 4**.

### **Step 1: Define Training Pipeline**
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-bert-absa",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=absa_train_dataset,
    eval_dataset=absa_eval_dataset,
)

trainer.train()
```

### **Step 2: Evaluate LoRA-BERT on ABSA**
```python
def predict_sentiment(model, sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    sentiment = torch.argmax(logits, dim=-1).item()
    return ["Negative", "Neutral", "Positive"][sentiment]

# Test Example
sentence = "The battery life is great but the camera is bad."
print("Predicted Sentiment:", predict_sentiment(bert_model, sentence))
```

---

## **6. Key Takeaways**
✅ **LoRA reduces trainable parameters** while keeping full model performance.  
✅ **Applied to Multi-Head Attention**, it modifies Query (Q) and Value (V).  
✅ **Useful for ABSA**, allowing efficient fine-tuning on sentiment datasets.  
✅ **LoRA-BERT performs well** in sentiment analysis while being **memory-efficient**.  

🚀 **LoRA is the future of efficient fine-tuning in NLP!**  

---
