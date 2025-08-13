# --- Common Utilities and Setup ---
import os
import json
import time
import torch
import transformers
import accelerate
import huggingface_hub
import peft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from sklearn.utils import resample
from collections import Counter
import time
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


class DistilBERTWithLoRA(nn.Module):
    def __init__(self, num_labels, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super(DistilBERTWithLoRA, self).__init__()
        # Load the base model with the correct number of labels
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased",
            num_labels=num_labels  # Ensure this matches the number of classes
        )
        
        # LoRA Configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_lin", "k_lin", "v_lin"]
        )
        self.bert = get_peft_model(self.bert, lora_config)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # Return the logits directly


class LoRAMobileLLM(nn.Module):
    def __init__(self, num_labels, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super(LoRAMobileLLM, self).__init__()
        # Load the base model with the correct number of labels
        self.bert = AutoModelForCausalLM.from_pretrained(
            'facebook/MobileLLM-125M'
            # num_labels=num_labels  # Ensure this matches the number of classes
        )
        
        # LoRA Configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
        self.model = get_peft_model(self.bert, lora_config)
        
        # Custom classifier for issue prediction
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.classifier = nn.Linear(32000, num_labels)


    def forward(self, input_ids, attention_mask):
        # Get the model outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # For CausalLM models, we need to use the last hidden state of the last token
        # Get the last token for each sequence in the batch
        batch_size = input_ids.shape[0]
        last_token_indices = torch.sum(attention_mask, dim=1) - 1
        
        # Extract hidden states from the last layer
        hidden_states = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # Use the representation at the first token (CLS token) for classification
        cls_embedding = hidden_states[:, 0, :]
        # print("Hidden state shape before classifier:", cls_embedding.shape)
        
        # Pass through the classifier
        return self.classifier(cls_embedding)