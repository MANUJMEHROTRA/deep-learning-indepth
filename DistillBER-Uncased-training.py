# %%
# --- Common Utilities and Setup ---
import os
import json
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from sklearn.utils import resample
from collections import Counter
import time
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle

print("peft:", peft.__version__)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("Huggingface Hub:", huggingface_hub.__version__)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def update_model_dict(model_alias, MODEL_NAME):
    if not os.path.exists('model_dict.json'):
        model_dict = {}
    else:
        with open('model_dict.json', 'r') as file:
            model_dict = json.load(file)

    model_dict[model_alias] = MODEL_NAME

    with open('model_dict.json', 'w') as file:
        json.dump(model_dict, file)

# %%
def load_and_preprocess_data(filepath="./data/train-00000-of-00001-a5a7c6e4bb30b016.parquet"):
    """Loads and preprocesses the dataset."""
    df = pd.read_parquet(filepath)
    df = df[['conversation', 'issue_area']]
    print("Original distribution:\n", df['issue_area'].value_counts())
    label_encoder = LabelEncoder()
    df["labels"] = label_encoder.fit_transform(df["issue_area"])

    #saving Label-encoder
    label_encoder_path = f"model-metric/{model_alias}/label_encoder.pkl"
    os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
        
    return df, label_encoder

# %%
def balance_dataset(df, max_count=100, random_state=42):
    """Balances the dataset using oversampling."""
    balanced_df = pd.DataFrame()
    for issue in df['issue_area'].unique():
        subset = df[df['issue_area'] == issue]
        balanced_subset = resample(subset, replace=True, n_samples=max_count, random_state=random_state)
        balanced_df = pd.concat([balanced_df, balanced_subset])
    return balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)


# %%
def preprocess_conversation(conversation):
    """Preprocesses a conversation."""
    if isinstance(conversation, list):
        return " ".join([turn.get('text', '') for turn in conversation if isinstance(turn, dict)])
    return str(conversation).lower()

# %%
# Define PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        inputs = self.tokenizer(
            row["conversation"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        label = torch.tensor(row["labels"], dtype=torch.long)
        return input_ids, attention_mask, label

# %%
def create_dataloaders(df, tokenizer, batch_size=8, train_ratio=0.75):
    """Creates train and test DataLoaders."""
    train_size = int(train_ratio * len(df))
    train_df, test_df = df[:train_size], df[train_size:]
    train_dataset = CustomDataset(train_df, tokenizer)
    test_dataset = CustomDataset(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_df

# %%
class DistilBERTCustom(nn.Module):
    def __init__(self, num_labels, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super(DistilBERTCustom, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased",
            num_labels=num_labels  # Ensure this matches the number of classes
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # Return the logits directly

# %%
# Function to compute class weights
def compute_class_weights(labels, num_classes):
    counter = Counter(labels)
    total_samples = len(labels)
    weights = [total_samples / (num_classes * counter[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)

# %%
def train_model(model, train_loader, model_alias, epochs=3, learning_rate=5e-5, class_weights=None):
    """Trains the model and saves logs, metrics, and model weights."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Create directory for storing model metrics
    model_dir = f"model-metric/{model_alias}"
    os.makedirs(model_dir, exist_ok=True)

    # TensorBoard writer in the model directory
    writer = SummaryWriter(log_dir=model_dir)

    # Set up loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    epoch_losses = []
    metrics_data = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            labels = labels.cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

            # Log batch loss every 10 batches
            if batch_idx % 10 == 0:
                writer.add_scalar("BatchLoss/train", loss.item(), epoch * len(train_loader) + batch_idx)

        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        epoch_time = time.time() - start_time

        # Store metrics for CSV logging
        metrics_data.append([epoch + 1, avg_loss, accuracy, precision, recall, f1, epoch_time])

        # Print metrics
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}, Time={epoch_time:.2f}s")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        writer.add_scalar("Precision/train", precision, epoch)
        writer.add_scalar("Recall/train", recall, epoch)
        writer.add_scalar("F1-score/train", f1, epoch)
        writer.add_scalar("Time/Epoch", epoch_time, epoch)

    # Save model KPIs as CSV
    metrics_df = pd.DataFrame(metrics_data, columns=["Epoch", "Loss", "Accuracy", "Precision", "Recall", "F1-score", "Time (s)"])
    metrics_df.to_csv(os.path.join(model_dir, "training_metrics.csv"), index=False)

    # Save training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_plot_path = os.path.join(model_dir, "training_loss.png")
    plt.savefig(loss_plot_path)
    writer.add_figure("Training Loss", plt.gcf(), close=True)

    # Save model weights
    model_path = os.path.join(model_dir, f"{model_alias}.pth")
    torch.save(model.state_dict(), model_path)

    writer.flush()
    writer.close()


# %%
import os
import time
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def evaluate_model(model, test_loader, label_encoder, model_alias):
    """Evaluates the model and saves metrics, logs, and confusion matrix."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create directory for storing model metrics
    model_dir = f"model-metric/{model_alias}"
    os.makedirs(model_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=model_dir)

    all_preds, all_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            labels = labels.cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    eval_time = time.time() - start_time
    class_names = label_encoder.classes_

    # Compute metrics
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    class_metrics = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })

    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # Print and save classification report
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Save confusion matrix plot
    confusion_matrix_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    writer.add_figure("Confusion Matrix", plt.gcf(), close=True)

    # Print overall metrics
    print("\nPer-class Metrics:\n", class_metrics.to_string(index=False))
    print(f"\nOverall Metrics:\nPrecision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1-score: {overall_f1:.4f}, Eval Time: {eval_time:.2f}s")

    # Log metrics to TensorBoard
    writer.add_scalar("Precision/test", overall_precision)
    writer.add_scalar("Recall/test", overall_recall)
    writer.add_scalar("F1-score/test", overall_f1)
    writer.add_scalar("Evaluation Time", eval_time)

    # Log per-class metrics
    for i, class_name in enumerate(class_names):
        writer.add_scalar(f"Precision/{class_name}", precision[i])
        writer.add_scalar(f"Recall/{class_name}", recall[i])
        writer.add_scalar(f"F1-score/{class_name}", f1[i])

    writer.flush()
    writer.close()

    # Save evaluation metrics
    class_metrics.to_csv(os.path.join(model_dir, "class_metrics.csv"), index=False)
    cm_df.to_csv(os.path.join(model_dir, "confusion_matrix.csv"))

    return class_metrics, cm_df


# %%
def compute_class_weights(labels, num_classes):
    counter = Counter(labels)
    total_samples = len(labels)
    weights = [total_samples / (num_classes * counter[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)

# %%
MODEL_NAME = "distilbert/distilbert-base-uncased-pruned"
model_alias = 'distilbert-cased-lora'
update_model_dict(model_alias, MODEL_NAME)

# %%
df, label_encoder = load_and_preprocess_data()
balanced_df = balance_dataset(df)
balanced_df['conversation'] = balanced_df['conversation'].apply(preprocess_conversation)

# %%
    # Tokenization and DataLoaders
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
train_loader, test_loader, test_df = create_dataloaders(balanced_df, tokenizer)

# %%
# Model Initialization and Training
num_classes = len(label_encoder.classes_)
model = DistilBERTWithLoRA(num_labels=num_classes)
class_weights = compute_class_weights(balanced_df['labels'], num_classes)

# %%
train_model(model, train_loader,model_alias=model_alias, epochs=10, learning_rate=5e-5, class_weights=class_weights)

# %%
# Model Evaluation
evaluate_model(model, test_loader, label_encoder, model_alias)

# %%
compare_inference_performance(model, tokenizer, test_df, label_encoder, model_alias=model_alias)

# %%
tokenizer.save_pretrained(f"model-metric/{model_alias}/tokenizer/")
