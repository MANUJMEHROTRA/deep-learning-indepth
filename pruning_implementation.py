import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score


def load_model_and_tokenizer(model_alias):
    """Loads the trained model and tokenizer."""
    model_dir = f"model-metric/{model_alias}"
    model_path = os.path.join(model_dir, f"{model_alias}.pth")
    
    # Load label encoder
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    
    num_classes = len(label_encoder.classes_)
    
    # Initialize model 
    model = DistilBERTCustom(num_labels=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return model, label_encoder


def get_model_size(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def evaluate_accuracy(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            labels = labels.cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def prune_attention_heads(model, encoder_index, prune_percent):
    """
    Prune attention heads in a specific transformer encoder layer.
    
    Args:
        model: The DistilBERT model
        encoder_index: Index of the encoder layer (0-5)
        prune_percent: Percentage of attention heads to prune
    
    Returns:
        Pruned model
    """
    # Skip if requested to not prune the first or last encoder
    if encoder_index == 0 or encoder_index == 5:
        print(f"Skipping encoder {encoder_index} as requested")
        return model
    
    # Get attention layer
    attention = model.bert.distilbert.transformer.layer[encoder_index].attention
    
    # DistilBERT typically has 12 attention heads
    num_heads = 12
    num_to_prune = int(num_heads * prune_percent / 100)
    
    if num_to_prune == 0:
        return model
    
    # Create a mask for the attention weights
    # For simplicity, we'll prune the first n heads based on prune_percent
    head_size = attention.q_lin.weight.size(0) // num_heads
    
    # Create masks for q, k, v projections
    for name, param in [('q_lin', attention.q_lin), ('k_lin', attention.k_lin), ('v_lin', attention.v_lin)]:
        mask = torch.ones_like(param.weight)
        for head in range(num_to_prune):
            # Zero out the weights for this head
            mask[head * head_size:(head + 1) * head_size, :] = 0
        
        # Apply the mask
        param.weight.data *= mask
    
    # Mask for output projection
    mask = torch.ones_like(attention.out_lin.weight)
    for head in range(num_to_prune):
        # Zero out corresponding columns in the output projection
        mask[:, head * head_size:(head + 1) * head_size] = 0
    
    attention.out_lin.weight.data *= mask
    
    return model


def prune_ffn(model, encoder_index, prune_percent):
    """
    Prune neurons in the FFN (Feed-Forward Network) of a specific encoder layer.
    
    Args:
        model: The DistilBERT model
        encoder_index: Index of the encoder layer (0-5)
        prune_percent: Percentage of neurons to prune
    
    Returns:
        Pruned model
    """
    # Skip if requested to not prune the first or last encoder
    if encoder_index == 0 or encoder_index == 5:
        print(f"Skipping encoder {encoder_index} as requested")
        return model
    
    # Get FFN layer
    ffn = model.bert.distilbert.transformer.layer[encoder_index].ffn
    
    # Calculate number of neurons to prune
    hidden_dim = ffn.lin1.weight.size(0)
    num_to_prune = int(hidden_dim * prune_percent / 100)
    
    if num_to_prune == 0:
        return model
    
    # For simplicity, we'll prune the first n neurons
    # Create mask for the first linear layer
    mask1 = torch.ones_like(ffn.lin1.weight)
    mask1[:num_to_prune, :] = 0
    ffn.lin1.weight.data *= mask1
    
    if ffn.lin1.bias is not None:
        bias_mask1 = torch.ones_like(ffn.lin1.bias)
        bias_mask1[:num_to_prune] = 0
        ffn.lin1.bias.data *= bias_mask1
    
    # Create mask for the second linear layer
    mask2 = torch.ones_like(ffn.lin2.weight)
    mask2[:, :num_to_prune] = 0
    ffn.lin2.weight.data *= mask2
    
    return model


def run_sensitivity_scan(model_alias, train_loader, test_loader, prune_type='attention', 
                         encoder_indices=range(1, 5), prune_percentages=None):
    """
    Run a sensitivity scan by pruning different percentages of the model.
    
    Args:
        model_alias: Name of the model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        prune_type: 'attention' or 'ffn'
        encoder_indices: List of encoder indices to prune
        prune_percentages: List of pruning percentages to try
    
    Returns:
        DataFrame with pruning results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if prune_percentages is None:
        prune_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    results = []
    
    # Load original model to get baseline accuracy
    original_model, _ = load_model_and_tokenizer(model_alias)
    original_model.to(device)
    baseline_accuracy = evaluate_accuracy(original_model, test_loader, device)
    baseline_size = get_model_size(original_model)
    
    print(f"Baseline model - Accuracy: {baseline_accuracy:.4f}, Size: {baseline_size:.2f} MB")
    
    # Store baseline for reference
    results.append({
        'encoder_index': 'baseline',
        'prune_percent': 0,
        'accuracy': baseline_accuracy,
        'accuracy_drop_percent': 0.0,
        'model_size_mb': baseline_size,
        'size_reduction_percent': 0.0,
        'prune_type': 'none'
    })
    
    # Run pruning for each encoder and pruning percentage
    for encoder_idx in encoder_indices:
        for prune_percent in prune_percentages:
            if prune_percent == 0:
                continue  # Skip 0% pruning as it's the baseline
                
            print(f"Pruning {prune_type} in encoder {encoder_idx} by {prune_percent}%")
            
            # Load a fresh model for each pruning experiment
            model, _ = load_model_and_tokenizer(model_alias)
            model.to(device)
            
            # Apply pruning
            if prune_type == 'attention':
                model = prune_attention_heads(model, encoder_idx, prune_percent)
            elif prune_type == 'ffn':
                model = prune_ffn(model, encoder_idx, prune_percent)
            
            # Evaluate
            accuracy = evaluate_accuracy(model, test_loader, device)
            model_size = get_model_size(model)
            
            # Calculate metrics
            accuracy_drop = baseline_accuracy - accuracy
            accuracy_drop_percent = (accuracy_drop / baseline_accuracy) * 100
            size_reduction = baseline_size - model_size
            size_reduction_percent = (size_reduction / baseline_size) * 100
            
            print(f"Results - Accuracy: {accuracy:.4f} ({accuracy_drop_percent:.2f}% drop), "
                  f"Size: {model_size:.2f} MB ({size_reduction_percent:.2f}% reduction)")
            
            # Store results
            results.append({
                'encoder_index': encoder_idx,
                'prune_percent': prune_percent,
                'accuracy': accuracy,
                'accuracy_drop_percent': accuracy_drop_percent,
                'model_size_mb': model_size,
                'size_reduction_percent': size_reduction_percent,
                'prune_type': prune_type
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(f"model-metric/{model_alias}/pruning", exist_ok=True)
    results_df.to_csv(f"model-metric/{model_alias}/pruning/{prune_type}_sensitivity_scan.csv", index=False)
    
    return results_df


def plot_sensitivity_scan(results_df, model_alias, prune_type):
    """
    Plot the sensitivity scan results.
    
    Args:
        results_df: DataFrame with pruning results
        model_alias: Name of the model
        prune_type: 'attention' or 'ffn'
    """
    # Filter out baseline row
    plot_df = results_df[results_df['encoder_index'] != 'baseline']
    
    # Create output directory
    os.makedirs(f"model-metric/{model_alias}/pruning", exist_ok=True)
    
    # Plot 1: Accuracy drop vs pruning percentage
    plt.figure(figsize=(12, 8))
    for encoder_idx in plot_df['encoder_index'].unique():
        subset = plot_df[plot_df['encoder_index'] == encoder_idx]
        plt.plot(subset['prune_percent'], subset['accuracy_drop_percent'], 
                 marker='o', label=f'Encoder {encoder_idx}')
    
    plt.title(f'Impact of {prune_type.upper()} Pruning on Model Accuracy')
    plt.xlabel('% Parameters Pruned')
    plt.ylabel('% Drop in Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"model-metric/{model_alias}/pruning/{prune_type}_accuracy_impact.png", dpi=300)
    
    # Plot 2: Model size reduction vs pruning percentage
    plt.figure(figsize=(12, 8))
    for encoder_idx in plot_df['encoder_index'].unique():
        subset = plot_df[plot_df['encoder_index'] == encoder_idx]
        plt.plot(subset['prune_percent'], subset['size_reduction_percent'], 
                 marker='o', label=f'Encoder {encoder_idx}')
    
    plt.title(f'Impact of {prune_type.upper()} Pruning on Model Size')
    plt.xlabel('% Parameters Pruned')
    plt.ylabel('% Reduction in Model Size')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"model-metric/{model_alias}/pruning/{prune_type}_size_impact.png", dpi=300)
    
    # Plot 3: Accuracy drop vs model size reduction
    plt.figure(figsize=(12, 8))
    for encoder_idx in plot_df['encoder_index'].unique():
        subset = plot_df[plot_df['encoder_index'] == encoder_idx]
        plt.plot(subset['size_reduction_percent'], subset['accuracy_drop_percent'], 
                 marker='o', label=f'Encoder {encoder_idx}')
    
    plt.title(f'Accuracy-Size Trade-off with {prune_type.upper()} Pruning')
    plt.xlabel('% Reduction in Model Size')
    plt.ylabel('% Drop in Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"model-metric/{model_alias}/pruning/{prune_type}_tradeoff.png", dpi=300)
    
    # Plot 4: Combined plot with both attention and FFN if available
    if os.path.exists(f"model-metric/{model_alias}/pruning/{'ffn' if prune_type == 'attention' else 'attention'}_sensitivity_scan.csv"):
        other_df = pd.read_csv(f"model-metric/{model_alias}/pruning/{'ffn' if prune_type == 'attention' else 'attention'}_sensitivity_scan.csv")
        other_df = other_df[other_df['encoder_index'] != 'baseline']
        
        # Create a combined plot
        plt.figure(figsize=(14, 10))
        
        # Plot current type with solid lines
        for encoder_idx in plot_df['encoder_index'].unique():
            subset = plot_df[plot_df['encoder_index'] == encoder_idx]
            plt.plot(subset['prune_percent'], subset['accuracy_drop_percent'], 
                     marker='o', linestyle='-', 
                     label=f'{prune_type.upper()} - Encoder {encoder_idx}')
        
        # Plot other type with dashed lines
        for encoder_idx in other_df['encoder_index'].unique():
            subset = other_df[other_df['encoder_index'] == encoder_idx]
            plt.plot(subset['prune_percent'], subset['accuracy_drop_percent'], 
                     marker='x', linestyle='--', 
                     label=f"{other_df['prune_type'].iloc[0].upper()} - Encoder {encoder_idx}")
        
        plt.title('Impact of Different Pruning Strategies on Model Accuracy')
        plt.xlabel('% Parameters Pruned')
        plt.ylabel('% Drop in Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"model-metric/{model_alias}/pruning/combined_accuracy_impact.png", dpi=300)