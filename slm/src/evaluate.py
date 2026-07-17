import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from slm.src.dataset import SLMTextDataset

def evaluate_pipeline(
    test_csv_path: str,
    model_dir: str,
    metrics_output_path: str,
    confusion_matrix_output_path: str
) -> dict:
    """
    Evaluates the trained SLM on unseen test data and saves the metrics.
    """
    print(f"Loading test dataset from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    
    from slm.src.model import get_tokenizer, get_model
    tokenizer = get_tokenizer(model_dir)
    model = get_model(model_dir)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Create dataset and data loader
    test_dataset = SLMTextDataset(
        texts=test_df["text"],
        labels=test_df["label"],
        tokenizer=tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running inference on test dataset...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Probability of positive class (class 1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Handle ROC-AUC calculation (needs positive and negative class in test labels)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC: {e}")
        roc_auc = 0.5
        
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1])
        }
    }
    
    # Write metrics to file
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved successfully at: {metrics_output_path}")
    
    # Plot confusion matrix
    os.makedirs(os.path.dirname(confusion_matrix_output_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    
    # Simple heatmap plotting
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - SLM Model")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Denied (0)", "Certified (1)"])
    plt.yticks(tick_marks, ["Denied (0)", "Certified (1)"])
    
    # Label cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(confusion_matrix_output_path)
    plt.close()
    print(f"Confusion Matrix plot saved at: {confusion_matrix_output_path}")
    
    return metrics
