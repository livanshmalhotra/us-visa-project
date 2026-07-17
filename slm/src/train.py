import os
import random
import time
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    IntervalStrategy
)
from slm.src.model import get_model_and_tokenizer
from slm.src.dataset import SLMTextDataset

def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(eval_pred):
    """Computes evaluation metrics on predictions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0)
    }

def train_pipeline(
    train_csv_path: str,
    val_csv_path: str,
    model_name: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    seed: int = 42,
    early_stopping_patience: int = 2
):
    """
    Orchestrates the entire SLM training process.
    """
    set_seed(seed)
    start_time = time.time()
    
    print("Loading datasets...")
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_name)
    
    # Create PyTorch datasets
    print("Tokenizing datasets...")
    train_dataset = SLMTextDataset(
        texts=train_df["text"],
        labels=train_df["label"],
        tokenizer=tokenizer
    )
    val_dataset = SLMTextDataset(
        texts=val_df["text"],
        labels=val_df["label"],
        tokenizer=tokenizer
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        logging_steps=10,
        disable_tqdm=True,  # Keeps logs clean and concise
        report_to="none"    # Prevent wandb/tensorboard connection prompts
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )
    
    print("Starting training...")
    train_result = trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # Save best model
    best_model_path = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    print(f"Saving best model and tokenizer to: {best_model_path}")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    # Save training metrics and config info
    metrics_summary = {
        "training_time_seconds": training_time,
        "epochs_trained": train_result.metrics.get("epoch", epochs),
        "total_flos": train_result.metrics.get("total_flos", 0),
        "train_loss": train_result.metrics.get("train_loss", 0.0),
        "hyperparameters": {
            "model_name": model_name,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "seed": seed,
            "epochs": epochs
        }
    }
    
    summary_path = os.path.join(best_model_path, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"Saved training summary at: {summary_path}")
    return trainer, training_time
