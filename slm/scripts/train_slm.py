import os
import sys
import argparse
from dotenv import load_dotenv

# Add project root and slm dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

load_dotenv()

from slm.src.train import train_pipeline

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a lightweight SLM on text representation of EasyVisa dataset")
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny", help="Hugging Face model identifier")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_csv = os.path.join(base_dir, "data", "text_dataset", "train.csv")
    val_csv = os.path.join(base_dir, "data", "text_dataset", "validation.csv")
    output_dir = os.path.join(base_dir, "models")
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print(f"Error: Converted split datasets not found. Please run: python slm/scripts/convert_tabular_to_text.py first.")
        sys.exit(1)
        
    print(f"=== SLM Pipeline Training Configuration ===")
    print(f"  Pretrained Model : {args.model_name}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Batch Size       : {args.batch_size}")
    print(f"  Learning Rate    : {args.learning_rate}")
    print(f"  Random Seed      : {args.seed}")
    print(f"  Early Stopping   : patience={args.patience}")
    print(f"===========================================\n")
    
    train_pipeline(
        train_csv_path=train_csv,
        val_csv_path=val_csv,
        model_name=args.model_name,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        early_stopping_patience=args.patience
    )

if __name__ == "__main__":
    main()
