import os
import sys
from dotenv import load_dotenv

# Add project root and slm dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

load_dotenv()

from slm.src.evaluate import evaluate_pipeline

def main():
    print("=== SLM Pipeline Evaluation ===")
    
    # Resolve paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_csv = os.path.join(base_dir, "data", "text_dataset", "test.csv")
    model_dir = os.path.join(base_dir, "models", "best_model")
    metrics_json = os.path.join(base_dir, "models", "metrics.json")
    cm_png = os.path.join(base_dir, "models", "confusion_matrix.png")
    
    if not os.path.exists(test_csv):
        print(f"Error: Converted test dataset not found at {test_csv}. Please run: python slm/scripts/convert_tabular_to_text.py first.")
        sys.exit(1)
        
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"Error: Trained best model not found at {model_dir}. Please run: python slm/scripts/train_slm.py first.")
        sys.exit(1)
        
    metrics = evaluate_pipeline(
        test_csv_path=test_csv,
        model_dir=model_dir,
        metrics_output_path=metrics_json,
        confusion_matrix_output_path=cm_png
    )
    
    print("\n--- Final Test Performance Metrics ---")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("-" * 38)
    print("Confusion Matrix:")
    print(f"  True Negatives (Denied -> Denied): {metrics['confusion_matrix']['true_negative']}")
    print(f"  False Positives (Denied -> Cert) : {metrics['confusion_matrix']['false_positive']}")
    print(f"  False Negatives (Cert -> Denied) : {metrics['confusion_matrix']['false_negative']}")
    print(f"  True Positives (Cert -> Cert)    : {metrics['confusion_matrix']['true_positive']}")
    print("======================================\n")

if __name__ == "__main__":
    main()
