import os
import sys
from dotenv import load_dotenv

# Add project root and slm dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Load environment variables
load_dotenv()

from slm.src.data_conversion import (
    load_schema,
    load_raw_dataset,
    convert_dataframe,
    create_splits
)

def main():
    print("=== US Visa Tabular-to-Text Dataset Conversion Pipeline ===")
    
    # 1. Load schema config
    schema = load_schema()
    
    # 2. Ingest raw dataset
    raw_df = load_raw_dataset()
    
    # 3. Convert tabular records to natural language texts
    print("Converting structured records to natural language representation...")
    converted_df = convert_dataframe(raw_df, schema)
    
    # 4. Display 5 sample records for validation
    print("\n--- Converted Dataset Samples (5 Examples) ---")
    for i in range(min(5, len(converted_df))):
        print(f"Sample #{i+1}:")
        print(f"  TEXT : {converted_df.iloc[i]['text']}")
        print(f"  LABEL: {converted_df.iloc[i]['label']}")
        print("-" * 50)
        
    # 5. Create train/validation/test splits
    print("\nCreating stratified splits (70% Train, 15% Validation, 15% Test)...")
    train_df, val_df, test_df = create_splits(converted_df, seed=42)
    
    print(f"Split sizes:")
    print(f"  Training Set  : {len(train_df)} samples")
    print(f"  Validation Set: {len(val_df)} samples")
    print(f"  Testing Set   : {len(test_df)} samples")
    
    # 6. Save split datasets to data directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "text_dataset"))
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved split datasets successfully:")
    print(f"  Train Set      : {train_path}")
    print(f"  Validation Set : {val_path}")
    print(f"  Test Set       : {test_path}")
    print("========================================================\n")

if __name__ == "__main__":
    main()
