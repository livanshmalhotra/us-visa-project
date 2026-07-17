import os
import sys
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def load_schema(schema_path: str = None) -> dict:
    """Loads the dataset schema from yaml config."""
    if schema_path is None:
        # Default path relative to this file
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "schema.yaml"))
    
    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)
    return schema

def load_raw_dataset() -> pd.DataFrame:
    """
    Attempts to load the raw tabular dataset from:
    1. MongoDB Atlas (using MONGODB_URL from environment).
    2. Local artifact feature store files.
    3. Falling back to the remote seeding CSV URL.
    """
    # 1. Try MongoDB
    mongodb_url = os.environ.get("MONGODB_URL") or os.environ.get("MONGODB_URI")
    if mongodb_url:
        try:
            print("Attempting to connect to MongoDB to load dataset...")
            client = MongoClient(mongodb_url, serverSelectionTimeoutMS=5000)
            db = client["US_VISA"]
            collection = db["visa_data"]
            if collection.count_documents({}) > 0:
                cursor = collection.find()
                df = pd.DataFrame(list(cursor))
                if "_id" in df.columns:
                    df = df.drop(columns=["_id"])
                print(f"Successfully loaded {len(df)} records from MongoDB.")
                return df
            else:
                print("MongoDB collection is empty.")
        except Exception as e:
            print(f"Failed to load from MongoDB: {e}")
            
    # 2. Try Local Artifacts
    try:
        print("Searching for local artifacts...")
        artifact_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifact"))
        if os.path.exists(artifact_root):
            timestamps = sorted(os.listdir(artifact_root))
            for ts in reversed(timestamps):
                feature_store_path = os.path.join(artifact_root, ts, "data_ingestion", "feature_store", "easyvisa.csv")
                if os.path.exists(feature_store_path):
                    df = pd.read_csv(feature_store_path)
                    print(f"Successfully loaded {len(df)} records from local artifact feature store at: {feature_store_path}")
                    return df
    except Exception as e:
        print(f"Failed to scan local artifacts: {e}")
        
    # 3. Fallback to Remote seeding URL
    fallback_url = "https://raw.githubusercontent.com/rochitasundar/Customer-profiling-using-ML-EasyVisa/master/EasyVisa.csv"
    print(f"Falling back to download dataset from URL: {fallback_url}")
    try:
        df = pd.read_csv(fallback_url)
        print(f"Successfully loaded {len(df)} records from remote URL.")
        return df
    except Exception as e:
        raise RuntimeError(f"Could not load dataset from any source: {e}")

def convert_row_to_text(row: dict) -> str:
    """
    Converts a single tabular record dictionary into a natural language sentence.
    Handles potential missing values safely.
    """
    continent = row.get("continent", "an unknown continent")
    education = row.get("education_of_employee", "some education")
    
    # Map job experience
    has_exp = str(row.get("has_job_experience", "N")).strip().upper()
    exp_str = "has previous job experience" if has_exp == "Y" else "does not have job experience"
    
    # Map job training
    req_train = str(row.get("requires_job_training", "N")).strip().upper()
    train_str = "requires job training" if req_train == "Y" else "does not require job training"
    
    # Numerical values handling
    no_emp = row.get("no_of_employees")
    if pd.isna(no_emp) or no_emp is None:
        emp_str = "an unspecified number of"
    else:
        emp_str = f"{int(no_emp)}"
        
    yr_estab = row.get("yr_of_estab")
    if pd.isna(yr_estab) or yr_estab is None:
        estab_str = "an unknown year"
    else:
        estab_str = f"the year {int(yr_estab)}"
        
    region = row.get("region_of_employment", "an unspecified")
    
    wage = row.get("prevailing_wage")
    wage_unit = str(row.get("unit_of_wage", "Year")).strip().lower()
    if pd.isna(wage) or wage is None:
        wage_str = "an unspecified wage"
    else:
        wage_str = f"a prevailing wage of {float(wage):.2f} per {wage_unit}"
        
    # Map full time position
    full_time = str(row.get("full_time_position", "Y")).strip().upper()
    ft_str = "is a full-time position" if full_time == "Y" else "is not a full-time position"
    
    text = (
        f"An applicant from {continent} has an education level of {education}. "
        f"The applicant {exp_str} and {train_str}. "
        f"The employer was established in {estab_str} and has {emp_str} employees. "
        f"The position is located in the {region} region, pays {wage_str}, and {ft_str}."
    )
    return text

def convert_dataframe(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Converts the feature columns of the dataframe into a natural language text column.
    Maps target labels dynamically (Certified -> 1, Denied -> 0).
    """
    target_col = schema["target_column"]
    
    texts = []
    labels = []
    
    # Dynamic target label mapping
    # Create target map based on actual values in column
    unique_targets = df[target_col].dropna().unique()
    print(f"Target column '{target_col}' unique values: {unique_targets}")
    
    target_mapping = {}
    if "Certified" in unique_targets:
        target_mapping["Certified"] = 1
    if "Denied" in unique_targets:
        target_mapping["Denied"] = 0
    
    # Fallback/dynamic target mapping if values are different
    for t in unique_targets:
        if t not in target_mapping:
            # Map values that contain positive context to 1, rest to 0
            if any(pos in str(t).lower() for pos in ["cert", "approve", "yes", "true", "1"]):
                target_mapping[t] = 1
            else:
                target_mapping[t] = 0
                
    print(f"Target label mapping: {target_mapping}")
    
    # Drop columns that are not features (like case_id, target_column)
    drop_cols = schema.get("drop_columns", [])
    feature_cols = [col for col in df.columns if col not in drop_cols and col != target_col]
    
    for idx, row in df.iterrows():
        # Convert row series to dictionary
        row_dict = row[feature_cols].to_dict()
        text_rep = convert_row_to_text(row_dict)
        texts.append(text_rep)
        
        raw_label = row[target_col]
        labels.append(target_mapping.get(raw_label, 0))
        
    converted_df = pd.DataFrame({
        "text": texts,
        "label": labels
    })
    return converted_df

def create_splits(df: pd.DataFrame, seed: int = 42) -> tuple:
    """
    Splits the dataframe into:
    - 70% Train
    - 15% Validation
    - 15% Test
    Using stratified splitting and a fixed random seed.
    """
    # 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label"]
    )
    
    # 15% validation, 15% test (50% of the 30% temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"]
    )
    
    # Verify no overlaps
    train_ids = set(train_df.index)
    val_ids = set(val_df.index)
    test_ids = set(test_df.index)
    
    assert len(train_ids.intersection(val_ids)) == 0, "Overlap found between train and validation sets!"
    assert len(train_ids.intersection(test_ids)) == 0, "Overlap found between train and test sets!"
    assert len(val_ids.intersection(test_ids)) == 0, "Overlap found between validation and test sets!"
    
    return train_df, val_df, test_df
