import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_tokenizer(model_name_or_path: str):
    """
    Loads and returns the tokenizer for the given model name or path.
    Bypasses auto-tokenizer validation failures on Windows by falling back to standard tokenizer classes.
    """
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    except Exception as auto_err:
        print(f"AutoTokenizer failed with error: {auto_err}. Falling back to standard class tokenizers...")
        model_name_lower = str(model_name_or_path).lower()
        if "bert" in model_name_lower:
            from transformers import BertTokenizer
            return BertTokenizer.from_pretrained(model_name_or_path)
        elif "distilbert" in model_name_lower:
            from transformers import DistilBertTokenizer
            return DistilBertTokenizer.from_pretrained(model_name_or_path)
        else:
            raise auto_err

def get_model(model_name_or_path: str, num_labels: int = 2):
    """
    Loads and returns the sequence classification model.
    Handles legacy checkpoints lacking 'model_type' configs by falling back to specific class initializers.
    """
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels
        )
    except Exception as auto_err:
        print(f"AutoModelForSequenceClassification failed: {auto_err}. Falling back to specific model classes...")
        model_name_lower = str(model_name_or_path).lower()
        if "bert" in model_name_lower:
            from transformers import BertForSequenceClassification
            return BertForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=num_labels
            )
        elif "distilbert" in model_name_lower:
            from transformers import DistilBertForSequenceClassification
            return DistilBertForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=num_labels
            )
        else:
            raise auto_err

def get_model_and_tokenizer(model_name_or_path: str, num_labels: int = 2):
    """
    Loads and returns a pretrained Tokenizer and Sequence Classification Model.
    """
    tokenizer = get_tokenizer(model_name_or_path)
    model = get_model(model_name_or_path, num_labels=num_labels)
    return model, tokenizer
