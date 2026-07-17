import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from slm.src.data_conversion import convert_row_to_text

class SLMPredictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        if not os.path.exists(model_dir) or not os.path.exists(os.path.join(model_dir, "config.json")):
            raise FileNotFoundError(f"Trained SLM model not found at {model_dir}. Please train the model first.")
            
        print(f"Loading SLM model and tokenizer from {model_dir}...")
        from slm.src.model import get_tokenizer, get_model
        self.tokenizer = get_tokenizer(model_dir)
        self.model = get_model(model_dir)
        self.model.eval()
        
        # Check device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict_text(self, text: str) -> dict:
        """
        Runs SLM inference directly on a raw text string.
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_class = torch.argmax(logits, dim=-1).item()
            
        confidence = probs[pred_class].item()
        
        # Binary target mapping
        label = "Certified" if pred_class == 1 else "Denied"
        
        return {
            "prediction": label,
            "confidence": confidence,
            "probability_certified": probs[1].item(),
            "probability_denied": probs[0].item()
        }

    def predict_structured(self, row: dict) -> dict:
        """
        Validates, converts a structured input dictionary to text, and runs SLM prediction.
        """
        # Validate data types and boundaries
        no_emp = row.get("no_of_employees")
        if no_emp is not None and (not isinstance(no_emp, (int, float)) or no_emp < 0):
            raise ValueError("Number of employees must be a non-negative number.")
            
        wage = row.get("prevailing_wage")
        if wage is not None and (not isinstance(wage, (int, float)) or wage < 0):
            raise ValueError("Prevailing wage must be a non-negative number.")
            
        yr_estab = row.get("yr_of_estab")
        if yr_estab is not None and (not isinstance(yr_estab, int) or yr_estab < 1800 or yr_estab > 2026):
            raise ValueError("Year of establishment must be between 1800 and 2026.")
            
        # Convert to text
        converted_text = convert_row_to_text(row)
        
        # Run inference
        prediction_result = self.predict_text(converted_text)
        prediction_result["converted_text"] = converted_text
        
        return prediction_result
