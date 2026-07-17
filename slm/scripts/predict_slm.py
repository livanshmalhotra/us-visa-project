import os
import sys
from dotenv import load_dotenv

# Add project root and slm dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

load_dotenv()

from slm.src.inference import SLMPredictor

def get_input(prompt: str, default: str) -> str:
    user_val = input(f"{prompt} [default: {default}]: ").strip()
    return user_val if user_val else default

def main():
    print("=== US Visa Approval prediction - SLM Model ===")
    
    # Resolve model directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(base_dir, "models", "best_model")
    
    try:
        predictor = SLMPredictor(model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running: python slm/scripts/train_slm.py")
        sys.exit(1)
        
    print("\nPlease enter the visa application features below:")
    
    # Demographics and company info input with fallback defaults
    continent = get_input("Continent of origin (Asia, Europe, North America, South America, Africa, Oceania)", "Asia")
    education = get_input("Education level (Bachelor's, Master's, High School, Doctorate)", "Bachelor's")
    has_exp = get_input("Job experience? (Y/N)", "Y").upper()
    req_train = get_input("Requires job training? (Y/N)", "N").upper()
    
    try:
        no_emp = int(get_input("Number of employees (Company Size)", "50"))
    except ValueError:
        print("Invalid company size. Using default: 50")
        no_emp = 50
        
    try:
        yr_estab = int(get_input("Year of establishment", "2000"))
    except ValueError:
        print("Invalid year. Using default: 2000")
        yr_estab = 2000
        
    region = get_input("Intended US Region of employment (West, Northeast, South, Midwest, Island)", "West")
    
    try:
        wage = float(get_input("Prevailing wage", "60000.00"))
    except ValueError:
        print("Invalid wage amount. Using default: 60000.00")
        wage = 60000.0
        
    wage_unit = get_input("Unit of wage (Year, Month, Week, Hour)", "Year")
    full_time = get_input("Full-time position? (Y/N)", "Y").upper()
    
    input_row = {
        "continent": continent,
        "education_of_employee": education,
        "has_job_experience": has_exp,
        "requires_job_training": req_train,
        "no_of_employees": no_emp,
        "yr_of_estab": yr_estab,
        "region_of_employment": region,
        "prevailing_wage": wage,
        "unit_of_wage": wage_unit,
        "full_time_position": full_time
    }
    
    try:
        result = predictor.predict_structured(input_row)
        
        print("\n" + "=" * 60)
        print("DYNAMICALLY CONVERTED NATURAL LANGUAGE TEXT:")
        print(f'"{result["converted_text"]}"')
        print("=" * 60)
        print(f"SLM PREDICTION VERDICT: {result['prediction']}")
        print(f"CONFIDENCE LEVEL      : {result['confidence'] * 100:.2f}%")
        print(f"  Probability Certified: {result['probability_certified'] * 100:.2f}%")
        print(f"  Probability Denied   : {result['probability_denied'] * 100:.2f}%")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
