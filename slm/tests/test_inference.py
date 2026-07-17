import os
import sys
import pytest
from unittest.mock import patch

# Add project root and slm dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from slm.src.inference import SLMPredictor

@patch('slm.src.inference.SLMPredictor.__init__', return_value=None)
def test_predictor_input_validation(mock_init):
    # Setup dummy predictor instance
    predictor = SLMPredictor("dummy_path")
    
    # Valid row
    valid_row = {
        "continent": "Asia",
        "education_of_employee": "Bachelor's",
        "has_job_experience": "Y",
        "requires_job_training": "N",
        "no_of_employees": 100,
        "yr_of_estab": 2000,
        "region_of_employment": "West",
        "prevailing_wage": 50000.0,
        "unit_of_wage": "Year",
        "full_time_position": "Y"
    }
    
    # We mock predict_text as we are only testing the validation and convert_row_to_text call in predict_structured
    with patch.object(predictor, 'predict_text', return_value={"prediction": "Certified", "confidence": 0.9}):
        res = predictor.predict_structured(valid_row)
        assert res["prediction"] == "Certified"
        assert "converted_text" in res
        
    # Invalid employees count (negative)
    invalid_employees = valid_row.copy()
    invalid_employees["no_of_employees"] = -10
    with pytest.raises(ValueError, match="Number of employees must be a non-negative number."):
        predictor.predict_structured(invalid_employees)
        
    # Invalid prevailing wage (negative)
    invalid_wage = valid_row.copy()
    invalid_wage["prevailing_wage"] = -100.0
    with pytest.raises(ValueError, match="Prevailing wage must be a non-negative number."):
        predictor.predict_structured(invalid_wage)
        
    # Invalid establishment year
    invalid_year = valid_row.copy()
    invalid_year["yr_of_estab"] = 1799
    with pytest.raises(ValueError, match="Year of establishment must be between 1800 and 2026."):
        predictor.predict_structured(invalid_year)
        
    # Invalid establishment year future
    invalid_year_future = valid_row.copy()
    invalid_year_future["yr_of_estab"] = 2027
    with pytest.raises(ValueError, match="Year of establishment must be between 1800 and 2026."):
        predictor.predict_structured(invalid_year_future)
