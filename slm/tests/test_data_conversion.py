import os
import sys
import pandas as pd
import pytest

# Add project root and slm dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from slm.src.data_conversion import convert_row_to_text, load_schema, convert_dataframe

def test_load_schema():
    schema = load_schema()
    assert "columns" in schema
    assert "target_column" in schema
    assert schema["target_column"] == "case_status"

def test_convert_row_to_text():
    mock_row = {
        "continent": "Asia",
        "education_of_employee": "Master's",
        "has_job_experience": "Y",
        "requires_job_training": "N",
        "no_of_employees": 150,
        "yr_of_estab": 2005,
        "region_of_employment": "Northeast",
        "prevailing_wage": 120000.0,
        "unit_of_wage": "Year",
        "full_time_position": "Y"
    }
    
    text = convert_row_to_text(mock_row)
    
    assert "Asia" in text
    assert "Master's" in text
    assert "has previous job experience" in text
    assert "does not require job training" in text
    assert "150 employees" in text
    assert "the year 2005" in text
    assert "Northeast region" in text
    assert "120000.00" in text
    assert "is a full-time position" in text

def test_convert_row_to_text_missing_values():
    # Test handling of missing values / NaN values
    mock_row_missing = {
        "continent": "Europe",
        "education_of_employee": "Doctorate",
        "has_job_experience": "N",
        "requires_job_training": "Y",
        "no_of_employees": None,  # Missing
        "yr_of_estab": float("nan"),  # Missing
        "region_of_employment": "West",
        "prevailing_wage": None,  # Missing
        "unit_of_wage": "Hour",
        "full_time_position": "N"
    }
    
    text = convert_row_to_text(mock_row_missing)
    
    assert "Europe" in text
    assert "Doctorate" in text
    assert "does not have job experience" in text
    assert "requires job training" in text
    assert "an unspecified number of" in text
    assert "an unknown year" in text
    assert "an unspecified wage" in text
    assert "is not a full-time position" in text
