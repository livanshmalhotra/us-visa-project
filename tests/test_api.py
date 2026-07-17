from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_validation():
    # Invalid: negative no_of_employees
    payload = {
        "continent": "Asia",
        "education_of_employee": "Bachelor's",
        "has_job_experience": "Y",
        "requires_job_training": "N",
        "no_of_employees": -5, 
        "yr_of_estab": 2000,
        "region_of_employment": "West",
        "prevailing_wage": 60000.0,
        "unit_of_wage": "Year",
        "full_time_position": "Y"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Number of employees must be a non-negative number" in response.json()["detail"]
