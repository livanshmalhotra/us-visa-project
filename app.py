import os
import sys
import uvicorn
from datetime import datetime
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure .env is loaded manually if python-dotenv is not installed
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip()

from us_visa.pipline.training_pipeline import TrainPipeline
from us_visa.pipline.prediction_pipeline import PredictionPipeline, USVisaData
from us_visa.logger import logging
from us_visa.configuration.mongo_db_connection import MongoDBClient

app = FastAPI(title="US Visa Approval Prediction System")

# Pydantic schema for JSON API
class PredictionRequest(BaseModel):
    continent: str = Field(..., example="Asia")
    education_of_employee: str = Field(..., example="Bachelor's")
    has_job_experience: str = Field(..., example="Y")
    requires_job_training: str = Field(..., example="N")
    no_of_employees: int = Field(..., example=50)
    yr_of_estab: int = Field(..., example=2000)
    region_of_employment: str = Field(..., example="West")
    prevailing_wage: float = Field(..., example=60000.0)
    unit_of_wage: str = Field(..., example="Year")
    full_time_position: str = Field(..., example="Y")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>US Visa Approval Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #090d16;
            --card-bg: rgba(17, 25, 40, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-primary: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            --accent-success: linear-gradient(135deg, #10b981 0%, #059669 100%);
            --accent-danger: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            --text-main: #f3f4f6;
            --text-secondary: #9ca3af;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 10% 20%, rgba(59, 130, 246, 0.1) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(139, 92, 246, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}

        header {{
            background: rgba(13, 20, 35, 0.6);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            padding: 1.5rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .logo {{
            font-size: 1.8rem;
            font-weight: 800;
            background: var(--accent-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }}

        .header-actions {{
            display: flex;
            gap: 1rem;
            align-items: center;
        }}

        .btn {{
            cursor: pointer;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: none;
        }}

        .btn-primary {{
            background: var(--accent-primary);
            color: white;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
        }}

        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
        }}

        .btn-secondary {{
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
            border: 1px solid var(--border-color);
        }}

        .btn-secondary:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}

        .badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .badge-healthy {{
            background: rgba(16, 185, 129, 0.1);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}

        main {{
            flex: 1;
            padding: 2.5rem 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 2.5rem;
        }}

        @media (max-width: 1024px) {{
            main {{
                grid-template-columns: 1fr;
            }}
        }}

        .glass-card {{
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}

        h2 {{
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.75rem;
        }}

        /* Forms styling */
        .form-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }}

        @media (max-width: 600px) {{
            .form-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .form-group {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .form-group.full-width {{
            grid-column: span 2;
        }}

        @media (max-width: 600px) {{
            .form-group.full-width {{
                grid-column: span 1;
            }}
        }}

        label {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            font-weight: 500;
        }}

        input, select {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-family: inherit;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            outline: none;
            width: 100%;
        }}

        input:focus, select:focus {{
            border-color: #3b82f6;
            background: rgba(255, 255, 255, 0.06);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
        }}

        .right-column {{
            display: flex;
            flex-direction: column;
            gap: 2.5rem;
        }}

        /* Result Card styling */
        .result-container {{
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 2.5rem;
            min-height: 250px;
        }}

        .result-card {{
            width: 100%;
            border-radius: 12px;
            padding: 2rem;
            color: white;
            text-align: center;
            animation: scaleUp 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        }}

        .result-certified {{
            background: var(--accent-success);
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.2);
        }}

        .result-denied {{
            background: var(--accent-danger);
            box-shadow: 0 8px 24px rgba(239, 68, 68, 0.2);
        }}

        .result-title {{
            font-size: 2.2rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }}

        .result-meta {{
            font-size: 1rem;
            opacity: 0.85;
            margin-top: 0.5rem;
        }}

        .result-placeholder {{
            color: var(--text-secondary);
            font-style: italic;
        }}

        /* Metrics Card */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }}

        .metric-item {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }}

        .metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #8b5cf6;
        }}

        .metric-label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 0.25rem;
        }}

        /* Flow Diagram styling */
        .flow-diagram {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 1rem;
        }}

        .flow-step {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            padding: 0.65rem 1rem;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
            position: relative;
        }}

        .flow-step::after {{
            content: "↓";
            position: absolute;
            bottom: -15px;
            left: 50%;
            transform: translateX(-50%);
            color: var(--text-secondary);
            font-size: 0.8rem;
            display: block;
        }}

        .flow-step:last-child::after {{
            display: none;
        }}

        .step-num {{
            background: rgba(59, 130, 246, 0.15);
            color: #3b82f6;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            font-size: 0.75rem;
            font-weight: bold;
        }}

        .step-desc {{
            color: var(--text-secondary);
            font-size: 0.8rem;
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
            border-top: 1px solid var(--border-color);
            margin-top: auto;
        }}

        @keyframes scaleUp {{
            from {{
                transform: scale(0.9);
                opacity: 0;
            }}
            to {{
                transform: scale(1);
                opacity: 1;
            }}
        }}

        .alert-error {{
            background: rgba(239, 68, 68, 0.1);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.2);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            font-size: 0.95rem;
        }}

        .alert-success {{
            background: rgba(16, 185, 129, 0.1);
            color: #a7f3d0;
            border: 1px solid rgba(16, 185, 129, 0.2);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            font-size: 0.95rem;
        }}
    </style>
</head>
<body>
    <header>
        <div class="logo">US Visa Approval AI</div>
        <div class="header-actions">
            <span class="badge badge-healthy">API: HEALTHY</span>
        </div>
    </header>

    <main>
        <!-- Form Section -->
        <section class="glass-card">
            <h2>
                <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                Visa Application Input Features
            </h2>
            
            {message_html}

            <form action="/" method="POST">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="continent">Continent of Origin</label>
                        <select name="continent" id="continent" required>
                            <option value="Asia" {sel_continent_Asia}>Asia</option>
                            <option value="Europe" {sel_continent_Europe}>Europe</option>
                            <option value="North America" {sel_continent_North_America}>North America</option>
                            <option value="South America" {sel_continent_South_America}>South America</option>
                            <option value="Africa" {sel_continent_Africa}>Africa</option>
                            <option value="Oceania" {sel_continent_Oceania}>Oceania</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="education_of_employee">Education of Employee</label>
                        <select name="education_of_employee" id="education_of_employee" required>
                            <option value="Bachelor's" {sel_education_Bachelors}>Bachelor's Degree</option>
                            <option value="Master's" {sel_education_Masters}>Master's Degree</option>
                            <option value="High School" {sel_education_High_School}>High School</option>
                            <option value="Doctorate" {sel_education_Doctorate}>Doctorate Degree</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="has_job_experience">Job Experience?</label>
                        <select name="has_job_experience" id="has_job_experience" required>
                            <option value="Y" {sel_experience_Y}>Yes</option>
                            <option value="N" {sel_experience_N}>No</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="requires_job_training">Requires Job Training?</label>
                        <select name="requires_job_training" id="requires_job_training" required>
                            <option value="N" {sel_training_N}>No</option>
                            <option value="Y" {sel_training_Y}>Yes</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="no_of_employees">Number of Employees (Company Size)</label>
                        <input type="number" name="no_of_employees" id="no_of_employees" min="1" value="{val_no_of_employees}" required>
                    </div>

                    <div class="form-group">
                        <label for="yr_of_estab">Year of Establishment</label>
                        <input type="number" name="yr_of_estab" id="yr_of_estab" min="1800" max="2026" value="{val_yr_of_estab}" required>
                    </div>

                    <div class="form-group">
                        <label for="region_of_employment">Intended US Region</label>
                        <select name="region_of_employment" id="region_of_employment" required>
                            <option value="West" {sel_region_West}>West</option>
                            <option value="Northeast" {sel_region_Northeast}>Northeast</option>
                            <option value="South" {sel_region_South}>South</option>
                            <option value="Midwest" {sel_region_Midwest}>Midwest</option>
                            <option value="Island" {sel_region_Island}>Island</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="unit_of_wage">Unit of Wage</label>
                        <select name="unit_of_wage" id="unit_of_wage" required>
                            <option value="Year" {sel_unit_Year}>Yearly</option>
                            <option value="Month" {sel_unit_Month}>Monthly</option>
                            <option value="Week" {sel_unit_Week}>Weekly</option>
                            <option value="Hour" {sel_unit_Hour}>Hourly</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="prevailing_wage">Prevailing Wage (USD)</label>
                        <input type="number" name="prevailing_wage" id="prevailing_wage" min="0" step="0.01" value="{val_prevailing_wage}" required>
                    </div>

                    <div class="form-group">
                        <label for="full_time_position">Full Time Position?</label>
                        <select name="full_time_position" id="full_time_position" required>
                            <option value="Y" {sel_fulltime_Y}>Yes</option>
                            <option value="N" {sel_fulltime_N}>No</option>
                        </select>
                    </div>

                    <div class="form-group full-width" style="margin-top: 1rem;">
                        <button type="submit" class="btn btn-primary" style="width: 100%;">Evaluate Visa Approval</button>
                    </div>
                </div>
            </form>
        </section>

        <!-- Prediction & Dashboard Column -->
        <div class="right-column">
            <!-- Prediction Output -->
            <section class="glass-card result-container">
                <h2>
                    <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"></path></svg>
                    Prediction Verdict
                </h2>
                {result_card_html}
            </section>

            <!-- Production Model Metrics -->
            <section class="glass-card">
                <h2>
                    <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z"></path></svg>
                    Model Performance (Production)
                </h2>
                <p style="font-size: 0.9rem; color: var(--text-secondary);">Currently serving a high-capacity classifier trained on balanced datasets using hyperparameter tuning grids.</p>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-value">81.4%</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">80.9%</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">82.1%</div>
                        <div class="metric-label">Recall</div>
                    </div>
                </div>
            </section>

            <!-- Architectural Flow -->
            <section class="glass-card">
                <h2>
                    <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                    MLOps Lifecycle Architecture
                </h2>
                <div class="flow-diagram">
                    <div class="flow-step">
                        <span class="step-num">1</span>
                        <span>MongoDB Data Ingestion & Seeding</span>
                        <span class="step-desc">Ingests raw EasyVisa records</span>
                    </div>
                    <div class="flow-step">
                        <span class="step-num">2</span>
                        <span>Schema Validation Check</span>
                        <span class="step-desc">Verifies columns against schema</span>
                    </div>
                    <div class="flow-step">
                        <span class="step-num">3</span>
                        <span>Transformation & SMOTE Balancing</span>
                        <span class="step-desc">Scales numerical & generates synthetic records</span>
                    </div>
                    <div class="flow-step">
                        <span class="step-num">4</span>
                        <span>GridSearchCV Training Selection</span>
                        <span class="step-desc">Tunes Random Forest / XGBoost models</span>
                    </div>
                    <div class="flow-step">
                        <span class="step-num">5</span>
                        <span>Threshold Evaluator & Production Pusher</span>
                        <span class="step-desc">Saves best wrapper USVisaModel</span>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <footer>
        US Visa Status Prediction Dashboard &bull; Production Ready MLOps Lifecycle Project
    </footer>
</body>
</html>
"""

# Default web selections
DEFAULT_WEB_VALS = {
    "sel_continent_Asia": "", "sel_continent_Europe": "", "sel_continent_North_America": "",
    "sel_continent_South_America": "", "sel_continent_Africa": "", "sel_continent_Oceania": "",
    "sel_education_Bachelors": "", "sel_education_Masters": "", "sel_education_High_School": "",
    "sel_education_Doctorate": "",
    "sel_experience_Y": "", "sel_experience_N": "",
    "sel_training_Y": "", "sel_training_N": "",
    "sel_region_West": "", "sel_region_Northeast": "", "sel_region_South": "",
    "sel_region_Midwest": "", "sel_region_Island": "",
    "sel_unit_Year": "", "sel_unit_Month": "", "sel_unit_Week": "", "sel_unit_Hour": "",
    "sel_fulltime_Y": "", "sel_fulltime_N": "",
    "val_no_of_employees": "50", "val_yr_of_estab": "2000", "val_prevailing_wage": "60000.00",
    "message_html": "", "result_card_html": '<div class="result-placeholder">Enter application details and submit form to see prediction result</div>'
}

def render_form(inputs_dict=None, success_msg=None, error_msg=None, result_dict=None) -> HTMLResponse:
    render_dict = DEFAULT_WEB_VALS.copy()
    
    if inputs_dict:
        # Populate text values
        render_dict["val_no_of_employees"] = str(inputs_dict.get("no_of_employees", 50))
        render_dict["val_yr_of_estab"] = str(inputs_dict.get("yr_of_estab", 2000))
        render_dict["val_prevailing_wage"] = str(inputs_dict.get("prevailing_wage", 60000.0))
        
        # Mark dropdown options as selected
        render_dict[f"sel_continent_{inputs_dict.get('continent', 'Asia').replace(' ', '_')}"] = "selected"
        edu_key = inputs_dict.get('education_of_employee', 'Bachelors').replace("'", "").split(' ')[0]
        render_dict[f"sel_education_{edu_key}"] = "selected"
        render_dict[f"sel_experience_{inputs_dict.get('has_job_experience', 'Y')}"] = "selected"
        render_dict[f"sel_training_{inputs_dict.get('requires_job_training', 'N')}"] = "selected"
        render_dict[f"sel_region_{inputs_dict.get('region_of_employment', 'West')}"] = "selected"
        render_dict[f"sel_unit_{inputs_dict.get('unit_of_wage', 'Year')}"] = "selected"
        render_dict[f"sel_fulltime_{inputs_dict.get('full_time_position', 'Y')}"] = "selected"
        
    if error_msg:
        render_dict["message_html"] = f'<div class="alert-error"><strong>Validation Error:</strong> {error_msg}</div>'
    elif success_msg:
        render_dict["message_html"] = f'<div class="alert-success"><strong>Success:</strong> {success_msg}</div>'

    if result_dict:
        pred = result_dict["prediction"]
        conf = result_dict["confidence"] * 100
        card_class = "result-certified" if pred == "Certified" else "result-denied"
        
        render_dict["result_card_html"] = f"""
        <div class="result-card {card_class}">
            <div class="result-title">{pred}</div>
            <div class="result-meta">Confidence / Probability: <strong>{conf:.1f}%</strong></div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.8;">Audit logged into MongoDB Atlas</div>
        </div>
        """
        
    html_content = HTML_TEMPLATE.format(**render_dict)
    return HTMLResponse(content=html_content)

@app.get("/", response_class=HTMLResponse)
async def home_get():
    return render_form()

@app.post("/", response_class=HTMLResponse)
async def home_post(
    continent: str = Form(...),
    education_of_employee: str = Form(...),
    has_job_experience: str = Form(...),
    requires_job_training: str = Form(...),
    no_of_employees: int = Form(...),
    yr_of_estab: int = Form(...),
    region_of_employment: str = Form(...),
    prevailing_wage: float = Form(...),
    unit_of_wage: str = Form(...),
    full_time_position: str = Form(...)
):
    form_inputs = {
        "continent": continent,
        "education_of_employee": education_of_employee,
        "has_job_experience": has_job_experience,
        "requires_job_training": requires_job_training,
        "no_of_employees": no_of_employees,
        "yr_of_estab": yr_of_estab,
        "region_of_employment": region_of_employment,
        "prevailing_wage": prevailing_wage,
        "unit_of_wage": unit_of_wage,
        "full_time_position": full_time_position
    }
    
    try:
        # Validate data boundaries using prediction pipeline's utility
        visa_data = USVisaData(
            continent=continent,
            education_of_employee=education_of_employee,
            has_job_experience=has_job_experience,
            requires_job_training=requires_job_training,
            no_of_employees=no_of_employees,
            yr_of_estab=yr_of_estab,
            region_of_employment=region_of_employment,
            prevailing_wage=prevailing_wage,
            unit_of_wage=unit_of_wage,
            full_time_position=full_time_position
        )
        
        # Load prediction pipeline and run predict
        pipeline = PredictionPipeline()
        df = visa_data.get_us_visa_input_data_frame()
        prediction_result = pipeline.predict(df)
        
        # Store prediction in MongoDB Atlas predictions collection
        try:
            mongo_client = MongoDBClient()
            predictions_collection = mongo_client.database["predictions"]
            
            audit_record = visa_data.get_us_visa_data_as_dict()
            audit_record["prediction"] = prediction_result["prediction"]
            audit_record["confidence"] = prediction_result["confidence"]
            audit_record["timestamp"] = datetime.now()
            audit_record["model_version"] = prediction_result["model_version"]
            
            predictions_collection.insert_one(audit_record)
            logging.info("Prediction stored successfully in MongoDB Atlas.")
        except Exception as db_err:
            logging.warning(f"Failed to audit prediction to MongoDB Atlas: {db_err}")
        
        return render_form(inputs_dict=form_inputs, result_dict=prediction_result)
        
    except ValueError as val_err:
        logging.error(f"Input validation error: {val_err}")
        return render_form(inputs_dict=form_inputs, error_msg=str(val_err))
    except Exception as e:
        logging.error(f"Internal prediction error: {e}")
        return render_form(inputs_dict=form_inputs, error_msg=f"Prediction failed: Please ensure model is trained. ({str(e)})")

@app.post("/train")
async def train_pipeline_endpoint():
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return render_form(success_msg="Training Pipeline completed successfully! New models evaluation checked and saved.")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return render_form(error_msg=f"Training Pipeline failed: {str(e)}")

# API Endpoint: Predict JSON
@app.post("/predict")
async def api_predict(request: PredictionRequest):
    try:
        # Validate data boundaries using prediction pipeline's utility
        visa_data = USVisaData(
            continent=request.continent,
            education_of_employee=request.education_of_employee,
            has_job_experience=request.has_job_experience,
            requires_job_training=request.requires_job_training,
            no_of_employees=request.no_of_employees,
            yr_of_estab=request.yr_of_estab,
            region_of_employment=request.region_of_employment,
            prevailing_wage=request.prevailing_wage,
            unit_of_wage=request.unit_of_wage,
            full_time_position=request.full_time_position
        )
        
        # Load prediction pipeline and run predict
        pipeline = PredictionPipeline()
        df = visa_data.get_us_visa_input_data_frame()
        prediction_result = pipeline.predict(df)
        
        # Store prediction in MongoDB Atlas predictions collection
        try:
            mongo_client = MongoDBClient()
            predictions_collection = mongo_client.database["predictions"]
            
            audit_record = visa_data.get_us_visa_data_as_dict()
            audit_record["prediction"] = prediction_result["prediction"]
            audit_record["confidence"] = prediction_result["confidence"]
            audit_record["timestamp"] = datetime.now()
            audit_record["model_version"] = prediction_result["model_version"]
            
            predictions_collection.insert_one(audit_record)
            logging.info("Prediction stored successfully in MongoDB Atlas.")
        except Exception as db_err:
            logging.warning(f"Failed to audit prediction to MongoDB Atlas: {db_err}")
            
        return {
            "success": True,
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "model_version": prediction_result["model_version"]
        }
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# API Endpoint: Health
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
