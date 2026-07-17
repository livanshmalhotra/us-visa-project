import os
import sys
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add project root and slm dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

load_dotenv()

from slm.src.inference import SLMPredictor

app = FastAPI(title="US Visa Approval Prediction - SLM Dashboard")

# Initialize predictor lazily
predictor = None
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model"))

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = SLMPredictor(model_dir)
        except Exception as e:
            raise RuntimeError(f"Could not load SLM model: {e}. Please ensure it is trained.")
    return predictor

# HTML Template with gorgeous glassmorphism layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>US Visa Approval - SLM Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #070913;
            --card-bg: rgba(13, 19, 36, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-primary: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
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
                radial-gradient(at 20% 20%, rgba(168, 85, 247, 0.12) 0px, transparent 40%),
                radial-gradient(at 80% 80%, rgba(59, 130, 246, 0.12) 0px, transparent 40%);
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}

        header {{
            background: rgba(8, 10, 24, 0.6);
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

        .header-badge {{
            background: rgba(168, 85, 247, 0.1);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.2);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
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
            padding: 2.2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
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
            background: rgba(255, 255, 255, 0.02);
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
            border-color: #a855f7;
            background: rgba(255, 255, 255, 0.05);
            box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.15);
        }}

        .btn {{
            cursor: pointer;
            padding: 0.85rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            border: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            text-decoration: none;
        }}

        .btn-primary {{
            background: var(--accent-primary);
            color: white;
            box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3);
        }}

        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(168, 85, 247, 0.4);
        }}

        .right-column {{
            display: flex;
            flex-direction: column;
            gap: 2.5rem;
        }}

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

        .text-preview {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.2rem;
            font-family: inherit;
            font-size: 0.95rem;
            line-height: 1.6;
            color: #d1d5db;
            margin-top: 0.5rem;
            position: relative;
        }}

        .text-preview-header {{
            font-size: 0.8rem;
            color: #c084fc;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }}

        .model-spec-list {{
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            margin-top: 1rem;
        }}

        .spec-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
        }}

        .spec-name {{
            color: var(--text-secondary);
        }}

        .spec-val {{
            font-weight: 600;
            color: #c084fc;
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
    </style>
</head>
<body>
    <header>
        <div class="logo">US Visa Approval SLM System</div>
        <div class="header-badge">Model: bert-tiny</div>
    </header>

    <main>
        <!-- Form Section -->
        <section class="glass-card">
            <h2>
                <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                Structured Candidate Parameters
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
                        <button type="submit" class="btn btn-primary" style="width: 100%;">Evaluate Visa using SLM</button>
                    </div>
                </div>
            </form>
        </section>

        <!-- Prediction & Dashboard Column -->
        <div class="right-column">
            <!-- Converted Text Preview -->
            <section class="glass-card">
                <h2>
                    <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h0m-4-8a3 3 0 013-3h0a3 3 0 013 3v0a3 3 0 01-3 3H8z"></path></svg>
                    SLM Text Input Representation
                </h2>
                <div class="text-preview-header">Generated Context Template</div>
                {converted_text_html}
            </section>

            <!-- Prediction Output -->
            <section class="glass-card result-container">
                <h2>
                    <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"></path></svg>
                    SLM Model Prediction
                </h2>
                {result_card_html}
            </section>

            <!-- Model Specifications -->
            <section class="glass-card">
                <h2>
                    <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    SLM Model Specifications
                </h2>
                <div class="model-spec-list">
                    <div class="spec-item">
                        <span class="spec-name">Base Model Architecture</span>
                        <span class="spec-val">prajjwal1/bert-tiny</span>
                    </div>
                    <div class="spec-item">
                        <span class="spec-name">Parameters Count</span>
                        <span class="spec-val">~4.4M parameters</span>
                    </div>
                    <div class="spec-item">
                        <span class="spec-name">Sequence Length</span>
                        <span class="spec-val">128 tokens</span>
                    </div>
                    <div class="spec-item">
                        <span class="spec-name">Inference Method</span>
                        <span class="spec-val">Softmax Probability (Binary)</span>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <footer>
        US Visa Status Prediction - Small Language Model Experiment &bull; MLOps Showcase
    </footer>
</body>
</html>
"""

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
    "message_html": "", 
    "converted_text_html": '<div class="text-preview">Enter application features and submit the form to see the generated text representation.</div>',
    "result_card_html": '<div class="result-placeholder">Pending candidate inputs...</div>'
}

def render_form(inputs_dict=None, error_msg=None, result_dict=None) -> HTMLResponse:
    render_dict = DEFAULT_WEB_VALS.copy()
    
    if inputs_dict:
        # Populate values
        render_dict["val_no_of_employees"] = str(inputs_dict.get("no_of_employees", 50))
        render_dict["val_yr_of_estab"] = str(inputs_dict.get("yr_of_estab", 2000))
        render_dict["val_prevailing_wage"] = str(inputs_dict.get("prevailing_wage", 60000.0))
        
        # Mark selected dropdown options
        render_dict[f"sel_continent_{inputs_dict.get('continent', 'Asia').replace(' ', '_')}"] = "selected"
        edu_key = inputs_dict.get('education_of_employee', 'Bachelors').replace("'", "").split(' ')[0]
        render_dict[f"sel_education_{edu_key}"] = "selected"
        render_dict[f"sel_experience_{inputs_dict.get('has_job_experience', 'Y')}"] = "selected"
        render_dict[f"sel_training_{inputs_dict.get('requires_job_training', 'N')}"] = "selected"
        render_dict[f"sel_region_{inputs_dict.get('region_of_employment', 'West')}"] = "selected"
        render_dict[f"sel_unit_{inputs_dict.get('unit_of_wage', 'Year')}"] = "selected"
        render_dict[f"sel_fulltime_{inputs_dict.get('full_time_position', 'Y')}"] = "selected"
        
    if error_msg:
        render_dict["message_html"] = f'<div class="alert-error"><strong>Error:</strong> {error_msg}</div>'

    if result_dict:
        pred = result_dict["prediction"]
        conf = result_dict["confidence"] * 100
        converted_text = result_dict["converted_text"]
        
        render_dict["converted_text_html"] = f'<div class="text-preview">"{converted_text}"</div>'
        
        card_class = "result-certified" if pred == "Certified" else "result-denied"
        
        render_dict["result_card_html"] = f"""
        <div class="result-card {card_class}">
            <div class="result-title">{pred}</div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.8;">Inference executed in PyTorch</div>
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
        slm_predictor = get_predictor()
        result = slm_predictor.predict_structured(form_inputs)
        return render_form(inputs_dict=form_inputs, result_dict=result)
    except Exception as e:
        return render_form(inputs_dict=form_inputs, error_msg=str(e))

if __name__ == "__main__":
    print("Starting Standalone SLM Presentation Web Dashboard on Port 8081...")
    uvicorn.run(app, host="127.0.0.1", port=8081)
