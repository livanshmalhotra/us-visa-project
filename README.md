# US Visa Approval Status Prediction (MLOps Production-Ready Project)

This repository implements an end-to-end, production-grade machine learning project for predicting US Visa approval outcomes ("Certified" vs. "Denied") based on employee demographics and employer attributes.

The architecture and workflow are designed for scalability, reproducibility, and deployment readiness, following professional MLOps best practices.

---

## 🏗️ Project Architecture & Data Flow

The project is structured with a modular architecture that separates configuration, execution pipeline phases, and serving layers:

```text
us-visa-project/
├── config/
│   ├── model.yaml              # Hyperparameter tuning grids for GridSearchCV
│   └── schema.yaml             # Column definitions, types, and target variable
├── models/
│   └── model.pkl               # Pushed active production model (preprocessing + estimator)
├── src/us_visa/                # Core Python Package
│   ├── components/
│   │   ├── data_ingestion.py   # Seeding & pulling raw data from MongoDB Atlas
│   │   ├── data_validation.py  # Schema matching & evidently data drift analysis
│   │   ├── data_transformation.py # ColumnTransformer scaling, OHE, and SMOTE balancing
│   │   ├── model_trainer.py    # Training, Grid Search, & F1 evaluation
│   │   ├── model_evaluation.py # Production vs. candidate model comparison
│   │   └── model_pusher.py     # Pushing approved model to production folder
│   ├── configuration/
│   │   └── mongo_db_connection.py # TLS certified connection to MongoDB Atlas
│   ├── constants/              # Global variable paths & constants
│   ├── entity/                 # Config inputs & artifact output definitions
│   ├── exception/              # Traceback detailed custom error handler
│   ├── logger/                 # Output loggers (to stdout and timestamped files)
│   ├── pipline/                # TrainPipeline and PredictionPipeline orchestrations
│   └── utils/                  # Safe read/write YAML, save/load numpy arrays & objects
├── tests/
│   ├── test_api.py             # Integration endpoints & validation boundary checks
│   └── test_pipeline.py        # Unit checks for schemas and constants
├── app.py                      # FastAPI web dashboard & prediction endpoint
├── Dockerfile                  # Production container configuration
├── requirements.txt            # Package dependencies
└── .env                        # Secret environment variables (MongoDB URLs)
```

### 🔁 Preprocessing & Prediction Lifecycle
1. **User Input / API Payload**: User supplies raw features (e.g. `no_of_employees: 150`, `yr_of_estab: 2005`, `prevailing_wage: 120000.0`, `education_of_employee: "Master's"`).
2. **Data Boundary Validation**: Input data is parsed and validated (e.g. non-negative numeric constraints, establishment year boundary check `1800 <= year <= current_year`).
3. **Feature Engineering**: Calculates `company_age = current_year - yr_of_estab` dynamically and drops `yr_of_estab` and `case_id`.
4. **Column Transformation**: 
   - Numerical features (`no_of_employees`, `prevailing_wage`, `company_age`) are scaled using `StandardScaler`.
   - Categorical features are encoded using `OneHotEncoder(handle_unknown='ignore')`.
5. **Model Inference**: The transformed feature array is fed into the best trained estimator (Random Forest Classifier). F1 score metrics are checked, and the classification result (0 -> Denied, 1 -> Certified) with probability/confidence is returned.
6. **MongoDB Audit Logging**: The input features, prediction result, confidence, timestamp, and model version are inserted into the MongoDB Atlas `US_VISA.predictions` collection.

---

## 🚀 Setup & Execution Guide

Follow these exact commands to set up the environment, run the pipeline, run tests, and start the dashboard on Windows using PowerShell.

### 1. Environment Creation
Create a new Python 3.11 virtual environment:
```powershell
python -m venv venv
```

### 2. Activation
Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
Install all package requirements and link the local package in editable mode:
```powershell
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the root directory and add your MongoDB connection string:
```env
MONGODB_URL=mongodb+srv://<username>:<password>@cluster0.emw5txk.mongodb.net/?appName=Cluster0
```

### 5. Run the Training Pipeline
Execute the full MLOps lifecycle to seed data, validate, preprocess, tune hyperparameters, and save the model:
```powershell
python -c "from us_visa.pipline.training_pipeline import TrainPipeline; TrainPipeline().run_pipeline()"
```

### 6. Run Automated Tests
Execute the unit and API integration tests:
```powershell
python -m pytest
```

### 7. Run the Web Application
Start the FastAPI server locally:
```powershell
python app.py
```
Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser to access the premium presentation-ready prediction dashboard!

---

## 📊 Verification Metrics & Live API Details

### API Endpoints
- **GET `/health`**: Returns `{"status": "healthy"}`
- **POST `/predict`**: Predicts approval status.
  - *Example Request:*
    ```json
    {
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
    ```
  - *Example Response:*
    ```json
    {
      "success": true,
      "prediction": "Certified",
      "confidence": 0.8094,
      "model_version": "1.0.0"
    }
    ```

### Model Performance Metrics
- **Selected Classifier**: Random Forest Classifier
- **Parameters**: `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}`
- **Test F1 Score**: `79.8%`
- **Test Precision**: `81.9%`
- **Test Recall**: `77.8%`
