# US Visa Approval Status Prediction

> **A dual-model MLOps project** combining a production-grade traditional Machine Learning pipeline with a separate Small Language Model (SLM) experimentation pipeline, both predicting US Visa approval outcomes (*"Certified"* vs. *"Denied"*).

---

## 📁 Project Structure

```text
us-visa-project/
│
├── src/us_visa/                     # ── TRADITIONAL ML PACKAGE ──
│   ├── components/
│   │   ├── data_ingestion.py        # Seeds & pulls raw data from MongoDB Atlas
│   │   ├── data_validation.py       # Schema matching & Evidently drift analysis
│   │   ├── data_transformation.py   # ColumnTransformer, OHE, SMOTE balancing
│   │   ├── model_trainer.py         # Training, GridSearchCV & F1 evaluation
│   │   ├── model_evaluation.py      # Production vs. candidate model comparison
│   │   └── model_pusher.py          # Pushes approved model to production folder
│   ├── configuration/
│   │   └── mongo_db_connection.py   # TLS-certified MongoDB Atlas connection
│   ├── constants/                   # Global path variables & constants
│   ├── entity/                      # Config input & artifact output definitions
│   ├── exception/                   # Custom traceback error handler
│   ├── logger/                      # Timestamped log files & stdout loggers
│   ├── pipline/                     # TrainPipeline & PredictionPipeline
│   └── utils/                       # YAML, numpy, pickle helpers
│
├── config/
│   ├── model.yaml                   # Hyperparameter tuning grids (GridSearchCV)
│   └── schema.yaml                  # Column definitions & target variable
│
├── models/
│   └── model.pkl                    # Active production model (preprocessing + RF)
│
├── data/                            # Raw structured tabular dataset
│
├── tests/
│   ├── test_api.py                  # Integration endpoint & validation tests
│   └── test_pipeline.py             # Unit checks for schemas and constants
│
├── app.py                           # Traditional ML FastAPI dashboard (port 8080)
├── Dockerfile                       # Production container configuration
├── requirements.txt                 # Traditional ML dependencies
├── .env                             # Secret environment variables
│
└── slm/                             # ── SLM EXPERIMENTATION PIPELINE ──
    ├── data/
    │   ├── raw/                     # Placeholder for raw SLM data
    │   ├── processed/               # Placeholder for processed data
    │   └── text_dataset/
    │       ├── train.csv            # 17,836 natural language training samples
    │       ├── validation.csv       # 3,822 validation samples
    │       └── test.csv             # 3,822 test samples
    │
    ├── models/
    │   ├── best_model/              # Fine-tuned bert-tiny checkpoint & tokenizer
    │   ├── metrics.json             # Test set evaluation metrics
    │   ├── confusion_matrix.png     # Confusion matrix visualization
    │   └── model_comparison.json    # Side-by-side RF vs. SLM metrics
    │
    ├── src/
    │   ├── data_conversion.py       # Tabular-to-natural-language conversion
    │   ├── dataset.py               # PyTorch Dataset class for tokenized inputs
    │   ├── model.py                 # Tokenizer & model loader with fallbacks
    │   ├── train.py                 # HuggingFace Trainer fine-tuning loop
    │   ├── evaluate.py              # Metrics computation & confusion matrix
    │   └── inference.py             # SLMPredictor class for structured inputs
    │
    ├── scripts/
    │   ├── convert_tabular_to_text.py  # Data conversion CLI
    │   ├── train_slm.py             # Fine-tuning CLI
    │   ├── evaluate_slm.py          # Evaluation CLI
    │   ├── predict_slm.py           # Interactive prediction CLI
    │   ├── compare.py               # Traditional ML vs. SLM comparison script
    │   └── app.py                   # SLM FastAPI dashboard (port 8081)
    │
    ├── tests/
    │   ├── test_data_conversion.py  # Text formatting validation tests
    │   └── test_inference.py        # Prediction boundary & input tests
    │
    ├── requirements.txt             # SLM-specific dependencies
    └── README.md                    # Detailed SLM pipeline documentation
```

---

## 🤖 Model 1 — Traditional ML: Random Forest Classifier

### Overview
A production-grade supervised binary classification pipeline built with scikit-learn, designed for reliability, speed, and interpretability. The model is trained on structured tabular data and served via a FastAPI web dashboard.

### How It Works
1. **Data Ingestion**: Seeds and pulls 25,480 records from MongoDB Atlas (`US_VISA.visa_data` collection).
2. **Data Validation**: Schema checks and Evidently drift analysis against a baseline dataset.
3. **Data Transformation**: 
   - Drops `case_id`, `yr_of_estab` → computes `company_age = current_year - yr_of_estab`
   - `StandardScaler` on numerical features (`no_of_employees`, `prevailing_wage`, `company_age`)
   - `OneHotEncoder(handle_unknown='ignore')` on categorical features
   - `SMOTE` oversampling for class balance
4. **Model Training**: `RandomForestClassifier` with `GridSearchCV` for hyperparameter tuning
5. **Model Evaluation**: Compares candidate vs. production model using F1 threshold gating
6. **Model Pushing**: Saves best `USVisaModel` wrapper (preprocessing + estimator) to `models/model.pkl`

### Performance Metrics (on 3,822 test records)

| Metric | Score |
|---|---|
| **Accuracy** | 73.86% |
| **F1-Score** | 79.91% |
| **Precision** | 82.07% |
| **Recall** | 77.86% |
| **ROC-AUC** | 80.32% |
| Training Time | ~12.5 seconds |
| Inference Latency | 0.045 ms/record |
| Model Size | 5.24 MB |

### Running the Traditional ML Pipeline

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
# Create .env file with your MongoDB connection string:
# MONGODB_URL=mongodb+srv://<username>:<password>@cluster0.emw5txk.mongodb.net/?appName=Cluster0

# 4. Run the full training pipeline
python -c "from us_visa.pipline.training_pipeline import TrainPipeline; TrainPipeline().run_pipeline()"

# 5. Run automated tests
python -m pytest tests/

# 6. Start the web dashboard (port 8080)
python app.py
```
Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** to access the glassmorphism prediction dashboard.

### API Endpoints (Traditional ML)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "healthy"}` |
| `GET` | `/` | Interactive HTML prediction form |
| `POST` | `/` | Form submission prediction |
| `POST` | `/predict` | JSON REST API prediction |
| `POST` | `/train` | Trigger full training pipeline |

**Example REST request:**
```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "continent": "Asia",
    "education_of_employee": "Master'\''s",
    "has_job_experience": "Y",
    "requires_job_training": "N",
    "no_of_employees": 150,
    "yr_of_estab": 2005,
    "region_of_employment": "Northeast",
    "prevailing_wage": 120000.0,
    "unit_of_wage": "Year",
    "full_time_position": "Y"
  }'
```

---

## 🧠 Model 2 — SLM: prajjwal1/bert-tiny (Fine-tuned)

### Overview
A completely independent Small Language Model (SLM) experimentation pipeline that converts structured tabular visa records into natural language descriptions and fine-tunes a `prajjwal1/bert-tiny` transformer for binary sequence classification. This pipeline is **fully isolated** from the traditional ML codebase.

### How It Works
1. **Tabular-to-Text Conversion**: Each structured record is converted to a natural language sentence:
   > *"An applicant from Asia has an education level of Master's. The applicant has previous job experience and does not require job training. The employer was established in the year 2005 and has 150 employees. The position is located in the Northeast region, pays a prevailing wage of 120000.00 per year, and is a full-time position."*
2. **Stratified Split**: 70% train / 15% validation / 15% test (seed=42, no overlap guaranteed)
3. **Fine-tuning**: `BertForSequenceClassification` (bert-tiny backbone) trained for 3 epochs using Hugging Face `Trainer` with early stopping on validation F1
4. **Evaluation**: Accuracy, F1, Precision, Recall, ROC-AUC computed on unseen test set
5. **Inference**: `SLMPredictor` accepts structured dicts, converts to text, tokenizes, and returns classification probabilities

### Model Architecture

| Property | Value |
|---|---|
| **Base Model** | `prajjwal1/bert-tiny` |
| **Architecture** | BERT (2 transformer layers, 128 hidden dims) |
| **Task** | Binary Sequence Classification |
| **Parameters** | ~4.4 Million |
| **Max Sequence Length** | 128 tokens |
| **Tokenizer** | WordPiece (`BertTokenizer`) |
| **Training Epochs** | 3 |
| **Learning Rate** | 5e-5 |
| **Batch Size** | 16 |

### Performance Metrics (on 3,822 test records)

| Metric | Score |
|---|---|
| **Accuracy** | 74.59% |
| **F1-Score** | 82.14% |
| **Precision** | 77.40% |
| **Recall** | 87.50% |
| **ROC-AUC** | 76.62% |
| Training Time | ~841 seconds (CPU) |
| Inference Latency | 2.983 ms/record |
| Model Size | 17.42 MB |

### Running the SLM Pipeline

```powershell
# Activate the virtual environment (same venv as traditional ML)
.\venv\Scripts\Activate.ps1

# Step 1 — Convert tabular records to natural language text splits
venv\Scripts\python.exe slm/scripts/convert_tabular_to_text.py

# Step 2 — Fine-tune bert-tiny on the text dataset
venv\Scripts\python.exe slm/scripts/train_slm.py --model_name prajjwal1/bert-tiny --epochs 3 --batch_size 16 --learning_rate 5e-5

# Step 3 — Evaluate the trained SLM on the test set
venv\Scripts\python.exe slm/scripts/evaluate_slm.py

# Step 4 — Compare Traditional ML vs. SLM side-by-side
venv\Scripts\python.exe slm/scripts/compare.py

# Step 5 — Run interactive CLI predictions
venv\Scripts\python.exe slm/scripts/predict_slm.py

# Step 6 — Start the SLM web dashboard (port 8081)
venv\Scripts\python.exe slm/scripts/app.py

# Step 7 — Run SLM unit tests
venv\Scripts\python.exe -m pytest slm/tests/
```
Open **[http://127.0.0.1:8081](http://127.0.0.1:8081)** to access the SLM glassmorphism prediction dashboard.

---

## 📊 Head-to-Head Model Comparison

Both models were evaluated on the **exact same 3,822 test records**:

| Metric | Random Forest (Traditional ML) | bert-tiny (SLM) | Winner |
|---|---|---|---|
| **Accuracy** | 73.86% | **74.59%** | 🧠 SLM |
| **F1-Score** | 79.91% | **82.14%** | 🧠 SLM |
| **Precision** | **82.07%** | 77.40% | 🌲 RF |
| **Recall** | 77.86% | **87.50%** | 🧠 SLM |
| **ROC-AUC** | **80.32%** | 76.62% | 🌲 RF |
| **Training Time** | **~12.5 s** | 841 s | 🌲 RF |
| **Inference Latency** | **0.045 ms** | 2.983 ms | 🌲 RF |
| **Model Size** | **5.24 MB** | 17.42 MB | 🌲 RF |
| **Parameters** | ~100 trees | 4.4 M params | — |

> **Interpretation**: The SLM achieves higher overall Accuracy and F1-Score, with significantly better Recall (catching more genuine Certified cases). The Random Forest wins on Precision, ROC-AUC, speed, and model footprint — making it the right choice for production serving, while the SLM demonstrates the potential of language model approaches on tabular data.

---

## 🧪 Test Suite Results

```text
# Traditional ML Tests
tests\test_api.py ..          [ 50%]
tests\test_pipeline.py ..     [100%]
4 passed in 11.91s ✅

# SLM Tests
slm\tests\test_data_conversion.py ...   [ 75%]
slm\tests\test_inference.py .           [100%]
4 passed in 18.93s ✅
```

---

## 🌐 Web Dashboards

| Dashboard | URL | Technology |
|---|---|---|
| Traditional ML | [http://127.0.0.1:8080](http://127.0.0.1:8080) | FastAPI + glassmorphism dark UI |
| SLM Experiment | [http://127.0.0.1:8081](http://127.0.0.1:8081) | FastAPI + glassmorphism dark UI |

---

## 🛠️ Technology Stack

| Layer | Traditional ML | SLM |
|---|---|---|
| **Language** | Python 3.11 | Python 3.11 |
| **ML Framework** | scikit-learn | PyTorch + HuggingFace Transformers |
| **Model** | RandomForestClassifier | prajjwal1/bert-tiny |
| **Data Source** | MongoDB Atlas | Local artifact CSV |
| **Serving** | FastAPI (port 8080) | FastAPI (port 8081) |
| **Preprocessing** | ColumnTransformer + SMOTE | Natural language conversion |
| **Experiment Tracking** | Evidently drift reports | HuggingFace Trainer logs |
| **Testing** | pytest | pytest |
| **Containerization** | Docker | — |
