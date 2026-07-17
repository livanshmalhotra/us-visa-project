# US Visa Approval Status Prediction: Small Language Model (SLM) Pipeline

This directory contains a complete, self-contained Small Language Model (SLM) experimentation and serving pipeline. It extends the traditional tabular machine learning workflow by transforming structured applicant records into descriptive natural language sentences and fine-tuning a pretrained transformer classifier.

---

## 🏗️ Architecture Comparison

### Traditional ML Pipeline
```text
Structured Input Data ➔ Preprocessing (One-Hot, Scaler, SMOTE) ➔ Random Forest Classifier ➔ Binary Verdict
```

### SLM Pipeline
```text
Structured Input Data ➔ Natural Language Converter ➔ Tokenizer ➔ Fine-tuned SLM (bert-tiny) ➔ Softmax ➔ Binary Verdict
```

---

## 📝 Tabular-to-Text Translation Template

To feed tabular data into a language model, each record is converted into a coherent, descriptive sentence using a deterministic schema-aware template:

**Structured Features:**
- `continent`: Asia
- `education_of_employee`: Master's
- `has_job_experience`: Y
- `requires_job_training`: N
- `no_of_employees`: 150
- `yr_of_estab`: 2005
- `region_of_employment`: Northeast
- `prevailing_wage`: 120000.00
- `unit_of_wage`: Year
- `full_time_position`: Y

**Converted Context:**
> "An applicant from Asia has an education level of Master's. The applicant has previous job experience and does not require job training. The employer was established in the year 2005 and has 150 employees. The position is located in the Northeast region, pays a prevailing wage of 120000.00 per year, and is a full-time position."

**Target Mapping:**
- `Certified` ➔ `1`
- `Denied` ➔ `0`

---

## 🚀 Model Details & Fine-Tuning

- **Selected Model**: `prajjwal1/bert-tiny` (approx. 4.4 million parameters)
  - *Why?* High training speed, low GPU/CPU footprint, and highly suitable for local verification and development on normal laptops.
- **Task**: Binary Sequence Classification.
- **Fine-Tuning Process**:
  - Tokenization using Hugging Face `AutoTokenizer` (max length = 128 tokens).
  - Fine-tuning using `AutoModelForSequenceClassification` with a sequence classification head.
  - Stratified split: **70% Training, 15% Validation, 15% Testing**.
  - Optimized with AdamW using a learning rate of `5e-5` and validation metrics tracking.
  - Prevented overfitting using `EarlyStoppingCallback` (evaluating validation F1 score).

---

## 💻 Windows Setup & Execution Guide

Follow these instructions to activate the environment and execute the scripts.

### 1. Environment Setup (Conda / Pip)
Create and activate the environment:
```powershell
conda create -n slm-project python=3.11 -y
conda activate slm-project
```
Install dependencies:
```powershell
pip install -r slm/requirements.txt
```

### 2. Run Tabular-to-Text Conversion
Run the conversion pipeline to load the raw data, format it into sentences, display samples, and save train/val/test splits to `slm/data/text_dataset/`:
```powershell
python slm/scripts/convert_tabular_to_text.py
```

### 3. Run SLM Training
Fine-tune the pretrained model. Parameters (learning rate, epochs, batch size) are fully configurable:
```powershell
python slm/scripts/train_slm.py --model_name prajjwal1/bert-tiny --epochs 3 --batch_size 16 --learning_rate 5e-5
```
The best checkpoint, tokenizer, and config will be saved to `slm/models/best_model/`.

### 4. Run SLM Evaluation
Evaluate the fine-tuned model on the unseen test dataset:
```powershell
python slm/scripts/evaluate_slm.py
```
This saves metrics to `slm/models/metrics.json` and exports a confusion matrix visualization to `slm/models/confusion_matrix.png`.

### 5. Compare Models (Traditional ML vs. SLM)
Compare metrics, inference latency, parameter counts, and disk footprints on the identical test set:
```powershell
python slm/scripts/compare.py
```
Outputs are written to `slm/models/model_comparison.json`.

### 6. Run CLI Predictions
Test interactive prediction prompting:
```powershell
python slm/scripts/predict_slm.py
```

### 7. Launch the Interactive Dashboard
Run the standalone presentation-ready web dashboard:
```powershell
python slm/scripts/app.py
```
Open [http://127.0.0.1:8081](http://127.0.0.1:8081) to view the dark-mode glassmorphism dashboard, enter candidate details, preview the converted natural language string, and execute SLM predictions!

### 8. Run Automated Unit Tests
Verify data conversion, dataset classes, and prediction validator schemas:
```powershell
pytest slm/tests/
```

---

## 📈 Limitations & Future Improvements

### Limitations:
- **Inference Latency**: Transformers have a larger computational footprint and higher latency than tree-based classifiers (Random Forest/XGBoost).
- **Interpretability**: Explaining model decisions requires techniques like attention weight visualizations, whereas tree classifiers offer direct feature importances.
- **Sequence Length**: Extremely large tabular representations could hit transformer context window limits (though not an issue for this dataset).

### Future Enhancements:
- **Larger SLMs**: Fine-tune larger models (e.g., `distilbert-base-uncased` or Llama-3-8B via LoRA) to capture more complex semantic connections.
- **RAG & Explainability**: Integrate Retrieval-Augmented Generation (RAG) to output natural-language explanations along with predictions.
- **Prompt Tuning**: Experiment with zero-shot/few-shot in-context learning using instruction-tuned models.
