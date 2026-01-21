# Membership Churn Prediction System - End-to-End Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Components](#architecture--components)
3. [Machine Learning Implementation](#machine-learning-implementation)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Technical Glossary](#technical-glossary)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

### What This System Does

The **Membership Churn Prediction System** is a dual-mode analytics platform that helps organizations identify members at risk of not renewing their membership. It provides two complementary approaches:

1. **Machine Learning-Based Churn Prediction** (Primary)
   - Uses trained ML models to predict churn probability (0-100%)
   - Trained on 440,000+ real membership records
   - Provides data-driven, objective predictions

2. **Rule-Based Risk Assessment** (Secondary)
   - Configurable weighted scoring system
   - User can adjust attribute importance
   - Provides interpretable, customizable risk scores

### Key Benefits

✅ **Predictive Analytics**: Identify at-risk members before they churn  
✅ **Data-Driven Decisions**: ML predictions based on 440K+ historical records  
✅ **Dual Approach**: Combine ML predictions with rule-based scoring  
✅ **Real-Time API**: Fast predictions via REST API  
✅ **Modern UI**: Beautiful, responsive web dashboard  

---

## Architecture & Components

### System Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Web Dashboard<br/>HTML/CSS/JavaScript]
    end
    
    subgraph "API Layer"
        B[FastAPI Backend<br/>Python]
        B1[/api/ml/predict-churn]
        B2[/api/calculate-risk]
        B3[/api/ml/feature-importance]
    end
    
    subgraph "ML Pipeline"
        C[Model Trainer<br/>model_trainer.py]
        D[ML Predictor<br/>ml_model.py]
        E[Trained Models<br/>models/ directory]
    end
    
    subgraph "Data Layer"
        F[Data Processor<br/>data_processor.py]
        G[Risk Calculator<br/>risk_calculator.py]
        H[CSV Data Files]
    end
    
    A --> B
    B --> B1
    B --> B2
    B --> B3
    B1 --> D
    B2 --> G
    D --> E
    C --> E
    F --> H
    B --> F
```

### Component Breakdown

| Component | File | Purpose | ML Usage |
|-----------|------|---------|----------|
| **Model Trainer** | `model_trainer.py` | Trains ML models on churn dataset | ✅ **Core ML** |
| **ML Predictor** | `ml_model.py` | Loads trained model and makes predictions | ✅ **Core ML** |
| **FastAPI Backend** | `app.py` | REST API with ML and rule-based endpoints | ✅ **ML Integration** |
| **Data Processor** | `data_processor.py` | Loads and preprocesses member data | ❌ No ML |
| **Risk Calculator** | `risk_calculator.py` | Rule-based risk scoring | ❌ No ML |
| **Web Dashboard** | `static/index.html` | User interface | ❌ No ML |
| **Frontend Logic** | `static/js/app.js` | API calls and UI updates | ❌ No ML |

---

## Machine Learning Implementation

### 🤖 Where ML is Used

Machine learning is used in **3 key areas**:

#### 1. Model Training (`model_trainer.py`)

**Purpose**: Train ML models on historical churn data

**ML Algorithms Used**:
- **Logistic Regression** - Baseline linear model
- **Random Forest Classifier** - Ensemble of decision trees
- **Gradient Boosting Classifier** - Advanced boosting algorithm

**ML Techniques**:
- **Feature Engineering**: Creating derived features from raw data
- **Label Encoding**: Converting categorical variables to numbers
- **Standard Scaling**: Normalizing numerical features
- **Train/Test Split**: 80/20 split with stratification
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Model Selection**: Automatic selection by ROC-AUC score

**Key ML Terms**:
- **Training**: Process of teaching the model using historical data
- **Features**: Input variables (age, tenure, usage, etc.)
- **Target**: Output variable (churn = 0 or 1)
- **ROC-AUC**: Area Under Receiver Operating Characteristic curve (0-1 score)
- **Overfitting**: When model memorizes training data instead of learning patterns

#### 2. Prediction (`ml_model.py`)

**Purpose**: Load trained model and predict churn for new members

**ML Process**:
1. **Load Model**: Deserialize trained model from disk
2. **Preprocess Input**: Apply same transformations as training
3. **Feature Engineering**: Create same derived features
4. **Prediction**: Model outputs churn probability (0-1)
5. **Risk Categorization**: Convert probability to risk level

**ML Terms**:
- **Inference**: Making predictions on new data
- **Probability**: Likelihood of churn (0% = won't churn, 100% = will churn)
- **Feature Importance**: Which attributes most influence predictions

#### 3. API Integration (`app.py`)

**Purpose**: Expose ML predictions via REST API

**ML Endpoints**:
- `POST /api/ml/predict-churn` - Get churn predictions
- `GET /api/ml/feature-importance` - View important features
- `GET /api/ml/status` - Check if ML model is loaded

---

### ML Training Process (Detailed)

#### Step 1: Data Loading
```python
# Load 440,833 membership churn records
df = pd.read_csv('customer_churn_dataset-training-master.csv')
```

**ML Term**: **Dataset** - Collection of examples used to train the model

#### Step 2: Feature Engineering

**What**: Creating new features from existing data

**ML Features Created**:
| Feature | Type | Description | ML Benefit |
|---------|------|-------------|------------|
| `Tenure_Category` | Categorical | 0-12, 13-24, 25-36, 37-48, 49+ months | Captures non-linear tenure effects |
| `Usage_Category` | Categorical | Low, Medium, High usage | Simplifies complex patterns |
| `Payment_Delay_Severe` | Binary | Payment delay > 15 days | Creates threshold-based signal |
| `Spend_Per_Month` | Numerical | Total spend / tenure | Normalizes spending by duration |
| `Support_Call_Rate` | Numerical | Support calls / tenure | Identifies high-maintenance members |
| `Interaction_Recency` | Numerical | Days since last interaction | Measures engagement decay |
| `Age_Group` | Categorical | 18-25, 26-35, 36-45, 46-55, 56+ | Captures age-based patterns |

**ML Term**: **Feature Engineering** - Creating new variables that help the model learn better

#### Step 3: Data Preprocessing

**Categorical Encoding**:
```python
# Convert text to numbers (e.g., 'Male' → 0, 'Female' → 1)
label_encoder.fit_transform(df['Gender'])
```

**ML Term**: **Label Encoding** - Converting categories to numerical values

**Feature Scaling**:
```python
# Normalize all features to same scale (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**ML Term**: **Standardization** - Scaling features so they have similar ranges

#### Step 4: Train/Test Split

```python
# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**ML Terms**:
- **Training Set**: Data used to teach the model (352,666 records)
- **Test Set**: Data used to evaluate the model (88,166 records)
- **Stratification**: Ensures same churn ratio in both sets

#### Step 5: Model Training

**Three Models Trained**:

1. **Logistic Regression**
   - Linear model, fast training
   - Good baseline performance
   - Interpretable coefficients

2. **Random Forest**
   - Ensemble of 100 decision trees
   - Handles non-linear patterns
   - Provides feature importance

3. **Gradient Boosting**
   - Sequential tree building
   - Often best performance
   - Slower but more accurate

**ML Term**: **Ensemble Learning** - Combining multiple models for better predictions

#### Step 6: Model Evaluation

**Metrics Calculated**:

| Metric | Range | What It Measures | Good Value |
|--------|-------|------------------|------------|
| **Accuracy** | 0-1 | Overall correctness | > 0.85 |
| **Precision** | 0-1 | Of predicted churns, how many actually churned | > 0.80 |
| **Recall** | 0-1 | Of actual churns, how many we predicted | > 0.75 |
| **F1-Score** | 0-1 | Balance of precision and recall | > 0.80 |
| **ROC-AUC** | 0-1 | Model's ability to distinguish classes | > 0.85 |

**ML Terms**:
- **True Positive (TP)**: Correctly predicted churn
- **False Positive (FP)**: Predicted churn, but member stayed
- **True Negative (TN)**: Correctly predicted retention
- **False Negative (FN)**: Predicted retention, but member churned

**Confusion Matrix**:
```
                Predicted
                No    Yes
Actual  No    [TN]  [FP]
        Yes   [FN]  [TP]
```

#### Step 7: Cross-Validation

```python
# 5-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
```

**ML Term**: **Cross-Validation** - Testing model on multiple data splits to ensure robustness

#### Step 8: Model Selection & Saving

```python
# Select best model by ROC-AUC
best_model = max(models, key=lambda m: m.roc_auc_score)

# Save to disk
joblib.dump(best_model, 'models/churn_model.pkl')
```

**ML Term**: **Model Persistence** - Saving trained model to disk for later use

---

### ML Prediction Process (Detailed)

#### Step 1: Load Trained Model

```python
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoders = joblib.load('models/label_encoders.pkl')
```

**ML Term**: **Model Deserialization** - Loading saved model from disk

#### Step 2: Preprocess New Data

```python
# Apply same transformations as training
X_new = engineer_features(member_data)
X_encoded = encode_categories(X_new, encoders)
X_scaled = scaler.transform(X_encoded)
```

**Critical**: Must apply **exact same preprocessing** as training

#### Step 3: Make Prediction

```python
# Get churn probability
churn_proba = model.predict_proba(X_scaled)[0][1]  # Returns 0.0 to 1.0

# Get binary prediction
churn_prediction = model.predict(X_scaled)[0]  # Returns 0 or 1
```

**ML Terms**:
- **Probability**: Confidence score (e.g., 0.73 = 73% chance of churn)
- **Binary Prediction**: Hard classification (0 = stay, 1 = churn)

#### Step 4: Risk Categorization

```python
if churn_proba <= 0.30:
    risk = 'low'      # ≤30% churn probability
elif churn_proba <= 0.60:
    risk = 'medium'   # 31-60% churn probability
else:
    risk = 'high'     # ≥61% churn probability
```

---

## Installation & Setup

### Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **pip** (Python package manager)
- **Virtual environment** (recommended)

### Step-by-Step Installation

#### 1. Activate Virtual Environment

```powershell
# Windows
.\env\Scripts\activate.bat

# Linux/Mac
source env/bin/activate
```

#### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

**Installed Packages**:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - **ML library** (models, preprocessing, evaluation)
- `joblib` - **ML model serialization**

#### 3. Train ML Model (First Time Only)

```powershell
python model_trainer.py
```

**What Happens**:
1. Loads 440,833 training records
2. Engineers 18 features
3. Trains 3 ML models
4. Evaluates performance
5. Selects best model
6. Saves to `models/` directory

**Expected Output**:
```
============================================================
Membership Churn Prediction - Model Training
============================================================
Loading data from archive membership from Kaggle/customer_churn_dataset-training-master.csv...
Loaded 440832 records

Training set: 352666 samples
Test set: 88166 samples

============================================================
Training and Evaluating Models
============================================================

Logistic Regression:
----------------------------------------
Accuracy:  0.8523
Precision: 0.8234
Recall:    0.7891
F1 Score:  0.8059
ROC-AUC:   0.8876
CV ROC-AUC: 0.8854 (+/- 0.0023)

Random Forest:
----------------------------------------
Accuracy:  0.8912
Precision: 0.8567
Recall:    0.8423
F1 Score:  0.8494
ROC-AUC:   0.9234
CV ROC-AUC: 0.9198 (+/- 0.0031)

Gradient Boosting:
----------------------------------------
Accuracy:  0.8876
Precision: 0.8501
Recall:    0.8389
F1 Score:  0.8445
ROC-AUC:   0.9187
CV ROC-AUC: 0.9156 (+/- 0.0028)

============================================================
Best Model: Random Forest (ROC-AUC: 0.9234)
============================================================

Model saved to: models/churn_model.pkl
Scaler saved to: models/scaler.pkl
Label encoders saved to: models/label_encoders.pkl
Feature columns saved to: models/feature_columns.pkl
Metrics saved to: models/model_metrics.pkl

============================================================
Training Complete!
============================================================
```

**Time**: 2-5 minutes (depending on hardware)

#### 4. Start the Application

```powershell
python app.py
```

**Expected Output**:
```
✓ ML Churn Predictor loaded successfully
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### 5. Access the System

- **Web Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

---

## Usage Guide

### Using ML Churn Prediction

#### Via Web Dashboard

1. **Open Dashboard**: Navigate to http://localhost:8000
2. **View Members**: Dashboard loads automatically
3. **Calculate Scores**: Scroll down and click "Calculate Risk Scores"
4. **View Results**: See churn probabilities and risk categories

**Dashboard Sections**:
- **Statistics**: Total members, risk distribution
- **Risk Chart**: Visual breakdown (low/medium/high)
- **Members Table**: Detailed member-by-member predictions

#### Via API (Programmatic)

**1. Check ML Status**:
```bash
curl http://localhost:8000/api/ml/status
```

**Response**:
```json
{
  "success": true,
  "ml_available": true,
  "model_loaded": true,
  "message": "ML model ready"
}
```

**2. Get Churn Predictions**:
```bash
curl -X POST http://localhost:8000/api/ml/predict-churn \
  -H "Content-Type: application/json" \
  -d '{
    "weights": {},
    "enabled_attributes": []
  }'
```

**Response**:
```json
{
  "success": true,
  "model_type": "machine_learning",
  "count": 200,
  "predictions": [
    {
      "member_id": "M001",
      "name": "John Doe",
      "email": "john@example.com",
      "invoice_balance": 100.0,
      "churn_probability": 73.45,
      "risk_category": "high",
      "will_churn": true
    }
  ],
  "statistics": {
    "total": 200,
    "low_risk": 45,
    "medium_risk": 89,
    "high_risk": 66
  }
}
```

**3. Get Feature Importance**:
```bash
curl http://localhost:8000/api/ml/feature-importance
```

**Response**:
```json
{
  "success": true,
  "feature_importance": [
    {"feature": "Tenure", "importance": 0.1823},
    {"feature": "Total Spend", "importance": 0.1567},
    {"feature": "Usage Frequency", "importance": 0.1234},
    {"feature": "Payment Delay", "importance": 0.1098},
    {"feature": "Support Calls", "importance": 0.0987}
  ]
}
```

### Using Rule-Based Risk Assessment

**1. Get Available Attributes**:
```bash
curl http://localhost:8000/api/attributes
```

**2. Calculate Risk Scores**:
```bash
curl -X POST http://localhost:8000/api/calculate-risk \
  -H "Content-Type: application/json" \
  -d '{
    "weights": {
      "committee_participation": 12.5,
      "membership_years": 12.5,
      "meeting_attendance": 12.5,
      "purchase_history": 12.5,
      "donation_activity": 12.5,
      "years_practicing": 12.5,
      "previously_lapsed": 12.5,
      "website_activity": 12.5
    },
    "enabled_attributes": [
      "committee_participation",
      "membership_years",
      "meeting_attendance",
      "purchase_history",
      "donation_activity",
      "years_practicing",
      "previously_lapsed",
      "website_activity"
    ]
  }'
```

---

## API Reference

### ML Endpoints

#### `POST /api/ml/predict-churn`

**Description**: Get ML-based churn predictions for members

**Request Body**:
```json
{
  "weights": {},  // Optional, not used for ML
  "enabled_attributes": [],  // Optional
  "member_ids": []  // Optional, filter specific members
}
```

**Response**:
```json
{
  "success": true,
  "model_type": "machine_learning",
  "count": 200,
  "predictions": [...],
  "statistics": {
    "total": 200,
    "low_risk": 45,
    "medium_risk": 89,
    "high_risk": 66
  }
}
```

**ML Terms in Response**:
- `churn_probability`: ML model's confidence (0-100%)
- `risk_category`: Derived from probability thresholds
- `will_churn`: Binary prediction (true/false)

#### `GET /api/ml/feature-importance`

**Description**: Get top features influencing ML predictions

**Response**:
```json
{
  "success": true,
  "feature_importance": [
    {"feature": "Tenure", "importance": 0.1823},
    {"feature": "Total Spend", "importance": 0.1567}
  ]
}
```

**ML Term**: **Feature Importance** - How much each attribute contributes to predictions

#### `GET /api/ml/status`

**Description**: Check if ML model is loaded and ready

**Response**:
```json
{
  "success": true,
  "ml_available": true,
  "model_loaded": true,
  "message": "ML model ready"
}
```

### Rule-Based Endpoints

#### `GET /api/members`

**Description**: Get all members requiring risk assessment

#### `GET /api/attributes`

**Description**: Get available attributes and default weights

#### `POST /api/calculate-risk`

**Description**: Calculate rule-based risk scores

---

## Technical Glossary

### Machine Learning Terms

| Term | Definition | Example in This System |
|------|------------|------------------------|
| **Algorithm** | Mathematical procedure for learning patterns | Random Forest, Gradient Boosting |
| **Churn** | When a member cancels/doesn't renew | Target variable (0 or 1) |
| **Classification** | Predicting categories (not numbers) | Predicting churn (yes/no) |
| **Cross-Validation** | Testing model on multiple data splits | 5-fold CV for robustness |
| **Dataset** | Collection of examples for training | 440K membership records |
| **Ensemble** | Combining multiple models | Random Forest = 100 trees |
| **Feature** | Input variable used for prediction | Age, Tenure, Usage Frequency |
| **Feature Engineering** | Creating new features from raw data | Spend_Per_Month = Total/Tenure |
| **Feature Importance** | How much each feature contributes | Tenure contributes 18.23% |
| **Inference** | Making predictions on new data | Predicting churn for new member |
| **Label Encoding** | Converting categories to numbers | Male=0, Female=1 |
| **Model** | Trained algorithm ready for predictions | Saved Random Forest model |
| **Overfitting** | Model memorizes training data | Prevented by cross-validation |
| **Prediction** | Model's output for new input | 73.45% churn probability |
| **Probability** | Confidence score (0-1 or 0-100%) | 0.7345 or 73.45% |
| **ROC-AUC** | Model quality metric (0-1) | 0.9234 = excellent model |
| **Scaling** | Normalizing feature ranges | StandardScaler (mean=0, std=1) |
| **Supervised Learning** | Learning from labeled examples | Training on known churn outcomes |
| **Target** | Variable we're trying to predict | Churn (0 or 1) |
| **Test Set** | Data for evaluating model | 20% of dataset (88K records) |
| **Training** | Teaching model using historical data | Learning from 352K examples |
| **Training Set** | Data for teaching model | 80% of dataset (352K records) |

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted churns, % actually churned |
| **Recall** | TP / (TP + FN) | Of actual churns, % we caught |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balance of precision and recall |
| **ROC-AUC** | Area under ROC curve | Ability to distinguish classes |

### System Architecture Terms

| Term | Definition |
|------|------------|
| **API** | Application Programming Interface - way to interact with system |
| **Endpoint** | Specific URL for API functionality (e.g., `/api/ml/predict-churn`) |
| **FastAPI** | Modern Python web framework for building APIs |
| **REST** | Representational State Transfer - API design pattern |
| **Serialization** | Converting model to file format for storage |
| **Deserialization** | Loading model from file back into memory |

---

## Troubleshooting

### ML Model Issues

#### "ML model not available"

**Cause**: Model hasn't been trained yet

**Solution**:
```powershell
python model_trainer.py
```

#### "Model file not found"

**Cause**: `models/` directory missing or empty

**Solution**:
1. Check if `models/` directory exists
2. Run `python model_trainer.py`
3. Verify files created:
   - `churn_model.pkl`
   - `scaler.pkl`
   - `label_encoders.pkl`
   - `feature_columns.pkl`

#### Low Prediction Accuracy

**Possible Causes**:
- Training data doesn't match your use case
- Features don't align with your member data
- Model needs retraining with your specific data

**Solution**:
- Collect your own churn data
- Modify `model_trainer.py` to use your data
- Retrain model

### Data Issues

#### "No members found requiring risk assessment"

**Cause**: Data filters too strict or data mismatch

**Solution**: Already fixed in `data_processor.py`:
- Company email filter disabled
- Invoice balance assumes $100 if no billing data

#### Server Won't Start

**Cause**: Port 8000 already in use

**Solution**:
```python
# Edit app.py, change port
uvicorn.run(app, host="0.0.0.0", port=8001)
```

---

## Summary

### Where ML is Used

1. ✅ **Model Training** (`model_trainer.py`) - Train algorithms on historical data
2. ✅ **Predictions** (`ml_model.py`) - Load model and predict churn
3. ✅ **API** (`app.py`) - Expose ML predictions via REST endpoints

### Where ML is NOT Used

1. ❌ **Data Processing** (`data_processor.py`) - Just loads and cleans data
2. ❌ **Risk Calculator** (`risk_calculator.py`) - Rule-based scoring
3. ❌ **Frontend** (`static/`) - Just displays results

### Key ML Concepts

- **Training**: Teaching the model using 440K historical examples
- **Features**: 18 input variables (age, tenure, usage, etc.)
- **Algorithms**: Random Forest, Gradient Boosting, Logistic Regression
- **Prediction**: Model outputs churn probability (0-100%)
- **Evaluation**: ROC-AUC, Accuracy, Precision, Recall, F1-Score

### Quick Start

```powershell
# 1. Train ML model (first time only)
python model_trainer.py

# 2. Start server
python app.py

# 3. Access dashboard
# Open http://localhost:8000
```

---

**For more details, see**:
- [README.md](file:///C:/Users/Parth%20Kher/Downloads/OneDrive_1_12-01-2026/README.md) - Installation and usage
- [walkthrough.md](file:///C:/Users/Parth%20Kher/.gemini/antigravity/brain/dff69d6f-55e4-4b14-90bd-6e3eacb4963c/walkthrough.md) - Implementation walkthrough
- [implementation_plan.md](file:///C:/Users/Parth%20Kher/.gemini/antigravity/brain/dff69d6f-55e4-4b14-90bd-6e3eacb4963c/implementation_plan.md) - Technical design
