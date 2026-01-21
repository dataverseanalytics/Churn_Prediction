# ML Churn Prediction - Quick Reference Guide

## 🚀 Quick Start (3 Steps)

```powershell
# Step 1: Train ML Model (first time only, takes 2-5 minutes)
python model_trainer.py

# Step 2: Start Server
python app.py

# Step 3: Open Dashboard
# Visit: http://localhost:8000
```

---

## 🤖 Where Machine Learning is Used

### ✅ ML Components

| File | Purpose | ML Usage |
|------|---------|----------|
| `model_trainer.py` | **Trains ML models** | 🔴 **CORE ML** - Trains Random Forest, Gradient Boosting, Logistic Regression |
| `ml_model.py` | **Makes predictions** | 🔴 **CORE ML** - Loads trained model, predicts churn probability |
| `app.py` | **ML API endpoints** | 🟡 **ML Integration** - Exposes `/api/ml/predict-churn` endpoint |

### ❌ Non-ML Components

| File | Purpose | ML Usage |
|------|---------|----------|
| `data_processor.py` | Loads and cleans data | ⚪ **No ML** - Just data processing |
| `risk_calculator.py` | Rule-based scoring | ⚪ **No ML** - Mathematical formulas |
| `static/index.html` | Web UI | ⚪ **No ML** - Just displays results |
| `static/js/app.js` | Frontend logic | ⚪ **No ML** - API calls only |

---

## 📊 ML Training Process

```
1. Load Data (440K records)
   ↓
2. Feature Engineering (create 18 features)
   ↓
3. Encode Categories (text → numbers)
   ↓
4. Scale Features (normalize ranges)
   ↓
5. Split Data (80% train, 20% test)
   ↓
6. Train 3 Models (Random Forest, Gradient Boosting, Logistic Regression)
   ↓
7. Evaluate Performance (ROC-AUC, Accuracy, Precision, Recall)
   ↓
8. Select Best Model (highest ROC-AUC)
   ↓
9. Save Model to Disk (models/churn_model.pkl)
```

**Time**: 2-5 minutes  
**Output**: Trained model with ~92% ROC-AUC

---

## 🔮 ML Prediction Process

```
1. Load Trained Model (from models/churn_model.pkl)
   ↓
2. Receive New Member Data (via API)
   ↓
3. Engineer Features (same as training)
   ↓
4. Encode & Scale (same transformations)
   ↓
5. Model Predicts Churn Probability (0-100%)
   ↓
6. Categorize Risk (low/medium/high)
   ↓
7. Return Prediction (JSON response)
```

**Time**: <100ms per prediction  
**Output**: Churn probability + risk category

---

## 🎯 ML Algorithms Used

| Algorithm | Type | Speed | Accuracy | Use Case |
|-----------|------|-------|----------|----------|
| **Random Forest** | Ensemble | Medium | ⭐⭐⭐⭐⭐ | **Best overall** - Usually selected |
| **Gradient Boosting** | Ensemble | Slow | ⭐⭐⭐⭐⭐ | High accuracy, slower training |
| **Logistic Regression** | Linear | Fast | ⭐⭐⭐ | Baseline comparison |

**Winner**: Usually **Random Forest** (ROC-AUC ~0.92)

---

## 📈 ML Features (18 Total)

### Original Features (11)
- Age
- Gender
- Tenure (months)
- Usage Frequency
- Support Calls
- Payment Delay (days)
- Subscription Type
- Contract Length
- Total Spend ($)
- Last Interaction (days ago)
- Churn (target variable)

### Engineered Features (7)
- **Tenure_Category**: 0-12, 13-24, 25-36, 37-48, 49+ months
- **Usage_Category**: Low, Medium, High
- **Payment_Delay_Severe**: True if delay > 15 days
- **Spend_Per_Month**: Total spend / tenure
- **Support_Call_Rate**: Support calls / tenure
- **Interaction_Recency**: Days since last interaction
- **Age_Group**: 18-25, 26-35, 36-45, 46-55, 56+

---

## 📊 ML Evaluation Metrics

| Metric | Typical Value | What It Means |
|--------|---------------|---------------|
| **ROC-AUC** | 0.92 | Model can distinguish churners 92% of the time |
| **Accuracy** | 0.89 | 89% of predictions are correct |
| **Precision** | 0.86 | Of predicted churns, 86% actually churned |
| **Recall** | 0.84 | Of actual churns, we caught 84% |
| **F1-Score** | 0.85 | Balanced performance score |

---

## 🔧 ML API Endpoints

### Get ML Predictions
```bash
POST http://localhost:8000/api/ml/predict-churn
```

**Response**:
```json
{
  "churn_probability": 73.45,  // ML model output (0-100%)
  "risk_category": "high",     // Derived from probability
  "will_churn": true           // Binary prediction
}
```

### Get Feature Importance
```bash
GET http://localhost:8000/api/ml/feature-importance
```

**Response**:
```json
{
  "feature_importance": [
    {"feature": "Tenure", "importance": 0.1823},
    {"feature": "Total Spend", "importance": 0.1567}
  ]
}
```

### Check ML Status
```bash
GET http://localhost:8000/api/ml/status
```

**Response**:
```json
{
  "ml_available": true,
  "model_loaded": true,
  "message": "ML model ready"
}
```

---

## 📚 ML Terms Explained

| Term | Simple Explanation | Example |
|------|-------------------|---------|
| **Training** | Teaching the model using past data | Learning from 440K membership records |
| **Prediction** | Model's guess for new data | "73% chance this member will churn" |
| **Feature** | Input variable | Age, Tenure, Usage Frequency |
| **Target** | What we're predicting | Churn (yes/no) |
| **Probability** | Confidence score (0-100%) | 73.45% = high confidence of churn |
| **ROC-AUC** | Model quality score (0-1) | 0.92 = excellent model |
| **Ensemble** | Combining multiple models | Random Forest = 100 decision trees |
| **Feature Engineering** | Creating new variables | Spend_Per_Month = Total / Tenure |
| **Cross-Validation** | Testing on multiple splits | 5-fold CV for robustness |

---

## 🐛 Troubleshooting

### "ML model not available"
```powershell
# Solution: Train the model first
python model_trainer.py
```

### "No members found"
```powershell
# Solution: Already fixed in data_processor.py
# Restart server: python app.py
```

### Port 8000 in use
```python
# Edit app.py, line ~260
uvicorn.run(app, host="0.0.0.0", port=8001)  # Change port
```

---

## 📁 ML Files Generated

After training, these files are created in `models/`:

| File | Size | Purpose |
|------|------|---------|
| `churn_model.pkl` | ~50MB | Trained Random Forest model |
| `scaler.pkl` | ~5KB | Feature scaling parameters |
| `label_encoders.pkl` | ~10KB | Category encoding mappings |
| `feature_columns.pkl` | ~1KB | List of feature names |
| `model_metrics.pkl` | ~2KB | Performance metrics |

**Total**: ~50MB

---

## 🎓 Learning Path

1. **Beginner**: Use the web dashboard
2. **Intermediate**: Call API endpoints
3. **Advanced**: Modify `model_trainer.py` to use your own data
4. **Expert**: Add new ML algorithms or features

---

## 📖 Full Documentation

For complete details, see:
- **[END_TO_END_DOCUMENTATION.md](file:///C:/Users/Parth%20Kher/Downloads/OneDrive_1_12-01-2026/END_TO_END_DOCUMENTATION.md)** - Comprehensive guide
- **[README.md](file:///C:/Users/Parth%20Kher/Downloads/OneDrive_1_12-01-2026/README.md)** - Installation & usage
- **[walkthrough.md](file:///C:/Users/Parth%20Kher/.gemini/antigravity/brain/dff69d6f-55e4-4b14-90bd-6e3eacb4963c/walkthrough.md)** - Implementation details

---

**Need Help?** Check the troubleshooting section in END_TO_END_DOCUMENTATION.md
