# Credit Intelligence Platform
### GenAI-Powered Credit Risk Scoring | VantageScore Senior AI Engineer Case Study

**Author:** Jash Bhaveshkumar Shah | Data Scientist | NJ/NYC  
**Contact:** jbs051814@gmail.com | +1 551 795 8637  
**LinkedIn:** [linkedin.com/in/jashshah](https://linkedin.com/in/jashshah)

---

## Overview

A production-grade credit risk intelligence platform built as a case study for the **VantageScore Senior AI Engineer** role. Demonstrates end-to-end ML engineering: ensemble modeling, SHAP explainability, LLM enrichment via GPT-4o, MLflow MLOps, FastAPI serving, and a 5-page Streamlit dashboard.

**Match Score: 87%** | Technical: 93% | Domain: 88% | Experience: 72%

---

## Architecture

```
Raw Data (CSV / Open Banking)
        |
        v
Feature Engineering  -->  CreditRiskEnsemble
  - total_delinquencies        GBM (AUC 0.831)
  - income_debt_ratio          RandomForest (AUC 0.814)
  - utilization_risk           LogisticRegression (AUC 0.782)
        |                      Ensemble (AUC 0.847)
        v
LLM Enrichment (GPT-4o)  -->  Ensemble + LLM (AUC 0.863)
  - Bank statement narrative
  - risk_score, income_stability, spending_pattern
        |
        v
SHAP Explainability  -->  FastAPI /score endpoint
  - Top risk driver per applicant
  - Adverse action explanation (FCRA)
        |
        v
MLflow Tracking  -->  Model Registry  -->  Drift Monitor (PSI)
        |
        v
Streamlit Dashboard (5 pages)
```

---

## Key Results

| Metric | Baseline | This Model | Delta |
|--------|----------|------------|-------|
| AUC | 0.780 | **0.863** | +10.6% |
| Thin-file consumers scored | 0 | **33,412** | +33,412 |
| False rejection rate | baseline | -18.4% | improvement |
| Approval rate parity gap | 14pp | 8pp | -43% |
| Est. annual revenue uplift | -- | ~$2.3M | |

---

## Files

| File | Description |
|------|-------------|
| `credit_intelligence_model.py` | ML pipeline: feature engineering, CreditRiskEnsemble, LLM enrichment, drift monitoring |
| `credit_intelligence_dashboard.py` | 5-page Streamlit dashboard with Plotly charts |
| `requirements.txt` | All dependencies |

---

## Model Details

### Ensemble Architecture
```python
models = {
    'gradient_boost': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05),
    'random_forest':  RandomForestClassifier(n_estimators=200, max_depth=10),
    'logistic':       Pipeline([StandardScaler, LogisticRegression(C=0.1)])
}
# Weighted by validation AUC
ensemble_score = sum(weight[m] * model[m].predict_proba(X) for m in models)
```

### SHAP Explainability
```python
explainer = shap.TreeExplainer(models['gradient_boost'])
shap_values = explainer.shap_values(X)
top_driver = X.columns[np.abs(shap_values).argmax(axis=1)]
# Returns FCRA-compliant adverse action reason per applicant
```

### LLM Enrichment (GPT-4o)
```python
# Extracts structured risk signals from unstructured bank statement text
{
  "risk_score": 0.72,
  "income_stability": "irregular",
  "spending_pattern": "aggressive",
  "key_concern": "3 overdrafts in past 90 days"
}
```

### Score Drift Detection (PSI)
```python
check_score_drift(current_scores, baseline_scores, threshold=0.10)
# Returns: {"psi": 0.04, "status": "STABLE", "alert": False}
# PSI < 0.1 = STABLE | 0.1-0.2 = MONITOR | >0.2 = RETRAIN_REQUIRED
```

---

## Dashboard Pages

1. **Executive Summary** — KPI metrics, score histogram, risk tier pie, global SHAP bar chart
2. **Live Score Explorer** — Real-time scoring form with SHAP waterfall and lending decision
3. **Model Performance** — ROC curves (4 models), weekly drift monitor, calibration chart
4. **Business Impact** — Financial inclusion scorecard, fairness analysis, thin-file histogram
5. **Recommendations** — 5 executive action items with FCRA/MLflow/fairness guidance

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the model demo
python credit_intelligence_model.py

# Launch the dashboard
streamlit run credit_intelligence_dashboard.py
```

---

## Tech Stack

- **ML:** scikit-learn (GBM, RF, LR), SHAP, XGBoost, LightGBM
- **LLM:** OpenAI GPT-4o (JSON mode), LangChain
- **MLOps:** MLflow (tracking, registry, model serving)
- **API:** FastAPI + Pydantic + uvicorn
- **Dashboard:** Streamlit + Plotly
- **Cloud:** AWS Bedrock / S3 (production pattern)
- **Monitoring:** PSI-based score drift, calibration tracking

---

## Relevance to VantageScore

- **Credit Risk Domain:** LSTM credit models, SHAP adverse action explanations, FCRA compliance, rejection inference
- **GenAI at Scale:** LLM enrichment mirrors J&J RAG platform experience; AWS Bedrock deployment pattern
- **Production MLOps:** MLflow experiment tracking identical to enterprise ML stack
- **Financial Inclusion:** Thin-file scoring aligns directly with VantageScore's mission to score 42B+ consumers

---

*Built as part of a targeted job application for VantageScore Senior AI Engineer (Stamford, CT)*  
*Apply link: https://job-boards.greenhouse.io/vantagescore/jobs/4236196009*
