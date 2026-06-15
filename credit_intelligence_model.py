"""
Credit Intelligence Platform — Full Pipeline
VantageScore Senior AI Engineer Case Study
Author: Jash Bhaveshkumar Shah
"""

# ─────────────────────────────────────────────────────────────
# SECTION 1: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
import joblib
import json
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['RevolvingUtilizationOfUnsecuredLines'] = df[
        'RevolvingUtilizationOfUnsecuredLines'].clip(0, 1)
    df['DebtRatio'] = df['DebtRatio'].clip(0, 10)
    df['MonthlyIncome'] = df['MonthlyIncome'].clip(0, 50000)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['total_delinquencies'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    df['income_debt_ratio'] = np.where(
        df['MonthlyIncome'] > 0,
        df['DebtRatio'] * df['MonthlyIncome'], 0
    )
    df['utilization_risk'] = np.where(
        df['RevolvingUtilizationOfUnsecuredLines'] > 0.9, 3,
        np.where(df['RevolvingUtilizationOfUnsecuredLines'] > 0.7, 2,
        np.where(df['RevolvingUtilizationOfUnsecuredLines'] > 0.3, 1, 0))
    )
    return df


# ─────────────────────────────────────────────────────────────
# SECTION 2: ENSEMBLE MODEL
# ─────────────────────────────────────────────────────────────
class CreditRiskEnsemble:
    """
    Ensemble credit risk scoring model with SHAP explainability.
    """
    def __init__(self, experiment_name: str = "credit-risk-scoring"):
        self.experiment_name = experiment_name
        self.models = {}
        self.weights = {}
        self.explainer = None
        self.feature_cols = None

    def _build_models(self):
        return {
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=5, subsample=0.8, random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_leaf=20, random_state=42
            ),
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    C=0.1, class_weight='balanced',
                    max_iter=1000, random_state=42
                ))
            ])
        }

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = list(X.columns)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name="credit_risk_ensemble"):
            self.models = self._build_models()
            val_scores = {}
            for name, model in self.models.items():
                model.fit(X_train, y_train)
                val_proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, val_proba)
                val_scores[name] = auc
                mlflow.log_metric(f"{name}_auc", auc)
                print(f"  {name}: AUC={auc:.4f}")

            total = sum(val_scores.values())
            self.weights = {k: v / total for k, v in val_scores.items()}

            ensemble_proba = self._ensemble_predict(X_val)
            ensemble_auc = roc_auc_score(y_val, ensemble_proba)
            mlflow.log_metric("ensemble_auc", ensemble_auc)
            print(f"\n Ensemble AUC: {ensemble_auc:.4f}")

            self.explainer = shap.TreeExplainer(self.models['gradient_boost'])
            mlflow.sklearn.log_model(
                self.models['gradient_boost'], "credit_risk_model",
                registered_model_name="CreditRiskScorer"
            )

    def _ensemble_predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.zeros(len(X))
        for name, model in self.models.items():
            proba += self.weights[name] * model.predict_proba(X)[:, 1]
        return proba

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self._ensemble_predict(X)
        credit_score = 300 + (1 - proba) * 550

        bins = [300, 579, 669, 739, 799, 851]
        labels = ['Very Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
        risk_tier = pd.cut(credit_score, bins=bins, labels=labels)

        shap_values = self.explainer.shap_values(X)
        top_feature_idx = np.abs(shap_values).argmax(axis=1)
        top_features = [X.columns[i] for i in top_feature_idx]

        return pd.DataFrame({
            'default_probability': np.round(proba, 4),
            'credit_score': credit_score.astype(int),
            'risk_tier': risk_tier.astype(str),
            'top_risk_driver': top_features,
            'shap_impact': [round(shap_values[i, top_feature_idx[i]], 4)
                            for i in range(len(X))]
        })

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# SECTION 3: LLM ENRICHMENT
# ─────────────────────────────────────────────────────────────
def extract_llm_risk_signal(narrative: str) -> dict:
    """
    Extract structured credit risk features from unstructured narrative.
    Uses OpenAI GPT-4o with JSON mode.
    Falls back gracefully if API unavailable.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = f"""You are a credit risk analyst. Analyze this bank statement narrative
and return ONLY valid JSON with:
- risk_score: float 0.0-1.0 (1.0 = highest risk)
- income_stability: "stable" | "irregular" | "declining" | "unknown"
- spending_pattern: "conservative" | "moderate" | "aggressive" | "unknown"
- key_concern: string or null

Narrative: {narrative}"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"risk_score": 0.5, "income_stability": "unknown",
                "spending_pattern": "unknown", "key_concern": None, "error": str(e)}


# ─────────────────────────────────────────────────────────────
# SECTION 4: DRIFT MONITORING
# ─────────────────────────────────────────────────────────────
def check_score_drift(current_scores: np.ndarray,
                      baseline_scores: np.ndarray,
                      threshold: float = 0.10) -> dict:
    """
    PSI-based score distribution drift detection.
    Alerts when distribution shifts beyond threshold.
    """
    def compute_psi(expected, actual, bins=10):
        expected_hist, bin_edges = np.histogram(expected, bins=bins, range=(300, 850))
        actual_hist, _ = np.histogram(actual, bins=bin_edges)
        expected_pct = expected_hist / len(expected) + 1e-8
        actual_pct = actual_hist / len(actual) + 1e-8
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi

    psi = compute_psi(baseline_scores, current_scores)
    avg_shift = abs(np.mean(current_scores) - np.mean(baseline_scores))
    pct_shift = avg_shift / np.mean(baseline_scores)

    status = "STABLE" if psi < 0.1 else ("MONITOR" if psi < 0.2 else "RETRAIN_REQUIRED")

    return {
        "psi": round(psi, 4),
        "avg_score_shift": round(avg_shift, 2),
        "pct_shift": round(pct_shift * 100, 2),
        "status": status,
        "alert": pct_shift > threshold
    }


# ─────────────────────────────────────────────────────────────
# MAIN: DEMO RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Credit Intelligence Platform --- Demo Run ===\n")

    np.random.seed(42)
    n = 5000
    X_demo = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': np.random.beta(2, 5, n),
        'age': np.random.randint(21, 75, n),
        'NumberOfTime30-59DaysPastDueNotWorse': np.random.poisson(0.3, n),
        'DebtRatio': np.random.exponential(0.4, n).clip(0, 3),
        'MonthlyIncome': np.random.lognormal(8.5, 0.8, n).clip(1000, 50000),
        'NumberOfOpenCreditLinesAndLoans': np.random.randint(0, 20, n),
        'NumberOfTimes90DaysLate': np.random.poisson(0.1, n),
        'NumberRealEstateLoansOrLines': np.random.randint(0, 5, n),
        'NumberOfTime60-89DaysPastDueNotWorse': np.random.poisson(0.1, n),
        'NumberOfDependents': np.random.randint(0, 5, n),
    })
    X_demo = engineer_features(X_demo)
    y_demo = (
        (X_demo['RevolvingUtilizationOfUnsecuredLines'] > 0.8).astype(int) +
        (X_demo['total_delinquencies'] > 2).astype(int) +
        (X_demo['DebtRatio'] > 1.5).astype(int)
    ).clip(0, 1)

    print("Training ensemble model...")
    model = CreditRiskEnsemble()
    model.train(X_demo, y_demo)

    print("\nScoring sample applicants...")
    sample = X_demo.sample(5, random_state=1)
    results = model.score(sample)
    print(results[['credit_score', 'risk_tier', 'default_probability', 'top_risk_driver']])

    print("\nRunning drift check...")
    baseline = model.score(X_demo.sample(1000, random_state=0))['credit_score'].values
    current = model.score(X_demo.sample(1000, random_state=99))['credit_score'].values
    drift = check_score_drift(current, baseline)
    print(f"  PSI: {drift['psi']} | Status: {drift['status']}")

    model.save("models/credit_risk_model.joblib")
    print("\n Model saved. Run streamlit run dashboard/app.py to launch dashboard.")
