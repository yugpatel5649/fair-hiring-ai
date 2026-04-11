# model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ──────────────────────────────────────────────
def load_data():
    df = pd.read_csv('data.csv')
    features = ['experience_years', 'test_score', 'interview_score', 'skills_count']
    X = df[features]
    y = df['selected']
    sensitive = df['gender']
    return X, y, sensitive, df

# ── Train Biased Model (without fairness) ──────────────────
def train_biased_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, X_train, X_test, y_train, y_test, y_pred, acc

# ── Train Fair Model (with Fairlearn) ─────────────────────
def train_fair_model(X, y, sensitive):
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42
    )
    base_model = LogisticRegression(max_iter=1000)
    mitigator = ExponentiatedGradient(base_model, constraints=DemographicParity())
    mitigator.fit(X_train, y_train, sensitive_features=s_train)
    y_pred = mitigator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return mitigator, X_test, y_test, s_test, y_pred, acc

# ── Measure Bias ───────────────────────────────────────────
def measure_bias(y_true, y_pred, sensitive):
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    return round(abs(dpd), 4)

# ── SHAP Explanation ───────────────────────────────────────
def explain_prediction(model, X_train, single_input):
    try:
        import shap
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(single_input)
        # shap_values 2D array હોય તો 1D બનાવો
        if hasattr(shap_values, 'tolist'):
            if len(shap_values.shape) == 2:
                return shap_values  # already correct
        return shap_values
    except Exception as e:
        print(f"SHAP error: {e}")
        return None