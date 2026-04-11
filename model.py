# model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import warnings
warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv('data.csv')
    # ⚠️ Gender feature માં ADD કરો — bias બતાવવા
    features = ['experience_years', 'test_score', 'interview_score', 'skills_count', 'gender']
    X = df[features]
    y = df['selected']
    sensitive = df['gender']
    return X, y, sensitive, df

def train_biased_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, X_train, X_test, y_train, y_test, y_pred, acc

def train_fair_model(X, y, sensitive):
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42
    )
    # Fair model માં gender feature હટાવો
    gender_col = X_train.columns.get_loc('gender')
    X_train_fair = X_train.drop(columns=['gender'])
    X_test_fair  = X_test.drop(columns=['gender'])

    base_model = LogisticRegression(max_iter=1000)
    mitigator = ExponentiatedGradient(base_model, constraints=DemographicParity())
    mitigator.fit(X_train_fair, y_train, sensitive_features=s_train)
    y_pred = mitigator.predict(X_test_fair)
    acc = accuracy_score(y_test, y_pred)
    return mitigator, X_test_fair, y_test, s_test, y_pred, acc

def measure_bias(y_true, y_pred, sensitive):
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    return round(abs(dpd), 4)

def explain_prediction(model, X_train, single_input):
    try:
        import shap
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(single_input)
        return shap_values
    except Exception as e:
        print(f"SHAP error: {e}")
        return None
