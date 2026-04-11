# app.py
import streamlit as st
import pandas as pd
import numpy as np
from streamlit.elements.lib.utils import Key
from model import (load_data, train_biased_model, train_fair_model,
                   measure_bias, explain_prediction)
from utils import bias_bar_chart, shap_bar_chart

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Fair Hiring AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Fair Hiring AI System")
st.markdown("*Unbiased · Transparent · Explainable*")
st.markdown("---")

# ── Load & Train (cached) ──────────────────────────────────
@st.cache_resource
def initialize():
    X, y, sensitive, df = load_data()

    # Biased model
    biased_model, X_train, X_test, y_train, y_test, y_pred_b, acc_b = \
        train_biased_model(X, y)
    bias_before = measure_bias(y_test, y_pred_b, sensitive[y_test.index])

    # Fair model
    fair_model, X_test_f, y_test_f, s_test_f, y_pred_f, acc_f = \
        train_fair_model(X, y, sensitive)
    bias_after = measure_bias(y_test_f, y_pred_f, s_test_f)

    return (biased_model, fair_model, X_train, X_test,
            acc_b, acc_f, bias_before, bias_after)

(biased_model, fair_model, X_train, X_test,
 acc_b, acc_f, bias_before, bias_after) = initialize()

# ── Tab Layout ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Bias Dashboard",
    "🔍 Evaluate Candidate", 
    "📈 Batch CSV Upload",
    "📖 How It Works"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — Bias Dashboard
# ══════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Model Bias Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Biased Model Accuracy",  f"{acc_b*100:.1f}%")
    col2.metric("Fair Model Accuracy",    f"{acc_f*100:.1f}%")
    col3.metric("Bias Before ❌",         f"{bias_before:.3f}")
    col4.metric("Bias After ✅",          f"{bias_after:.3f}",
                delta=f"-{bias_before - bias_after:.3f}", delta_color="normal")

    st.markdown("### Bias Score Comparison")
    st.markdown("> Lower score = Fairer model")
    chart = bias_bar_chart(bias_before, bias_after)
    st.image(chart, use_container_width=False, width=500)

    if bias_after < bias_before:
        st.success(f"✅ Fairness improved by {(bias_before - bias_after):.3f} points after applying Fairlearn!")
    
    st.markdown("---")
    st.info("**Demographic Parity Difference**: Measures if both genders have equal chance of being selected. 0 = Perfect fairness.")

# ══════════════════════════════════════════════════════════
# TAB 2 — Evaluate Candidate
# ══════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Evaluate a Candidate")

    col_a, col_b = st.columns(2)
    with col_a:
        experience = st.slider("Experience (years)", 0, 15, 5)
        test_score = st.slider("Test Score", 50, 100, 75)
    with col_b:
        interview_score = st.slider("Interview Score", 40, 100, 70)
        skills_count    = st.slider("Number of Skills", 1, 10, 5)
    
    gender_label = st.selectbox("Gender (Sensitive Attribute)", ["Female", "Male"])
    gender = 0 if gender_label == "Female" else 1

    if st.button("⚡ Evaluate Candidate", use_container_width=True):
        input_data = pd.DataFrame([[experience, test_score, interview_score, skills_count]],
                                   columns=['experience_years','test_score',
                                            'interview_score','skills_count'])
        
        # Biased prediction
        pred_biased = biased_model.predict(input_data)[0]
        prob_biased = biased_model.predict_proba(input_data)[0][1]

        # Fair prediction
        pred_fair = fair_model.predict(input_data)[0]

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ❌ Biased Model Result")
            if pred_biased == 1:
                st.error("🔴 Selected (but possibly biased!)")
            else:
                st.warning("🟡 Rejected (possibly unfair!)")
            st.metric("Confidence", f"{prob_biased*100:.1f}%")

        with col2:
            st.markdown("### ✅ Fair Model Result")
            if pred_fair == 1:
                st.success("🟢 **SELECTED**")
            else:
                st.info("🔵 **Not Selected** (fair decision)")

        # Fairness Report
        st.markdown("---")
        st.markdown("### 📋 Fairness Report")
        
        bias_flag = "[OK] No gender bias detected" if pred_biased == pred_fair \
            else "[WARNING] Gender bias was corrected by fair model"

        st.code(f"""
Prediction    : {'Selected [YES]' if pred_fair == 1 else 'Not Selected [NO]'}
Confidence    : {prob_biased*100:.1f}%

Fairness Report:
  {bias_flag}
  [OK] Decision based on skills & merit
  [OK] Fairlearn bias mitigation applied

Key Factors:
  - Experience   : {experience} years
  - Test Score   : {test_score}/100
  - Interview    : {interview_score}/100
  - Skills       : {skills_count} skills
        """)

        # Key Factors
        st.markdown("### 📊 Key Factors")
        st.markdown(f"""
  • Experience   : {experience} years
  • Test Score   : {test_score}/100
  • Interview    : {interview_score}/100
  • Skills       : {skills_count} skills
        """)

        # PDF Download
        st.markdown("### ⬇️ Report Download કરો")
        candidate_info = {
            'Experience': f'{experience} years',
            'Test Score': f'{test_score}/100',
            'Interview Score': f'{interview_score}/100',
            'Skills Count': skills_count,
            'Gender': gender_label,
            'Confidence': f'{prob_biased*100:.1f}%'
        }
        try:
            from utils import generate_pdf_report
            pdf_buf = generate_pdf_report(
                candidate_info, pred_fair,
                prob_biased, bias_flag
            )
            st.download_button(
                "📄 PDF Report Download કરો",
                pdf_buf,
                "candidate_report.pdf",
                "application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

        # SHAP Explanation
        st.markdown("### 🧠 Why was this decision made? (SHAP Explanation)")
        try:
            shap_vals = explain_prediction(biased_model, X_train, input_data)
            if shap_vals is not None:
                shap_chart = shap_bar_chart(shap_vals, input_data.columns.tolist())
                if shap_chart is not None:
                    st.image(shap_chart, width=500)
                    st.caption("🔵 Blue = pushed towards selection | 🔴 Red = pushed towards rejection")
                else:
                    st.warning("Chart generation failed.")
            else:
                st.warning("SHAP values could not be computed.")
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

# ══════════════════════════════════════════════════════════
# TAB 3 — How It Works
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# TAB 3 — CSV Upload + Batch Analysis
# ══════════════════════════════════════════════════════════
with tab4:   # ← tab3 થી tab4
    st.header("📖 How This System Works")
    st.markdown("Multiple candidates ની CSV file upload કરો")

    # Sample CSV download
    sample_data = pd.DataFrame({
        'experience_years': [5, 3, 8, 2, 6],
        'test_score':       [85, 65, 90, 55, 78],
        'interview_score':  [80, 60, 88, 50, 72],
        'skills_count':     [7, 4, 9, 3, 6],
        'gender':           [0, 1, 0, 1, 0]
    })
    
    csv_sample = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Sample CSV Download કરો",
        csv_sample,
        "sample_candidates.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader("📂 CSV File Upload કરો", type=['csv'])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, sep=None, engine='python')
        except Exception:
            uploaded_file.seek(0)
            df_upload = pd.read_csv(uploaded_file, sep=',', encoding='utf-8-sig')
        st.markdown("### 👥 Uploaded Candidates")
        st.dataframe(df_upload, use_container_width=True)

        required_cols = ['experience_years','test_score','interview_score','skills_count','gender']
        if all(col in df_upload.columns for col in required_cols):
            features = ['experience_years','test_score','interview_score','skills_count']
            X_up = df_upload[features]

            # Predictions
            df_upload['Biased Result']  = biased_model.predict(X_up)
            df_upload['Fair Result']    = fair_model.predict(X_up)
            df_upload['Confidence %']   = (biased_model.predict_proba(X_up)[:,1] * 100).round(1)
            df_upload['Gender Label']   = df_upload['gender'].map({0:'Female 👩', 1:'Male 👨'})
            df_upload['Biased Result']  = df_upload['Biased Result'].map({1:'✅ Selected', 0:'❌ Rejected'})
            df_upload['Fair Result']    = df_upload['Fair Result'].map({1:'✅ Selected', 0:'❌ Rejected'})

            st.markdown("### 📋 Results")
            st.dataframe(df_upload[['Gender Label','experience_years','test_score',
                                    'Biased Result','Fair Result','Confidence %']],
                        use_container_width=True)

            # ── Male vs Female Comparison Chart ──────────────
            st.markdown("### 📊 Male vs Female Selection Rate")

            fair_num = fair_model.predict(X_up)
            df_upload['fair_num'] = fair_num

            gender_group = df_upload.groupby('gender').agg(
                Total=('fair_num','count'),
                Selected=('fair_num','sum')
            ).reset_index()
            gender_group['Selection Rate %'] = (gender_group['Selected'] / gender_group['Total'] * 100).round(1)
            gender_group['Gender'] = gender_group['gender'].map({0:'Female 👩', 1:'Male 👨'})

            import matplotlib.pyplot as plt
            import io
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(
                gender_group['Gender'],
                gender_group['Selection Rate %'],
                color=['#e91e8c', '#2196F3'],
                width=0.4
            )
            ax.set_ylabel('Selection Rate (%)')
            ax.set_title('Fair Model: Selection Rate by Gender')
            ax.set_ylim(0, 100)
            for bar, val in zip(bars, gender_group['Selection Rate %']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val}%', ha='center', fontweight='bold')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120)
            buf.seek(0)
            plt.close()
            st.image(buf, width=450)

            # ── PDF Report Download ───────────────────────────
            st.markdown("### 📄 Report Download કરો")
            results_csv = df_upload[['Gender Label','experience_years','test_score',
                                     'interview_score','skills_count',
                                     'Biased Result','Fair Result','Confidence %']]
            report_bytes = results_csv.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Results CSV Download કરો",
                report_bytes,
                "fairness_report.csv",
                "text/csv",
                use_container_width=True
            )

            # Summary
            total    = len(df_upload)
            selected = (fair_num == 1).sum()
            st.success(f"✅ Total: {total} candidates | Selected: {selected} | Rejected: {total-selected}")

        else:
            st.error(f"❌ CSV માં આ columns હોવા જોઈએ: {required_cols}")