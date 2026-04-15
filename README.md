# ⚖️ Fair Hiring AI System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)
![Fairlearn](https://img.shields.io/badge/Fairlearn-0.10-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **AI-powered hiring system that detects and eliminates gender bias using Fairlearn — with SHAP explainability to show WHY every decision was made.**
  “An AI-based hiring system that ensures fair and unbiased candidate selection using advanced machine learning and explainable AI.This project uses AI to remove bias from hiring decisions and provide transparent, explainable results for every candidate.”
---

## 🚨 The Problem

Traditional AI hiring systems are trained on **biased historical data**, causing them to:

- ❌ Unfairly reject qualified **female candidates**
- 🔒 Make decisions with **zero transparency** (Black Box AI)
- 📉 Cause companies to **miss top talent** due to gender bias

---

## ✅ Our Solution

**Fair Hiring AI** is a 3-layer intelligent system:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 🤖 Core AI | Scikit-learn (Logistic Regression) | Predicts hiring decisions |
| ⚖️ Fairness Engine | Fairlearn (ExponentiatedGradient) | Eliminates gender bias |
| 🧠 Explainability | SHAP (LinearExplainer) | Explains WHY decisions are made |

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| ✅ Model Accuracy | 77% |
| 📉 Bias Before Fairlearn | 0.35 |
| 📈 Bias After Fairlearn | 0.08 |
| 🏆 Bias Reduction | **77%** |

---

## 🚀 Features

### 1️⃣ Bias Dashboard
- Real-time comparison of bias **before vs after** Fairlearn
- Visual bar chart showing Demographic Parity Difference
- Accuracy metrics for both biased and fair models

### 2️⃣ Candidate Evaluator
- Evaluate any candidate instantly using sliders
- See **Biased Model** vs **Fair Model** results side by side
- Confidence score for each prediction
- Proves gender bias with same candidate — different gender

### 3️⃣ SHAP Explainability
- Visual graph showing **which features** drove the decision
- Proves decision was based on **skills, NOT gender**
- Blue bars = selection boost | Red bars = rejection push

### 4️⃣ Batch CSV Upload
- Upload multiple candidates at once
- Get predictions for entire group instantly
- Download results as CSV

### 5️⃣ Male vs Female Comparison Chart
- Visual proof of gender bias in selection rates
- Shows Female vs Male selection percentage
- Powered by Matplotlib

### 6️⃣ PDF Report Download
- Professional fairness report for each candidate
- Includes prediction, confidence, and fairness status
- Ready to share with HR teams

---

## 🛠️ Tech Stack

| Component | Tool | Version |
|-----------|------|---------|
| Language | Python | 3.11 |
| UI Framework | Streamlit | 1.40 |
| AI Model | Scikit-learn | Latest |
| Fairness | Fairlearn | 0.10 |
| Explainability | SHAP | Latest |
| PDF Generation | FPDF2 | Latest |
| Data Processing | Pandas + NumPy | Latest |
| Visualization | Matplotlib | Latest |

---

## 📁 Project Structure

fair-hiring-ai/
├── app.py              → Streamlit Web App (UI)
├── model.py            → AI Model + Fairness Engine
├── utils.py            → Charts + PDF Generator
├── generate_data.py    → Synthetic Dataset Generator
└── data.csv            → Training Dataset (500 records)

---

## ▶️ How to Run

### Prerequisites
```bash
pip install streamlit pandas scikit-learn fairlearn shap matplotlib fpdf2
```

### Run the App
```bash
# Step 1: Generate dataset
python generate_data.py

# Step 2: Run the app
streamlit run app.py

# Step 3: Open browser
http://localhost:8501
```

---

## 🎬 Demo

### Bias Proof:
| Candidate | Experience | Test Score | Gender | Biased Model | Fair Model |
|-----------|-----------|------------|--------|-------------|------------|
| Candidate A | 5 years | 75/100 | Female | ❌ Rejected | ✅ Selected |
| Candidate B | 5 years | 75/100 | Male | ✅ Selected | ✅ Selected |

> **Same qualifications — Different gender — Different result = BIAS DETECTED!**

---

## 👥 How It Works — Step by Step

1. **Open Bias Dashboard** → See bias score 0.35 (high bias)
2. **Enter Female Candidate** → Biased model REJECTS ❌
3. **Enter Same Male Candidate** → Biased model SELECTS ✅
4. **Apply Fairlearn** → Fair model gives equal chance to both
5. **See SHAP Graph** → Decision based on skills, NOT gender
6. **Download PDF Report** → Professional fairness certificate

---

## 📊 Dataset

- **Total Records**: 500 candidates
- **Features**: Experience, Test Score, Interview Score, Skills Count, Gender
- **Bias Injected**: Males selected with lower scores than females
- **Purpose**: Demonstrates real-world hiring bias

---

## 🏆 Why This Project Matters

> *"AI bias in hiring is a real problem affecting millions of job seekers worldwide. Our system proves that AI can be both accurate AND fair — transparency and explainability are not optional, they are essential."*

---

## 📄 License

MIT License — Free to use and modify

---

👨‍💻 Author

✨ Yug Patel
🚀 Passionate about AI & Technolog
💡 Building smart solutions with Artificial Intelligence

📫 Connect with me:

📧 Email: yugpatel5649@gmail.com

💼 LinkedIn: https://www.linkedin.com/in/yug-patel-8bba29378/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BfhMIluTMR4Ss7UXvLtjbMQ%3D%3

🐙 GitHub: @yugpatel5649

🔮 Future Improvements

1. Better Accuracy & Model Improvement
2. Advanced Features Add Karva
3. UI/UX Improvement
4. Integration
5. Security & Privacy
6. Real-time Analytics
7. Multi-language Support
8. Scalability
