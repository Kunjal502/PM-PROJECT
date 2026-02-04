# 🌍 PM2.5 Air Quality Prediction & Analytics Platform

An end-to-end machine learning system for predicting PM2.5 air pollution levels across India using ensemble models and explainable AI.

---

## 🎯 Objective

To build an accurate, scalable, and interpretable machine learning system for predicting PM2.5 levels across India, enabling data-driven environmental insights.

---

## 🚀 Key Features

- 🤖 Ensemble ML models: Random Forest, XGBoost, LightGBM, GAM  
- 🧹 Comprehensive data preprocessing pipeline  
- 🧪 Advanced feature engineering  
- 🔄 Log-transformed target modeling for stability  
- 🧠 Explainable AI using SHAP  
- ⚡ GPU-accelerated training (LightGBM)  
- 📈 Model comparison & evaluation  

---

## 📋 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- LightGBM, XGBoost  
- TensorFlow  
- SHAP  
- Matplotlib, Plotly  

---

## 🧹 Data Preprocessing

- Handling missing values  
- Outlier treatment  
- Date-time feature extraction  
- Normalization & scaling  
- Log transformation of PM2.5 target variable  
- Train-validation split  

---

## 🧪 Feature Engineering

- Temporal features (month, season, trends)  
- Statistical rolling features  
- Pollution interaction features  
- Regional encoding  
- Weather-influenced variables  

---

## 🤖 Modeling Approach

| Model | Purpose |
|------|--------|
| Random Forest | Baseline ensemble |
| XGBoost | High performance boosting |
| LightGBM (GPU) | Fast large-scale learning |
| GAM | Final smoothing & interpretability |

Final predictions generated through ensemble blending.

---

## 🧠 Explainability

- SHAP summary plots  
- Feature importance ranking  
- Model behavior visualization  

---

## 📁 Project Structure

PM-PROJECT/
│
├── data/
│ └── finaldataset.parquet
│
├── notebooks/
│ └── FINAL-PM2.5-model.ipynb
│
├── preprocessing/
│ └── data_cleaning.py
│ └── feature_engineering.py
│
├── models/
│ └── rf_model.pkl
│ └── xgb_model.pkl
│ └── lgbm_model.pkl
│ └── gam_model.pkl
│
├── explainability/
│ └── shap_analysis.py
│ └── plots/
│
├── requirements.txt
└── README.md


---

## 📊 Results

- Improved prediction stability using log transformation  
- LightGBM achieved highest accuracy  
- Ensemble model reduced overall error  
- SHAP identified key pollution and temporal drivers  

---

## 📄 License

MIT License  