# 📉 Customer Churn Prediction & Business Dashboard

An end-to-end machine learning project that predicts customer churn 
for a telecom company and presents findings through an interactive 
business dashboard.

## 🔗 Live Demo
[Click here to view the live dashboard](https://customer-churn-prediction-8y66blgkgpzkmolqkshmwq.streamlit.app/)

## 📌 Project Overview
Customer churn is one of the most costly problems a business faces.
This project builds a machine learning pipeline that:
- Analyses churn patterns across 7,000+ customer records
- Trains and compares 3 classification models
- Deploys a live dashboard with predictions and business recommendations

## 📊 Key Findings
- Overall churn rate is ~26%
- Month-to-month customers churn at 3x the rate of annual customers
- New customers (first 12 months) are the highest-risk segment
- High monthly charges are a strong predictor of churn

## 🛠️ Tools & Technologies
| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| IDE | VS Code |

## 🚀 How to Run Locally
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Launch the dashboard
streamlit run app.py
```

## 📁 Project Structure
```
customer-churn-prediction/
├── data/                   # Dataset
├── notebook/               # EDA notebook
├── app.py                  # Streamlit dashboard
├── eda.py                  # EDA functions
├── train_model.py          # Model training script
├── model.pkl               # Saved model
├── requirements.txt
└── README.md
```

## 💼 Business Impact
This tool enables retention teams to identify at-risk customers
before they leave and take targeted action to reduce churn.
