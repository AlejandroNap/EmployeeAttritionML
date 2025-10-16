Western Governors University – Computer Science Capstone

Author: Alejandro Napoles
ID: 011244229
Project Type: Predictive Data Product (Machine Learning + Dashboard)
Dataset: IBM HR Analytics Employee Attrition Dataset

This project demonstrates a fully functional data product that predicts the likelihood of employee attrition (whether an employee is likely to leave the company).
It was developed as part of the WGU Capstone requirements for Part C.

The project consists of two main components:

1. HR_Attrition_Analysis.ipynb — Jupyter Notebook
    Handles data cleaning, feature engineering, model training, and evaluation.
    Trains a Logistic Regression model using IBM’s Employee Attrition dataset.
    Produces a reusable predictive model and scaler (.pkl files).
    Includes three visualization types (bar, heatmap, scatter).

2. app.py — Streamlit Web Application
    Loads the pretrained model created in the notebook (no retraining).
    Provides an interactive dashboard for exploring attrition risks.
    Allows users to input employee details and get a live prediction.


Setup Instructions:
1. Clone or download the project:
git clone https://github.com/AlejandroNap/EmployeeAttritionML.git 
cd EmployeeAttritionML

2. Create and activate a virtual environment:
python -m venv .venv


On Windows:
.venv\Scripts\activate

On macOS/Linux:
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the Jupyter Notebook (optional, for training)
jupyter notebook HR_Attrition_Analysis.ipynb

5. Run the Streamlit web app
streamlit run app.py


The Streamlit app will open in your default web browser at:
http://localhost:8501

Streamlit App Overview
Page 1 – Predict Single Employee
    Enter attributes such as Department, Job Role, Overtime, Job Satisfaction, Age, etc.
    The app loads the trained model and scaler to predict attrition risk.
    Displays:
        Predicted risk score (0–1)
        Predicted label (Likely to Stay / Likely to Leave)
        Suggested HR action based on risk factors.

Page 2 – Dashboard
    Interactive filtering by Department, Overtime, and risk threshold.
    Displays:
        Bar chart: Attrition distribution.
        Heatmap: Numeric correlations.
        Scatter plot: Attrition Risk vs. Job Satisfaction.
    Includes a top-20 high-risk employee table for decision support.

Machine Learning Summary:
    Algorithm: Logistic Regression
    Scaler: StandardScaler (fit on training data)
    Encoding: One-Hot Encoding with pd.get_dummies(drop_first=True)
    Target Variable: Attrition (Yes/No -> binary 1/0)
    Accuracy: ≈ 0.85 on the IBM dataset
    Saved Artifacts:
        attrition_model.pkl – Trained model
        scaler.pkl – Feature scaler
        employees_with_risk.csv – Predicted probabilities for each employee


Requirements:
    Install all dependencies via:
    pip install -r requirements.txt




Notes for Evaluators:

The model was trained in HR_Attrition_Analysis.ipynb using IBM’s HR Analytics dataset.
The Streamlit app (app.py) uses the pretrained Logistic Regression model to make real-time predictions.
All rubric components from C1–C12 are addressed.

The project demonstrates a complete machine-learning lifecycle:
data ingestion → feature engineering → model training → evaluation → deployment.


This project is for educational purposes under WGU academic use.
Dataset courtesy of IBM’s public HR Analytics sample.