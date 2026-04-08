# Employee Attrition Prediction using Logistic Regression

This project demonstrates a complete machine learning pipeline to predict employee attrition using Logistic Regression. It includes data preprocessing, feature engineering, model training, evaluation, and model deployment preparation.

---

## Project Overview

The goal of this project is to analyze employee data and predict whether an employee is likely to leave the company (attrition). This is a binary classification problem solved using Logistic Regression.

---

## Features

* Data cleaning and preprocessing
* Handling missing values
* Feature transformation and encoding
* Logistic Regression model training
* Model evaluation using accuracy and classification report
* Model saving using joblib
* Basic Streamlit integration (for deployment readiness)

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Seaborn
* Matplotlib
* Joblib
* Streamlit

---

## Dataset

**File:** Employee Attrition Data (1).csv

### Key Features:

* Age
* Department
* Job Satisfaction
* OverTime
* Attrition (Target Variable)

---

## Data Preprocessing

The following preprocessing steps were performed:

* Removed missing values
* Converted categorical values (Yes/No, Y/N) into numeric (1/0)
* Cleaned numeric columns (Age, JobSatisfaction)
* One-hot encoding applied to categorical feature: Department

---

## Model Training

* Algorithm: Logistic Regression
* Train-test split: 80% training, 20% testing
* Pipeline used for preprocessing + model

---

## Model Evaluation

The model was evaluated using:

* Accuracy Score
* Classification Report (Precision, Recall, F1-score)

---

## Model Saving

The trained model is saved as:

logistic_regression_model.pkl

Load it using:

import joblib
model = joblib.load('logistic_regression_model.pkl')

---

## Project Structure

├── logisticRegression.ipynb
├── Employee Attrition Data (1).csv
├── logistic_regression_model.pkl
├── README.md

---

## How to Run

1. Clone the repository:
   git clone https://github.com/Anaamalikhan/Logistic_Regression.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the notebook:
   jupyter notebook

4. (Optional) Run Streamlit app:
   streamlit run app.py

---

## Future Improvements

* Add more feature engineering
* Hyperparameter tuning
* Use advanced models (Random Forest, XGBoost)
* Build a complete Streamlit UI
* Deploy on cloud

---

## Author

## Anaam Khan
