
# Diabetes Progression Prediction Using Machine Learning

## Project Overview

This project uses machine learning to predict a numerical diabetes progression score using clinical features from the scikit-learn diabetes dataset.

The project was developed as part of the BIOTECHTREK AI in Healthcare & Drug Discovery Bootcamp.

## Objective

The objective is to develop and evaluate regression models for predicting diabetes progression and deploy the best-performing model through a simple Streamlit application.

## Dataset

The project uses the built-in `load_diabetes()` dataset from scikit-learn.

The dataset contains:

- 442 observations
- 10 predictor variables
- 1 target variable representing a diabetes progression score

## Models Used

Two regression approaches were evaluated:

1. Linear Regression
2. Random Forest Regressor

Random Forest models with 100 and 200 trees were also compared.

## Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- R² Score

### Final Results

| Model | MAE | R² Score |
|---|---:|---:|
| Linear Regression | 42.79 | 0.45 |
| Random Forest (100 trees) | 44.05 | 0.44 |
| Random Forest (200 trees) | 44.28 | 0.44 |

Linear Regression was selected as the final model because it achieved the lowest MAE and highest R² score among the tested models.

## Application

A Streamlit application was developed to allow users to enter the model features and obtain a predicted diabetes progression score.

## Disclaimer

This application is intended for educational purposes only. It is not a medical diagnostic tool and should not be used for clinical decision-making.

## Project Files

- `diabetes_project.ipynb` - Main analysis notebook
- `app.py` - Streamlit application
- `diabetes_linear_model.joblib` - Trained Linear Regression model
- `diabetes_scaler.joblib` - Feature scaler
- `requirements.txt` - Required Python packages
