# Heart Disease Prediction: Data Preprocessing, Model Training, and Evaluation

This repository contains a complete pipeline for heart disease prediction, starting from data preprocessing to model training and evaluation. The project implements various machine learning techniques, including feature engineering, normalization, standardization, feature selection, and model evaluation using multiple metrics.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Pipeline](#pipeline)
    - [Data Preprocessing](#data-preprocessing)
    - [Feature Engineering](#feature-engineering)
    - [Model Training](#model-training)
    - [Model Evaluation](#model-evaluation)
4. [Model Deployment](#model-deployment)
5. [How to Run](#how-to-run)
6. [License](#license)

---

## Project Overview

Heart disease is a critical health concern worldwide. This project aims to build a predictive model that leverages machine learning to assess the likelihood of heart disease based on patient data. The workflow includes:
- Handling missing data and duplicates.
- Visualizing data distributions.
- Performing feature selection to identify relevant predictors.
- Training models like Random Forest, SVM, and Logistic Regression.
- Evaluating performance with metrics such as confusion matrix and balanced accuracy.

---

## Requirements

To run this project, install the following Python libraries:
```bash
pip install pandas matplotlib numpy scikit-learn

---

## pipeline

Data Preprocessing
Load and explore the dataset (heart_disease_df_1.csv).
Visualize key features (e.g., cholesterol distribution).
Handle missing data using SimpleImputer.
Drop unnecessary columns (oldpeak) and duplicates.
Impute missing values in the restecg column with the mean.
Feature Engineering
Normalization:

Normalize features (e.g., age) to the range [0, 1] using MinMaxScaler.
Visualize the effect of normalization.
Standardization:

Standardize features to have a mean of 0 and a variance of 1 using StandardScaler.
Visualize the effect of standardization.
Feature Selection:

Use a RandomForestClassifier with SelectFromModel to identify the most important features.
Visualize feature importance using bar plots.
Model Training
Train multiple models, including:
Support Vector Machine (SVM):
Use a linear kernel for classification.
Logistic Regression:
Train with max_iter=1000 to ensure convergence.
Handle missing data during training using SimpleImputer.
Model Evaluation
Evaluate models using:
Balanced Accuracy: Measure how well the model handles imbalanced datasets.
Confusion Matrix: Interpret true positives, false positives, true negatives, and false negatives.
K-Fold Cross-Validation: Ensure robust evaluation by testing on multiple splits of the data.
Confusion Matrix Interpretation
Use the confusion matrix to analyze model performance across all predicted classes.
