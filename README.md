# Machine Learning and Neural Network–Based Corrosion Analysis

📌 **Dissertation Project**

---

## 📌 Project Overview

This project investigates corrosion behavior using regression, machine learning, and neural network models applied to a synthetic corrosion dataset. The objective is to model corrosion rate as a function of environmental and operational variables and to evaluate how different modeling approaches perform in terms of accuracy, generalization, and interpretability.

The analysis progresses from a statistically interpretable regression baseline to nonlinear machine learning and neural network models, supported by exploratory data analysis, diagnostics, and scenario-based evaluation.

---

## 🎯 Objectives

- Analyze relationships between corrosion rate and key influencing variables  
  (pH, temperature, chloride concentration, flow velocity, dissolved oxygen)
- Establish a linear regression baseline for interpretability
- Compare linear, tree-based, and neural network regression models
- Evaluate model performance using standard regression metrics
- Perform scenario-based corrosion rate prediction and Remaining Useful Life (RUL) estimation
- Generate outputs suitable for engineering interpretation and decision support

---

## 🧪 Methodology

- **Exploratory Data Analysis (EDA)**  
  Distribution analysis, feature relationships, and correlation assessment

- **Feature Engineering & Preprocessing**  
  Nonlinear feature construction (e.g., pH²), scaling, and train/validation/test splitting

- **Modeling Approaches**
  - Linear Regression (baseline, interpretable)
  - XGBoost Regressor (nonlinear interactions)
  - Multi-Layer Perceptron (MLP) neural network using PyTorch

- **Model Evaluation & Diagnostics**
  - MAE, RMSE, R², and adjusted R²
  - Residual analysis and prediction vs actual plots
  - Learning-curve analysis for neural network training

- **Scenario-Based Analysis**
  - Representative operating conditions
  - Corrosion rate prediction and Remaining Useful Life (RUL) estimation

---

## 📂 Repository Structure

DATA/
└── corrosion_synthetic_timeseries.csv

NOTEBOOK/
├── corrosion_v1.ipynb        # Initial regression-based analysis
└── corrosion_v2.ipynb        # ML, neural network, and model comparison workflow

deployment/
├── app.py                   # Flask API for corrosion rate inference
├── Dockerfile               # Containerized deployment configuration
├── requirements.txt         # Deployment-specific dependencies
└── artifacts/
    ├── linear_reg_model.joblib   # Trained Linear Regression model
    └── feature_order.joblib      # Feature schema for inference consistency

REQUIREMENTS.txt             # Training & analysis dependencies
README.md                    # Project documentation


---

## 📊 Outputs and Insights

- Exploratory analysis of corrosion-driving variables  
- Comparative performance evaluation across regression and neural models  
- Generalization assessment using held-out test data  
- Scenario-based corrosion rate and RUL estimation  
- Exportable results for downstream visualization and reporting  

An interactive Tableau dashboard was created to visualize corrosion trends, model predictions, and scenario outcomes.

🔗 **Interactive Dashboard:**  
https://public.tableau.com/app/profile/arifa.farhath.mohammedmazheralikhan/viz/CorrosionModelPredictionsandRULAnalysis/ModelPrediction

---

## 🚀 Quick Start

Open the main notebook and run all cells sequentially:

NOTEBOOK/corrosion_v2.ipynb


All data loading, preprocessing, modeling, evaluation, and exports are performed within the notebook.

---

🚀 Deployment & Model Serving

This project includes a lightweight production-style deployment for the trained Linear Regression corrosion model using Flask and Docker.

The deployment layer demonstrates how trained machine learning models can be transitioned from experimental notebooks into reusable, version-controlled inference services suitable for integration into engineering pipelines.

Key Deployment Features

RESTful API built using Flask

Model and feature-order persistence using joblib

Input validation and schema enforcement

Containerized deployment using Docker for portability

Separation of training (notebook) and inference (deployment) concerns

The deployed API accepts corrosion-driving parameters and returns predicted corrosion rate values in real time.

## 🔮 Future Work

- Include additional corrosion-related variables and domain-informed features  
- Apply the modeling framework to real inspection or sensor datasets  
- Extend the analysis with uncertainty quantification for risk-aware assessment
-  Extend the deployed API with advanced models (XGBoost, neural networks)
- Integrate uncertainty estimation for risk-aware corrosion predictions
- Connect real-time sensor streams to the inference service


---

## 👤 Author

**Arifa Farhath**  
_Dissertation project on machine learning and neural network–based corrosion analysis_

