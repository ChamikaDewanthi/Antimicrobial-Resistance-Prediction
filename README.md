# Antimicrobial Resistance (AMR) Prediction Using Machine Learning

## Project Overview

This project develops a machine learning pipeline to predict antibiotic susceptibility and resistance using the Antibiotic Resistance Microbiology Dataset (ARMD).

The project combines microbiology culture records with relevant patient, clinical, and prior antimicrobial exposure information to construct a dataset for binary classification.

The main objective is to evaluate machine learning models in identifying resistant cases and compare their predictive performance.

## Dataset

**Dataset:** Antibiotic Resistance Microbiology Dataset (ARMD)

- Research paper: https://www.nature.com/articles/s41597-025-05649-7
- Dataset repository: https://datadryad.org/dataset/doi:10.5061/dryad.jq2bvq8kp

### Target Variable

The target variable is `susceptibility`.

| Value | Meaning |
|---|---|
| `1` | Susceptible |
| `0` | Resistant |

## Data Preprocessing

The preprocessing workflow includes:

1. Loading and inspecting the source datasets.
2. Checking data types, missing values, duplicate records, and key consistency.
3. Standardizing numerical and categorical values.
4. Aggregating supporting datasets to the appropriate culture-event grain before merging.
5. Merging relevant source tables and validating the resulting dataset.
6. Removing or excluding invalid and conflicting target records.
7. Handling missing numerical values using median imputation and missing categorical values using an `Unknown` category, where applicable.
8. Retaining identifiers for data integration and validation while excluding them from model predictors.
9. Splitting the dataset by patient to reduce the risk of patient-level data leakage.
10. Applying fitted preprocessing steps to the training and test datasets consistently.

## Machine Learning Models

The project compares the following supervised binary classification models.

### Logistic Regression

Logistic Regression provides a baseline model for predicting the probability of antibiotic resistance. It offers a relatively interpretable approach for examining relationships between the selected features and the target.

### Random Forest

Random Forest combines multiple decision trees to predict antibiotic susceptibility and resistance. It can capture complex, nonlinear relationships between clinical features and the target variable.

### CatBoost

CatBoost is a gradient-boosted decision tree algorithm that can handle categorical features directly when configured appropriately. It is considered a more flexible model that can capture nonlinear relationships and interactions.

## Model Evaluation

Model performance is assessed using the following metrics:

- **Macro AUROC:** Measures the model's ability to distinguish between the two outcome classes.
- **Macro AUPRC:** Summarizes precision–recall performance.
- **Balanced Accuracy:** Calculates the average recall across the two classes.
- **Precision:** Measures the proportion of predicted positive cases that are correct.
- **Recall:** Measures the proportion of actual positive cases that are correctly identified.
- **F1-Score:** Summarizes the balance between precision and recall.
- **Brier Score:** Measures the accuracy of predicted probabilities, with lower values indicating better probability predictions.
- **Confusion Matrix:** Summarizes correct and incorrect predictions for each class.
- **Calibration Plot:** Compares predicted probabilities with observed outcome frequencies.

## Project Execution

1. Clone the Git repository to your local machine.
2. Open the project directory in a code editor or IDE.
3. Install the required Python packages listed in requirements.txt.
4. Download the ARMD dataset from the repository and place the files in the configured data directory.
5. Update any dataset paths or configuration settings required by the project.
6. Execute main.py to run the configured data preprocessing, model training, and evaluation workflows.
7. Refer to the results directory for the merged and final datasets, evaluation metrics, calibration plots, and saved models.

## Single-Case Prediction

Once main.py is successfully executed and the three trained models are available in the results directory, a prediction can be made for an individual case using predict_case.py.

1. Execute predict_case.py.
2. Select one of the available trained models when prompted.
3. Enter the required values for each feature when prompted.
4. The selected trained model processes the provided inputs and generates a prediction for the case.
5. The prediction result is displayed in the console.
