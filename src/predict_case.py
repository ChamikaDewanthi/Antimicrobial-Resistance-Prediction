import pandas as pd
import numpy as np
import joblib

from get_directory import get_directory

def predict_case(model_file):
    try:
        # Loads trained model
        model = joblib.load(model_file)

        # Identifies the model structure
        if hasattr(model, "named_steps"):
            steps = model.named_steps
        else:
            steps = {}
        classifier = steps.get("classifier", model)

        # Identifies numerical and categorical features
        numeric_columns = []
        categorical_columns = []
        if "preprocessor" in steps:
            preprocessor = steps["preprocessor"]
            feature_names = list(model.feature_names_in_)
            for name, transformer, columns in preprocessor.transformers_:
                if name == "numeric":
                    numeric_columns = list(columns)
                elif name == "categorical":
                    categorical_columns = list(columns)
        else:
            feature_names = list(classifier.feature_names_)
            categorical_indices = classifier.get_cat_feature_indices()
            categorical_columns = [
                feature_names[i]
                for i in categorical_indices
            ]
            numeric_columns = [
                column
                for column in feature_names
                if column not in categorical_columns
            ]

        print("Enter the values of features for the case")
        print("Leave the field blank if the value is unknown\n")

        # Collects input values
        input_values = {}
        for column in feature_names:
            while True:
                value = input(f"{column}: ").strip()
                if column in numeric_columns:
                    if value == "":
                        input_values[column] = np.nan
                        break
                    try:
                        input_values[column] = float(value)
                        break
                    except ValueError:
                        print("Please enter a valid number or leave blank")
                elif column in categorical_columns:
                    input_values[column] = (value if value else "Unknown")
                    break
                else:
                    input_values[column] = value
                    break

        # Creates a DataFrame with same feature columns as training
        input_df = pd.DataFrame([input_values], columns = feature_names)

        # Predicts susceptibility
        prediction = int(model.predict(input_df)[0])
        label = {
            1: "Susceptible",
            0: "Resistant"
        }[prediction]
        probabilities = model.predict_proba(input_df)[0]

        # Displays output
        print("\n=== Prediction Result ===")
        print(f"Predicted susceptibility: {label} ({prediction})")
        print("\nEstimated class probabilities:")
        for label, probability in zip(model.classes_, probabilities):
            name = "Susceptible" if label == 1 else "Resistant"
            print(f"{name} ({label}): {probability:.2%}")

    except Exception as e:
            print(f"Unexpected error: {e}")

def main():
    try:
        # Displays models menu
        print("\n=== Antibiotic Resistance Prediction ===\n")
        print("Models:")
        print("1. Logistic Regression Model")
        print("2. Random Forest Model")
        print("3. CatBoost Model\n")
        while True:
            # Gets user input to select a model
            value = input(f"Enter model number: ").strip()
            try:
                modelNum = int(value)                
                if modelNum == 1:
                    #predict_case("results/logistic_regression_model.joblib")
                    predict_case(get_directory("logistic_regression_model.joblib", "results"))
                    break
                elif modelNum == 2:
                    #predict_case("results/random_forest_model.joblib")
                    predict_case(get_directory("random_forest_model.joblib", "results"))
                    break
                elif modelNum == 3:
                    #predict_case("results/catboost_model.joblib")
                    predict_case(get_directory("catboost_model.joblib", "results"))
                    break
                else:
                    print("Please enter a valid number")
            except ValueError:
                print("Please enter a valid number")
    except Exception as e:
                print(f"Unexpected error: {e}")

main()