import joblib

from prepare_dataset import prepare_dataset
from split_data import split_dataset
from train_regression_model import train_regression_model
from train_random_forest_model import train_random_forest_model
from train_catboost import train_catboost_model
from validate_model import validate_model
from display_metrics import display_metrics

def main():

    try:
        # Prepares dataset
        prepare_dataset()
        # Generate training and testing datasets
        split_dataset()        
        # Trains models using training dataset        
        train_regression_model()       
        train_random_forest_model()
        train_catboost_model()
        # Tests models using testing data
        print("Validating the regression model...")
        lr_class_metrics, lr_metrics, lr_cm, lr_cp = validate_model("results/logistic_regression_model.joblib", "regression")
        print("Validating the random forest model...")
        rf_class_metrics, rf_metrics, rf_cm, rf_cp = validate_model("results/random_forest_model.joblib", "random_forest")
        print("Validating the CatBoost model...")
        cb_class_metrics, cb_metrics, cb_cm, cb_cp = validate_model("results/catboost_model.joblib", "CatBoost")
        # Displays evaluation metrics
        display_metrics("Logistic Regression", lr_class_metrics, lr_metrics, lr_cm, lr_cp)
        display_metrics("Random forest", rf_class_metrics, rf_metrics, rf_cm, rf_cp)
        display_metrics("CatBoost", cb_class_metrics, cb_metrics, cb_cm, cb_cp)

    except Exception as e:
        print(f"Unexpected error: {e}")

main()