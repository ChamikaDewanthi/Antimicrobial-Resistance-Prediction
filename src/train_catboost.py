import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.pipeline import Pipeline

def train_catboost_model():
    try:
        print("Training CatBoost model...")
        
        # Loads training dataset
        df = pd.read_csv("results/training_dataset.csv", low_memory = False)

        # Separates features (X) and target (y)
        X_train = df.drop(columns = ["susceptibility"]).copy()
        y_train = (
            df["susceptibility"]
            .astype("string")
            .str.strip()
            .map({
                "Susceptible": 1,
                "Resistant": 0
            })
        )

        # Identifies numeric and categorical columns
        numeric_columns = X_train.select_dtypes(include = "number").columns.tolist()
        categorical_columns = X_train.select_dtypes(include = ["object", "category", "string", "bool"]).columns.tolist()

        # Prepares categorical features for CatBoost model
        for col in categorical_columns:
            X_train[col] = (
                X_train[col]
                .astype("string")
                .fillna("Unknown")
                .astype(str)
            )

        # Trains CatBoost model
        model = Pipeline([
            #("undersampler", RandomUnderSampler(random_state = 42)),
            ("classifier", CatBoostClassifier(
                iterations = 100,
                depth = 6,
                learning_rate = 0.1,
                loss_function = "Logloss",
                eval_metric = "AUC",
                auto_class_weights = "Balanced",
                random_seed = 42,
                verbose = False,
                thread_count = 4,
                allow_writing_files = False,
                cat_features = categorical_columns
            ))
        ])
        model.fit(X_train, y_train)

        # Saves CatBoost model
        print("Saving CatBoost model...")
        joblib.dump(model, "results/catboost_model.joblib")
        print("\033[92mCatBoost model is successfully saved\033[0m")

    except Exception as e:
        print(f"Unexpected error: {e}")