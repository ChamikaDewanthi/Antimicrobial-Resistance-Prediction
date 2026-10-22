import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
#from imblearn.pipeline import Pipeline
#from imblearn.under_sampling import RandomUnderSampler
from get_directory import get_directory

def train_random_forest_model():
    try:
        print("Training random forest model...")

        # Loads training dataset
        df = pd.read_csv(get_directory("training_dataset.csv", "results"), low_memory = False)

        # Separates features (X) and target (y)
        X_train = df.drop(columns = ["susceptibility"])
        y_train = (
            df["susceptibility"]
            .str.strip()
            .map({
                "Susceptible": 1,
                "Resistant": 0
            })
        )

        # Identifies feature types
        numeric_columns = X_train.select_dtypes(include = "number").columns.tolist()
        categorical_columns = X_train.select_dtypes(include = ["string", "bool"]).columns.tolist()
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy = "median"))
        ])
        categorical_pipeline = Pipeline([(
            "imputer", SimpleImputer( strategy = "constant", fill_value = "Unknown")),
            ("encoder", OneHotEncoder(handle_unknown = "ignore"))
        ])

        # Preprocesses the features
        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ])

        # Creates random forest model pipeline
        model = Pipeline([
            ("preprocessor", preprocessor),
            #("undersampler", RandomUnderSampler(random_state = 42)),
            (
                "classifier", RandomForestClassifier(
                n_estimators = 50,
                class_weight = "balanced",
                random_state = 42,
                n_jobs = -1
            ))
        ])

        # Trains random forest model
        model.fit(X_train, y_train)
        print("\033[92mRandom forest model training is successfully completed\033[0m")

        # Saves random forest model
        print("Saving random forest model...")
        joblib.dump(model, "results/random_forest_model.joblib")
        print("\033[92mRandom forest model is successfully saved\033[0m")

    except Exception as e:
        print(f"Unexpected error: {e}")