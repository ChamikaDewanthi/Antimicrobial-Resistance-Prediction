import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
#from imblearn.pipeline import Pipeline
#from imblearn.under_sampling import RandomUnderSampler
from get_directory import get_directory

def train_regression_model():
    try:
        print("Training regression model...")
        
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

        # Identifies numeric and categorical columns
        numeric_columns = X_train.select_dtypes(include = "number").columns.tolist()
        categorical_columns = X_train.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy = "median")),
            ("scaler", StandardScaler())
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

        # Creates the complete regression model pipeline
        model = Pipeline([
            ("preprocessor", preprocessor),
            #("undersampler", RandomUnderSampler(random_state = 42)),
            (
                "classifier",
                LogisticRegression(
                    max_iter = 1000,
                    class_weight = "balanced",
                    random_state = 42
                )
            )
        ])

        # Trains regression model
        model.fit(X_train, y_train)
        print("\033[92mRegression model training is successfully completed\033[0m")

        # Saves regression model
        print("Saving regression model...")
        joblib.dump(model, "results/logistic_regression_model.joblib")
        print("\033[92mRegression model is successfully saved\033[0m")

    except Exception as e:
        print(f"Unexpected error: {e}")