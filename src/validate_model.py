import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

def validate_model(model_file, model_name):
    try:
        # Loads trained model and testing dataset
        model = joblib.load(model_file)
        df = pd.read_csv("results/testing_dataset.csv")

        # Separates features and actual labels
        y_test = (
            df["susceptibility"]
            .str.strip()
            .map({
                "Susceptible": 1,
                "Resistant": 0
            })
        )
        if(model_name == "CatBoost"):
            X_test = df.drop(columns=["susceptibility"]).copy()
            catboost_model = model.named_steps["classifier"]
            categorical_indices = catboost_model.get_cat_feature_indices()
            for i in categorical_indices:
                col = catboost_model.feature_names_[i]
                X_test[col] = (
                    X_test[col]
                    .fillna("Unknown")
                    .astype(str)
                )
        else:
            X_test = df.drop(columns = ["susceptibility"])

        # Generates predictions and class probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        classes = list(model.classes_)
        resistant_index = list(classes).index(0)

        # Converts actual labels into binary labels
        y_binary = np.column_stack([
            (y_test == label).astype(int)
            for label in classes
        ])

        # Calculates evaluation metrics
        print("\tCalculating evaluation metrics...")
        # Macro AUROC
        macro_auroc = roc_auc_score(y_binary, y_prob, average = "macro")
        # Macro AUPRC
        class_auprc = [
            average_precision_score(y_binary[:, i], y_prob[:, i])
            for i in range(len(classes))
        ]
        macro_auprc = np.mean(class_auprc)
        # Balanced accuracy
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        # Precision
        resistant_precision = precision_score(
            y_test, y_pred, pos_label = 0, zero_division = 0
        )
        susceptible_precision = precision_score(
            y_test, y_pred, pos_label = 1, zero_division = 0
        )
        # Recall
        resistant_recall = recall_score(
            y_test, y_pred, pos_label = 0, zero_division = 0
        )
        susceptible_recall = recall_score(
            y_test, y_pred, pos_label = 1, zero_division = 0
        )
        # F1 score
        resistant_f1 = f1_score(y_test, y_pred, pos_label = 0)
        susceptible_f1 = f1_score(y_test, y_pred, pos_label = 1)
        # Brier score
        resistant_probability = y_prob[:, resistant_index]
        resistant_y_resistant_binary = (y_test == 0).astype(int)
        brier = brier_score_loss(
            resistant_y_resistant_binary,
            resistant_probability
        )
        # Confusion matrix
        cm = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels = [0, 1]),
            index = ["Actual Resistant", "Actual Susceptible"],
            columns = ["Predicted Resistant", "Predicted Susceptible"]
        )
        cm["Total (Actual)"] = cm.sum(axis = 1)
        total_row = cm[["Predicted Resistant", "Predicted Susceptible"]].sum(axis = 0)
        total_row["Total (Actual)"] = cm["Total (Actual)"].sum()
        cm.loc["Total (Predicted)"] = total_row
        print("\t\033[92mEvaluation metrics are successfully calculated\033[0m")

        # Sets evluation metrics
        class_metrics = pd.DataFrame({
            "Metric": [
                "Precision",
                "Recall",
                "F1 score"
            ],
            "Value": [
                [resistant_precision, susceptible_precision],
                [resistant_recall, susceptible_recall],
                [resistant_f1, susceptible_f1]
            ]
        })
        class_metrics[["Resistance Class Value", "Susceptible Class Value"]] = (
            pd.DataFrame(
                class_metrics["Value"].tolist(),
                index = class_metrics.index
            )
        )
        class_metrics = class_metrics.drop(columns = ["Value"])
        metrics = pd.DataFrame({
            "Metric": [
                "Macro AUROC",
                "Macro AUPRC",
                "Balanced Accuracy",
                "Brier Score"
            ],
            "Value": [
                macro_auroc,
                macro_auprc,
                balanced_acc,
                brier
            ]
        })

        # Saves evaluation metrics
        print("\tSaving evaluation metrics...")
        with pd.ExcelWriter("results/" + model_name + "_evaluation_metrics.xlsx") as writer:
            metrics.to_excel(writer, sheet_name = "evaluation metrics", index = False)
            class_metrics.to_excel(writer, sheet_name = "class metrics", index = False)
            cm.to_excel(writer, sheet_name = "confusion metrics", index = False)
        print("\t\033[92mEvaluation metrics are successfully saved\033[0m")        

        # Reports class prevalance
        class_prevalence = (
            y_test.value_counts(normalize = True)
            .mul(100)
            .round(2)
            .rename_axis("Class")
            .reset_index(name = "Percentage")
        )
        class_prevalence["Class"] = class_prevalence["Class"].map({
            0: "Resistant",
            1: "Susceptible"
        })
        class_prevalence.to_csv("results/" + model_name + "_class_prevalence.csv", index = False)

        # Generates calibration plot
        print("\tGenerating calibration plot...")
        resistant_observed_rate, resistant_predicted_mean = calibration_curve(
            resistant_y_resistant_binary,
            resistant_probability,
            n_bins = 10,
            strategy = "quantile"
        )
        plt.figure(figsize = (7, 6))
        plt.plot(
            resistant_predicted_mean,
            resistant_observed_rate,
            marker = "o",
            label = "Logistic Regression" if model_name == "regression" else "CatBoost " if model_name == "CatBoost" else "Random Forest"
        )
        plt.plot(
            [0, 1], [0, 1],
            linestyle = "--",
            label = "Perfect calibration"
        )
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed proportion")
        plt.title("Calibration Plot - Resistant Class")
        plt.legend()
        plt.grid(True, alpha = 0.3)
        plt.tight_layout()
        # Saves calibration plot
        print("\tSaving calibration plot...")
        plt.savefig(
            "results/calibration_plot_" + model_name + ".png",
            dpi = 300,
            bbox_inches = "tight"
        )
        print("\t\033[92mCalibration plot is successfully saved\033[0m")

        # Returns evaluation metrics 
        return class_metrics, metrics, cm, class_prevalence

    except Exception as e:
        #import traceback
        #traceback.print_exc()
        print(f"Unexpected error: {e}")