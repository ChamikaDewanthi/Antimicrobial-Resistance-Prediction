import pandas as pd

# Displays class prevalence and evaluation metrics 
def display_metrics(model, class_metrics, metrics, cm, class_prevalence):
    try:
        print("===================================================================================")
        print("Class prevalence of " + model + " model:\n")
        print(class_prevalence.to_string(index = False))
        print("===================================================================================")
        print(model + " model evaluation metrics:\n")
        print(metrics.to_string(index = False, justify = "left", formatters = {
            "Metric": "{:<30}".format,
            "Value": "{:.4f}".format
        }))
        print("\n" + model + " model class metrics:\n")
        print(class_metrics.to_string(
            index = False,
            justify = "left",
            formatters = {
                "Metric": "{:<30}".format,
                "Resistance Class Value": "{:.4f}".format,
                "Susceptible Class Value": "{:.4f}".format
            }
        ))
        print("\n" + model + " model confusion matrix:\n")
        print(cm.to_string(justify = "left"))
    except Exception as e:
        print(f"Unexpected error: {e}")