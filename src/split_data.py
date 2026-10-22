import pandas as pd

from sklearn.model_selection import train_test_split
from clean_data import fill_null_values
from get_directory import get_directory

def split_dataset():
    try:
        # Loads the merged file
        print("Loading processed dataset...")
        df = pd.read_csv(get_directory("final_dateset.csv", "results"), low_memory = False)

        # Splits the dataset into training and testing datasets
        print("Splitting into training and testing datasets...")
        training_dataset, testing_dataset = train_test_split(
            df,
            test_size = 0.20,
            stratify = df["susceptibility"],
            random_state = 42
        )

        # Handles missing values
        print("Handling missing values of training dataset...")
        training_dataset = fill_null_values(training_dataset)

        # Saves training and testing datasets in results directory
        print("Saving training and testing datasets...")
        training_dataset.to_csv("results/training_dataset.csv", index = False)
        testing_dataset.to_csv("results/testing_dataset.csv", index = False)
        print("\033[92mTraining and testing datasets are successfully saved\033[0m")
        
    except Exception as e:
        print(f"Unexpected error: {e}")