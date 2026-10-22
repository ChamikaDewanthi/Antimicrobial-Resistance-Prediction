import pandas as pd

from load_file import load_file

# Handles duplicates by removing unncessary rows or summarizing data
def handle_duplicates(df, file_name):
    try:
        primary_keys = ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"]
        if(file_name == "demographics"):
            patient_conflicts = (
                df.groupby("anon_id")[["age", "gender"]]
                .nunique(dropna = True)
            )
            conflicting_patients = patient_conflicts[
                (patient_conflicts["age"] > 1) |
                (patient_conflicts["gender"] > 1)
            ].index
            df = df[~df["anon_id"].isin(conflicting_patients)].copy()
            df = df.drop_duplicates(subset = ["anon_id"], keep = "first").copy()
        elif(file_name == "prior_infecting_organism"):
            df["prior_infecting_organism_days_to_culutre"] = pd.to_numeric(df["prior_infecting_organism_days_to_culutre"], errors = "coerce")
            prior_organism_summary = (
                df.groupby(primary_keys)
                .agg(
                    prior_organism_count = ("prior_organism", "nunique"),
                    days_since_earliest_prior_organism = ("prior_infecting_organism_days_to_culutre", "max")
                )
                .reset_index()
            )
            df = prior_organism_summary
        elif(file_name == "labs"):
            labs_columns = ["median_wbc", "median_neutrophils", "median_lactate", "median_cr"]
            for col in labs_columns:
                df[col] = pd.to_numeric(df[col], errors = "coerce")
            labs_summary = (
                df.groupby(primary_keys)[labs_columns]
                .median()
                .reset_index()
            )
            df = labs_summary
        elif(file_name == "vitals"):
            vitals_columns = ["median_heartrate", "median_resprate", "median_temp"]
            for col in vitals_columns:
                df[col] = pd.to_numeric(df[col], errors = "coerce")
            vitals_summary = (
                df.groupby(primary_keys)[vitals_columns]
                .median()
                .reset_index()
            )
            df = vitals_summary
        elif(file_name == "antibiotic_class_exposure"):
            df["time_to_culturetime"] = pd.to_numeric(df["time_to_culturetime"], errors = "coerce")
            antibiotic_exposure_summary = (
                df.groupby(primary_keys)
                .agg(
                    antibiotic_class_count = ("antibiotic_class", "nunique"),
                    days_since_most_recent_exposure = ("time_to_culturetime", "min")
                )
                .reset_index()
            )
            df = antibiotic_exposure_summary
        else:
            df = df.drop_duplicates(subset = ["anon_id"], keep = "first")
        return df
    except Exception as e:
            print(f"Unexpected error: {e}")

# Gets other data files
def get_file(path, columns, file_name):
    df = load_file(path, columns)
    df = handle_duplicates(df, file_name)
    return df

# Removes unwanted rows (only rows with the target result "Susceptible" or "Resistant" is filtered)
def remove_unwanted_results(df):
    try:
        print("Filtering only Susceptible or Resistant results...")
        df = df[df["susceptibility"].isin(["Susceptible", "Resistant"])]
        return df
    except KeyError as e:
        print(f"Column not found: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Removes duplicate rows with exactly same or conflicting target values from main file
def remove_main_duplicates(df):
    try:
        print("Removing duplicate records from main file...")
        df = df.drop_duplicates(keep = "first")
        grain_columns = ["anon_id", "order_proc_id_coded", "organism", "antibiotic"]
        label_counts = (
            df
            .groupby(grain_columns)["susceptibility"]
            .nunique()
        )
        conflicting_keys = label_counts[label_counts > 1].index
        row_keys = pd.MultiIndex.from_frame(df[grain_columns])
        conflicting_rows = row_keys.isin(conflicting_keys)
        df = df.loc[~conflicting_rows].copy()
        return df
    except Exception as e:
        print(f"Unexpected error: {e}")

# Gets the main file
def get_main_file(path, columns):
    df_main = load_file(path, columns)
    df_main = remove_unwanted_results(df_main)
    df_main = remove_main_duplicates(df_main)
    return df_main