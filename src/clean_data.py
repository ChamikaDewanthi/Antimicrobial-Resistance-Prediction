import pandas as pd

# Removes identifier columns
def remove_identifiers(df):
    df = df.drop(columns = ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"])
    return df

# Remove exact duplicates
def remove_duplicates(df):
    try:
        df = df.drop_duplicates(keep = "first").copy()
        return df
    except Exception as e:
        print(f"Unexpected error: {e}")

# Identifies null values in different formats and sets as null
def clean_null_values(df):
    df = df.replace(r"^\s*$", pd.NA, regex = True)
    df = df.replace(["Null", "NULL", "null", "N/A", "NA"], pd.NA)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

# Drops columns with more than 40% null values
def check_null_percentage(df):
    try:
        print("Removing columns that consists of more than 40% of null values")
        for column in df.columns:
            null_percentage = df[column].isna().mean() * 100
            if(null_percentage > 40):
                df = df.drop(columns=[column])
        return df
    except KeyError as e:
        print(f"Column not found: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Handles missing values by setting a suitable value
def fill_null_values(df):
    try:
        # Replaces categorical features with a new unknown category
        unknown_columns = ["ordering_mode", "gender", "prior_organism_count", "days_since_earliest_prior_organism", "antibiotic_class_count", "days_since_most_recent_exposure"]
        for column in df.columns:
            if(column in unknown_columns):
                df[column] = df[column].fillna("Unknown")
        # Replaces numerical features with median
        numerical_columns = ["median_wbc", "median_cr", "median_heartrate", "median_resprate", "median_temp"]
        for column in df.columns:
            if(column in numerical_columns):
                df[column] = pd.to_numeric(df[column], errors = "coerce")
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
        return df
    except KeyError as e:
        print(f"Column not found: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def clean_data(df):
    df = remove_duplicates(df)
    df = clean_null_values(df)
    df = check_null_percentage(df)
    return df