import pandas as pd

# Loads files
def load_file(path, columns):
    try:
        df = pd.read_csv(path, usecols = columns, dtype = {column: "string" for column in columns}, low_memory = True)
        return df
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except KeyError as e:
        print(f"Column not found: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")