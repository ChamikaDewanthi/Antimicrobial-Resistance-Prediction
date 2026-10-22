import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Returns file path using environment configurations
def get_directory(key, file_type):
    try:
        if file_type == "data":
            directory = os.getenv("DATA_DIR")
        elif file_type == "results":
            directory = os.getenv("RESULTS_DIR")
        else:
            raise ValueError(f"Invalid file type: {file_type}")
        if not directory:
            raise ValueError(f"{file_type.upper()}_DIR is not defined in .env")
        return PROJECT_ROOT / directory / key
    except Exception as e:
            print(f"Unexpected error: {e}")