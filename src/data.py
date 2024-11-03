# src/data.py

import pandas as pd
from typing import Any


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path)
    print("✓ Dataset loaded successfully")
    return data
