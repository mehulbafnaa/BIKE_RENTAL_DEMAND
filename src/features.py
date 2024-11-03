# src/features.py

import pandas as pd
import numpy as np



def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset.

    This includes:
    - Converting date columns to datetime objects.
    - Extracting temporal features such as hour, day, month, year, and weekday.
    - Creating binary features for peak hours and weekends.
    - Mapping weather severity.
    - Creating sine and cosine transformations for cyclical features.

    Parameters:
        data (pd.DataFrame): Original dataset.

    Returns:
        pd.DataFrame: Dataset with engineered features.
    """
    data['dteday'] = pd.to_datetime(data['dteday'])
    data['hour'] = data['hr']
    data['day'] = data['dteday'].dt.day
    data['month'] = data['dteday'].dt.month
    data['year'] = data['dteday'].dt.year.map({2011: 0, 2012: 1})
    data['weekday'] = data['dteday'].dt.weekday
    data['peak_hour'] = (((data['hour'].between(7, 9)) | (data['hour'].between(16, 19)))).astype(int)
    data['is_weekend'] = (data['weekday'] >= 5).astype(int)
    data['weather_severity'] = data['weathersit'].map({1: 'Good', 2: 'Moderate', 3: 'Bad', 4: ' Very Bad'})
    
    for col, max_val in [('hour', 24), ('month', 12)]:
        data[f'{col}_sin'] = np.sin(2 * np.pi * data[col] / max_val)
        data[f'{col}_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    
    print("✓ Feature engineering completed")
    return data
