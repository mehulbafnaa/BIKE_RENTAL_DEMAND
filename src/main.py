# src/main.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm.auto import tqdm
from typing import Tuple, Dict

from data import load_data
from features import feature_engineering
from models import RidgeRegression, GradientBoosting
from evaluation import evaluate_model


warnings.filterwarnings('ignore')


def prepare_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix X and target vector y for model training.

    This involves:
    - Selecting numerical and categorical features.
    - Encoding categorical variables using one-hot encoding.
    - Scaling numerical features.
    - Handling missing values by imputing with mean.

    Parameters:
        data (pd.DataFrame): Dataset with engineered features.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X and target vector y.
    """
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    categorical_features = ['season', 'year', 'holiday', 'workingday', 'weather_severity', 'peak_hour', 'is_weekend']
    
    X = pd.concat([
        pd.get_dummies(data[categorical_features], drop_first=True),
        pd.DataFrame(StandardScaler().fit_transform(data[numerical_features]), columns=numerical_features)
    ], axis=1)
    
    y = data['cnt'].values
    X = X.fillna(X.mean()).values.astype(np.float64)
    y = y.astype(np.float64)
    return X, y


def main() -> None:
    """
    Main function to execute the Bike Rental Demand Prediction pipeline.

    Steps:
    1. Load data.
    2. Perform feature engineering.
    3. Prepare data for modeling.
    4. Split data into training and testing sets.
    5. Train and evaluate Ridge Regression and Gradient Boosting models.
    6. Identify and display the best performing model based on RMSE.
    """
    print("\nInitializing Bike Rental Demand Prediction...")
    print("-" * 50)
    
    data = load_data('data/UCI_bike_sharing.csv')
    data = feature_engineering(data)
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining and Evaluating Models...")
    models: Dict[str] = {
        'Ridge Regression': RidgeRegression(alpha=0.1, learning_rate=0.1),
        'Gradient Boosting': GradientBoosting(n_estimators=100, learning_rate=0.1)
    }
    
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        results[name] = evaluate_model(name, model, X_test, y_test)
    
    best_model = min(results.items(), key=lambda x: x[1]['RMSE'])
    print("\nBest Model:", best_model[0])
    print(f"RMSE: {best_model[1]['RMSE']:.4f}")
    print(f"P75 Precision: {best_model[1]['P75 Precision']:.4f}")
    print(f"P75 Recall: {best_model[1]['P75 Recall']:.4f}")


if __name__ == "__main__":
    main()
