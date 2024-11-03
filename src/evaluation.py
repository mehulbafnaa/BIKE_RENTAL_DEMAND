# src/evaluation.py

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score
import numpy as np
from typing import Tuple, Dict, Any
from models import RidgeRegression, GradientBoosting, DecisionTree


def calculate_demand_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold_percentile: int = 75) -> Tuple[float, float]:
    """
    Calculate precision and recall based on a specified demand percentile threshold.

    Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        threshold_percentile (int): Percentile to determine the threshold for high demand.

    Returns:
        Tuple[float, float]: Precision and recall scores.
    """
    threshold = np.percentile(y_true, threshold_percentile)
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    return precision, recall


def evaluate_model(name: str, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate a machine learning model using various regression and classification metrics.

    Metrics include:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)
    - Precision and Recall at specified demand percentiles.

    Parameters:
        name (str): Name of the model.
        model (Any): Trained model with a `predict` method.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): True target values for the test set.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred)
    }
    
    for percentile in [75, 90]:
        precision, recall = calculate_demand_metrics(y_test, y_pred, percentile)
        metrics[f'P{percentile} Precision'] = precision
        metrics[f'P{percentile} Recall'] = recall
    
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    return metrics
