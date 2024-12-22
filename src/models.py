import numpy as np
from typing import Any, Dict, Optional, Union

class LassoRegression:
    """
    Lasso Regression model with L1 regularization implemented using batch gradient descent.

    Attributes:
        alpha (float): Regularization strength.
        learning_rate (float): Learning rate for gradient descent.
        max_iters (int): Maximum number of iterations for training.
        beta (Optional[np.ndarray]): Model coefficients.
    """
    
    def __init__(self, alpha: float = 0.1, learning_rate: float = 0.1, max_iters: int = 5000) -> None:
        """
        Initialize the LassoRegression model.

        Parameters:
            alpha (float): Regularization strength.
            learning_rate (float): Learning rate for gradient descent.
            max_iters (int): Maximum number of iterations for training.
        """
        self.alpha: float = alpha
        self.learning_rate: float = learning_rate
        self.max_iters: int = max_iters
        self.beta: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Lasso Regression model using batch gradient descent.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)
        batch_size = min(200, n_samples)
        n_batches = n_samples // batch_size if batch_size > 0 else 1
        
        for _ in range(self.max_iters):
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                y_pred = X_batch @ self.beta
                gradient = (2 / batch_size) * (X_batch.T @ (y_pred - y_batch))
                # L1 regularization gradient
                l1_grad = self.alpha * np.sign(self.beta)
                self.beta -= self.learning_rate * (gradient + l1_grad)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained Lasso Regression model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
        """
        if self.beta is None:
            raise ValueError("Model has not been trained yet. Call 'fit' first.")
        return X @ self.beta


class ElasticNetRegression:
    """
    ElasticNet Regression model with both L1 and L2 regularization implemented using batch gradient descent.

    Attributes:
        alpha (float): Overall regularization strength.
        l1_ratio (float): Ratio of L1 to L2 regularization (0 to 1).
        learning_rate (float): Learning rate for gradient descent.
        max_iters (int): Maximum number of iterations for training.
        beta (Optional[np.ndarray]): Model coefficients.
    """
    
    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5, learning_rate: float = 0.1, max_iters: int = 5000) -> None:
        """
        Initialize the ElasticNetRegression model.

        Parameters:
            alpha (float): Overall regularization strength.
            l1_ratio (float): Ratio of L1 to L2 regularization (0 to 1).
            learning_rate (float): Learning rate for gradient descent.
            max_iters (int): Maximum number of iterations for training.
        """
        self.alpha: float = alpha
        self.l1_ratio: float = l1_ratio
        self.learning_rate: float = learning_rate
        self.max_iters: int = max_iters
        self.beta: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the ElasticNet Regression model using batch gradient descent.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)
        batch_size = min(200, n_samples)
        n_batches = n_samples // batch_size if batch_size > 0 else 1
        
        for _ in range(self.max_iters):
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                y_pred = X_batch @ self.beta
                gradient = (2 / batch_size) * (X_batch.T @ (y_pred - y_batch))
                # Combined L1 and L2 regularization gradient
                l1_grad = self.alpha * self.l1_ratio * np.sign(self.beta)
                l2_grad = self.alpha * (1 - self.l1_ratio) * 2 * self.beta
                self.beta -= self.learning_rate * (gradient + l1_grad + l2_grad)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained ElasticNet Regression model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
        """
        if self.beta is None:
            raise ValueError("Model has not been trained yet. Call 'fit' first.")
        return X @ self.beta