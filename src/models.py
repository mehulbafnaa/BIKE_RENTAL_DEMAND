# src/models.py

import numpy as np
from typing import Any, Dict, Optional, Union


class RidgeRegression:
    """
    Ridge Regression model with L2 regularization implemented using batch gradient descent.

    Attributes:
        alpha (float): Regularization strength.
        learning_rate (float): Learning rate for gradient descent.
        max_iters (int): Maximum number of iterations for training.
        beta (Optional[np.ndarray]): Model coefficients.
    """
    
    def __init__(self, alpha: float = 0.1, learning_rate: float = 0.1, max_iters: int = 5000) -> None:
        """
        Initialize the RidgeRegression model.

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
        Train the Ridge Regression model using batch gradient descent.

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
                gradient = (2 / batch_size) * (X_batch.T @ (y_pred - y_batch)) + 2 * self.alpha * self.beta
                self.beta -= self.learning_rate * gradient
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained Ridge Regression model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
        """
        if self.beta is None:
            raise ValueError("Model has not been trained yet. Call 'fit' first.")
        return X @ self.beta


class DecisionTree:
    """
    Decision Tree Regressor implemented from scratch.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        tree (Optional[Union[Dict[str, Any], float]]): Root node of the tree.
    """
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2) -> None:
        """
        Initialize the DecisionTree model.

        Parameters:
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples required to split an internal node.
        """
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.tree: Optional[Union[Dict[str, Any], float]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Decision Tree model.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Union[Dict[str, Any], float]:
        """
        Recursively build the decision tree.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            depth (int): Current depth of the tree.

        Returns:
            Union[Dict[str, Any], float]: A dictionary representing the node or a leaf value.
        """
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return np.mean(y)

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return np.mean(y)

        feature_idx, threshold = best_split
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[tuple]:
        """
        Find the best feature and threshold to split the data.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            Optional[tuple]: Best feature index and threshold or None if no split is found.
        """
        best_score = float('inf')
        best_split = None

        for feature_idx in range(X.shape[1]):
            thresholds = np.percentile(X[:, feature_idx], [25, 50, 75])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                if np.sum(left_mask) < self.min_samples_split or np.sum(~left_mask) < self.min_samples_split:
                    continue
                
                left_mean = np.mean(y[left_mask]) if np.sum(left_mask) > 0 else 0
                right_mean = np.mean(y[~left_mask]) if np.sum(~left_mask) > 0 else 0
                score = (np.sum((y[left_mask] - left_mean) ** 2) +
                         np.sum((y[~left_mask] - right_mean) ** 2))
                
                if score < best_score:
                    best_score = score
                    best_split = (feature_idx, threshold)
        return best_split

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained Decision Tree model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
        """
        if self.tree is None:
            raise ValueError("Model has not been trained yet. Call 'fit' first.")
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x: np.ndarray, tree: Union[Dict[str, Any], float]) -> float:
        """
        Recursively traverse the tree to make a prediction for a single sample.

        Parameters:
            x (np.ndarray): Single sample feature vector.
            tree (Union[Dict[str, Any], float]): Current node of the tree.

        Returns:
            float: Predicted target value.
        """
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature_idx']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        return self._predict_single(x, tree['right'])


class GradientBoosting:
    """
    Gradient Boosting Regressor implemented from scratch.

    Attributes:
        n_estimators (int): Number of boosting stages.
        learning_rate (float): Learning rate shrinks the contribution of each tree.
        max_depth (int): Maximum depth of individual regression trees.
        trees (List[DecisionTree]): List of fitted decision trees.
        baseline (Optional[float]): Initial prediction (mean of target).
    """
    
    def __init__(self, n_estimators: int = 250, learning_rate: float = 0.1, max_depth: int = 5, min_samples_split: int = 2) -> None:
        """
        Initialize the GradientBoosting model.

        Parameters:
            n_estimators (int): Number of boosting stages.
            learning_rate (float): Learning rate shrinks the contribution of each tree.
            max_depth (int): Maximum depth of individual regression trees.
            min_samples_split (int): Minimum number of samples required to split an internal node.
        """
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.trees: list = []
        self.baseline: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Gradient Boosting model.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        self.baseline = np.mean(y)
        residuals = y - self.baseline
        
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            self.trees.append(tree)
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained Gradient Boosting model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
        """
        if self.baseline is None:
            raise ValueError("Model has not been trained yet. Call 'fit' first.")
        predictions = np.full(X.shape[0], self.baseline)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
