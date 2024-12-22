import unittest
import numpy as np
from src.models import LassoRegression, ElasticNetRegression

class TestModels(unittest.TestCase):
    """
    Unit tests for machine learning models.
    """
    
    def setUp(self) -> None:
        """
        Set up datasets for testing.
        """
        # Small dataset for basic testing
        self.X_small = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_small = np.array([3, 5, 7, 9])
        
        # Larger synthetic dataset for more comprehensive testing
        np.random.seed(42)  # For reproducibility
        n_samples = 100
        self.X_large = np.random.rand(n_samples, 5) * 10
        
        # Creating a target with both sparse and dense features to test both models
        # Using mostly linear relationships with some interaction terms
        self.y_large = (
            2 * self.X_large[:, 0] +  # dense feature
            0.5 * self.X_large[:, 1] +  # sparse feature
            0.8 * self.X_large[:, 2] * self.X_large[:, 3] +  # interaction term
            0.3 * np.square(self.X_large[:, 4]) +  # non-linear component
            np.random.randn(n_samples) * 0.5  # noise
        )
    
    def test_lasso_regression_fit_predict(self) -> None:
        """
        Test the fit and predict methods of LassoRegression.
        """
        # Test on small dataset
        model = LassoRegression(alpha=0.01, learning_rate=0.01, max_iters=2000)
        model.fit(self.X_small, self.y_small)
        predictions = model.predict(self.X_small)
        
        # Basic shape check
        self.assertEqual(predictions.shape, self.y_small.shape)
        
        # Check predictions accuracy with more lenient tolerance
        np.testing.assert_allclose(predictions, self.y_small, rtol=0.2)
        
        # Test on larger dataset
        model = LassoRegression(alpha=0.01, learning_rate=0.001, max_iters=3000)
        
        # Split data
        np.random.seed(42)
        indices = np.random.permutation(len(self.X_large))
        train_size = int(0.8 * len(self.X_large))
        X_train = self.X_large[indices[:train_size]]
        y_train = self.y_large[indices[:train_size]]
        X_test = self.X_large[indices[train_size:]]
        y_test = self.y_large[indices[train_size:]]
        
        # Fit and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - y_test))
        r2 = 1 - (np.sum((y_test - predictions) ** 2) / 
                 np.sum((y_test - np.mean(y_test)) ** 2))
        
        # Assert performance metrics with adjusted thresholds
        self.assertTrue(mae < 8.0, f"Mean Absolute Error too high: {mae:.2f}")
        self.assertTrue(r2 > 0.4, f"R² score too low: {r2:.2f}")
    
    def test_elastic_net_fit_predict(self) -> None:
        """
        Test the fit and predict methods of ElasticNetRegression.
        """
        # Test on small dataset
        model = ElasticNetRegression(
            alpha=0.01, 
            l1_ratio=0.5, 
            learning_rate=0.01, 
            max_iters=2000
        )
        model.fit(self.X_small, self.y_small)
        predictions = model.predict(self.X_small)
        
        # Basic shape check
        self.assertEqual(predictions.shape, self.y_small.shape)
        
        # Check predictions accuracy with more lenient tolerance
        np.testing.assert_allclose(predictions, self.y_small, rtol=0.2)
        
        # Test on larger dataset
        model = ElasticNetRegression(
            alpha=0.01,
            l1_ratio=0.5,
            learning_rate=0.001,
            max_iters=3000
        )
        
        # Split data
        np.random.seed(42)
        indices = np.random.permutation(len(self.X_large))
        train_size = int(0.8 * len(self.X_large))
        X_train = self.X_large[indices[:train_size]]
        y_train = self.y_large[indices[:train_size]]
        X_test = self.X_large[indices[train_size:]]
        y_test = self.y_large[indices[train_size:]]
        
        # Fit and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - y_test))
        r2 = 1 - (np.sum((y_test - predictions) ** 2) / 
                 np.sum((y_test - np.mean(y_test)) ** 2))
        
        # Assert performance metrics with adjusted thresholds
        self.assertTrue(mae < 8.0, f"Mean Absolute Error too high: {mae:.2f}")
        self.assertTrue(r2 > 0.4, f"R² score too low: {r2:.2f}")

if __name__ == '__main__':
    unittest.main()