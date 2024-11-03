import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

def load_data(file_path):
    data = pd.read_csv(file_path)
    print("✓ Dataset loaded successfully")
    return data

def feature_engineering(data):
    data['dteday'] = pd.to_datetime(data['dteday'])
    data['hour'] = data['hr']
    data['day'] = data['dteday'].dt.day
    data['month'] = data['dteday'].dt.month
    data['year'] = data['dteday'].dt.year.map({2011: 0, 2012: 1})
    data['weekday'] = data['dteday'].dt.weekday
    data['peak_hour'] = (((data['hour'].between(7, 9)) | (data['hour'].between(16, 19)))).astype(int)
    data['is_weekend'] = (data['weekday'] >= 5).astype(int)
    data['weather_severity'] = data['weathersit'].map({1: 'Good', 2: 'Moderate', 3: 'Bad', 4: 'Bad'})
    
    for col, max_val in [('hour', 24), ('month', 12)]:
        data[f'{col}_sin'] = np.sin(2 * np.pi * data[col] / max_val)
        data[f'{col}_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    
    print("✓ Feature engineering completed")
    return data

def prepare_data(data):
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

class RidgeRegression:
    def __init__(self, alpha=0.1, learning_rate=0.1, max_iters=5000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.beta = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)
        batch_size = min(200, n_samples)
        n_batches = n_samples // batch_size
        
        for _ in range(self.max_iters):
            indices = np.random.permutation(n_samples)
            X, y = X[indices], y[indices]
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                y_pred = X_batch @ self.beta
                gradient = (2/batch_size) * (X_batch.T @ (y_pred - y_batch)) + 2 * self.alpha * self.beta
                self.beta -= self.learning_rate * gradient
    
    def predict(self, X):
        return X @ self.beta

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return np.mean(y)

        feature_idx, threshold = best_split
        left_mask = X[:, feature_idx] <= threshold
        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        }

    def _find_best_split(self, X, y):
        best_score = float('inf')
        best_split = None

        for feature_idx in range(X.shape[1]):
            thresholds = np.percentile(X[:, feature_idx], [25, 50, 75])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                if np.sum(left_mask) < 5 or np.sum(~left_mask) < 5:
                    continue
                
                score = (np.sum((y[left_mask] - np.mean(y[left_mask]))**2) +
                        np.sum((y[~left_mask] - np.mean(y[~left_mask]))**2))
                
                if score < best_score:
                    best_score = score
                    best_split = (feature_idx, threshold)
        return best_split

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature_idx']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        return self._predict_single(x, tree['right'])

class GradientBoosting:
    def __init__(self, n_estimators=250, learning_rate=0.1, max_depth=5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.baseline = None

    def fit(self, X, y):
        self.baseline = np.mean(y)
        residuals = y - self.baseline
        
        for _ in tqdm(range(self.n_estimators), desc="Training GBM", leave=False):
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions

    def predict(self, X):
        predictions = np.full(X.shape[0], self.baseline)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

def calculate_demand_metrics(y_true, y_pred, threshold_percentile=75):
    threshold = np.percentile(y_true, threshold_percentile)
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    return precision_score(y_true_binary, y_pred_binary), recall_score(y_true_binary, y_pred_binary)

def evaluate_model(name, model, X_test, y_test):
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

def main():
    print("\nInitializing Bike Rental Demand Prediction...")
    print("-" * 50)
    
    data = load_data('UCI_bike_sharing.csv')
    data = feature_engineering(data)
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining and Evaluating Models...")
    models = {
        'Ridge Regression': RidgeRegression(alpha=0.1, learning_rate=0.1),
        'Gradient Boosting': GradientBoosting(n_estimators=100, learning_rate=0.1)
    }
    
    results = {}
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