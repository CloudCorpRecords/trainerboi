"""
train_tabular.py - High-speed GPU-accelerated tabular model training with XGBoost.
Uses the NVIDIA GPU when one exists, otherwise the CPU.
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
import device_utils
import trainer_utils
from gpu_guard import ThermalGuard


class TabularThermalCallback(xgb.callback.TrainingCallback):
    """Thermostat every boosting round: XGBoost on GPU heats the same way torch does."""
    def __init__(self, guard=None):
        self.guard = guard if guard is not None else ThermalGuard()

    def after_iteration(self, model, epoch, evals_log):
        self.guard.maybe_cool()
        return False

def generate_sample_dataset(n_samples=100000, n_features=25, is_classification=True):
    print(f"--> Generating synthetic dataset ({n_samples:,} rows x {n_features} features)...")
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    feature_names = [f"feature_{i+1:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    if is_classification:
        # Binary classification target with non-linear interaction
        logits = (X[:, 0] * 2.0 + X[:, 1] ** 2 - X[:, 2] * 1.5 + np.random.randn(n_samples) * 0.5)
        df["target"] = (logits > 1.0).astype(int)
    else:
        # Regression target
        df["target"] = (X[:, 0] * 3.5 + X[:, 1] ** 2 - X[:, 2] * 2.0 + np.random.randn(n_samples) * 0.2)
    
    return df

def train_tabular_model(csv_path=None, target_col="target", task_type="classification", n_estimators=200, max_depth=6, learning_rate=0.1):
    trainer_utils.clear_gpu_memory()
    print("=" * 60)
    print("GPU-ACCELERATED TABULAR MODEL TRAINING (XGBoost)")
    print("=" * 60)
    
    if csv_path and os.path.exists(csv_path):
        print(f"--> Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        is_cls = (task_type.lower() == "classification")
        df = generate_sample_dataset(n_samples=100000, n_features=25, is_classification=is_cls)
        target_col = "target"
        
    print(f"--> Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle non-numeric columns if any
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_dev = device_utils.xgb_device()
    print(f"--> Train set: {X_train.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples")
    print(f"--> Training with XGBoost (device='{xgb_dev}', tree_method='hist')...")
    
    t0 = time.time()
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    # Native train API: the sklearn fit() in XGBoost 3.x takes no callbacks.
    if task_type.lower() == "classification":
        n_classes = len(np.unique(y_train))
        params = {
            "device": xgb_dev, "tree_method": "hist", "max_bin": 256,
            "subsample": 0.8, "colsample_bytree": 0.8, "max_depth": max_depth,
            "eta": learning_rate, "eval_metric": "logloss", "seed": 42,
            "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
        }
        if n_classes > 2:
            params["num_class"] = n_classes
        model = xgb.train(params, dtrain, num_boost_round=n_estimators,
                          evals=[(dtest, "test")], verbose_eval=50,
                          callbacks=[TabularThermalCallback()])
        fit_time = time.time() - t0

        proba = model.predict(dtest)
        preds = proba.argmax(axis=1) if proba.ndim > 1 else (proba > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        print("-" * 60)
        print(f"--> Training completed in {fit_time:.2f} seconds!")
        print(f"--> Test Accuracy: {acc * 100:.2f}%")
        print("\nClassification Report:\n", classification_report(y_test, preds))
        
        output_file = "model_xgboost_classifier.json"
    else:
        params = {
            "device": xgb_dev, "tree_method": "hist", "max_depth": max_depth,
            "eta": learning_rate, "eval_metric": "rmse", "seed": 42,
            "objective": "reg:squarederror",
        }
        model = xgb.train(params, dtrain, num_boost_round=n_estimators,
                          evals=[(dtest, "test")], verbose_eval=50,
                          callbacks=[TabularThermalCallback()])
        fit_time = time.time() - t0

        preds = model.predict(dtest)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print("-" * 60)
        print(f"--> Training completed in {fit_time:.2f} seconds!")
        print(f"--> Test R2 Score : {r2:.4f}")
        print(f"--> Test RMSE     : {rmse:.4f}")
        
        output_file = "model_xgboost_regressor.json"

    model.save_model(output_file)
    print(f"--> Model successfully saved to: {output_file}")
    print("=" * 60)
    return model, output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPU-accelerated XGBoost model")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="target", help="Name of target column")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "regression"])
    parser.add_argument("--n_estimators", type=int, default=200)
    args = parser.parse_args()
    
    train_tabular_model(args.csv, args.target, args.task, args.n_estimators)
