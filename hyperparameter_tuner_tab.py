"""
hyperparameter_tuner_tab.py - Automated Hyperparameter Optimization Studio (Powered by Optuna).
Uses Bayesian Optimization (TPE) on CUDA to automatically find the highest-performing training parameters.
"""

import time
import optuna
import pandas as pd
import gradio as gr
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import train_tabular
import trainer_utils

# Silence optuna logs to avoid clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_bayesian_optimization(task_type, n_trials, csv_file):
    trainer_utils.clear_gpu_memory()
    trials_count = int(n_trials)
    
    # 1. Load or generate data
    csv_path = csv_file.name if csv_file is not None else None
    if csv_path:
        df = pd.read_csv(csv_path)
        target_col = df.columns[-1]
    else:
        df = train_tabular.generate_sample_dataset(n_samples=25000, n_features=15, is_classification=True)
        target_col = "target"
        
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trial_records = []
    
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        n_estimators = trial.suggest_int("n_estimators", 50, 250, step=25)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        
        clf = xgb.XGBClassifier(
            device="cuda",
            tree_method="hist",
            max_bin=256,
            learning_rate=lr,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            random_state=42
        )
        clf.fit(X_train, y_train, verbose=False)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        trial_records.append({
            "Trial": trial.number + 1,
            "Accuracy (%)": round(acc * 100, 2),
            "Learning Rate": round(lr, 4),
            "Max Depth": max_depth,
            "Trees": n_estimators,
            "Subsample": round(subsample, 2)
        })
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials_count)
    
    best = study.best_trial
    summary_md = f"""### 🏆 Optimal Hyperparameters Found:
- **Best Accuracy Score:** `{best.value * 100:.2f}%`
- **Optimal Learning Rate:** `{best.params['learning_rate']:.4f}`
- **Optimal Max Depth:** `{best.params['max_depth']}`
- **Optimal Tree Estimators:** `{best.params['n_estimators']}`
- **Optimal Subsample:** `{best.params['subsample']:.2f}`
"""
    results_df = pd.DataFrame(trial_records).sort_values(by="Accuracy (%)", ascending=False)
    return summary_md, results_df

def build_tab():
    with gr.TabItem("🎯 Auto-tune", id="tuner"):
        gr.Markdown("### 🎯 Automated Hyperparameter Optimization (Optuna + GPU)")
        gr.Markdown("Let AI automatically discover the highest-performing hyperparameters via Bayesian Optimization.")
        
        with gr.Row():
            with gr.Column(scale=3):
                model_target = gr.Dropdown(
                    choices=["XGBoost Tabular Classifier", "LightGBM Gradient Boosting", "Neural Network"],
                    value="XGBoost Tabular Classifier",
                    label="Model to Optimize"
                )
                csv_upload = gr.File(label="Upload CSV (Leave empty to use 25k benchmark dataset)", file_types=[".csv"])
                trials_slider = gr.Slider(minimum=5, maximum=30, step=5, value=10, label="Number of Bayesian Search Trials")
                start_tune_btn = gr.Button("🚀 Run Bayesian Auto-Tuning on GPU", variant="primary")
                
            with gr.Column(scale=3):
                best_summary = gr.Markdown("ℹ️ Configure trials and click **Run Bayesian Auto-Tuning**.")
                
        trials_table = gr.DataFrame(
            headers=["Trial", "Accuracy (%)", "Learning Rate", "Max Depth", "Trees", "Subsample"],
            label="Optuna Trial Leaderboard (Sorted by Best Score)",
            interactive=False
        )
        
        start_tune_btn.click(
            run_bayesian_optimization,
            inputs=[model_target, trials_slider, csv_upload],
            outputs=[best_summary, trials_table]
        )
        
    return {
        "start_tune_btn": start_tune_btn
    }
