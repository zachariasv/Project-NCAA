import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import optuna
import joblib
import yaml
import logging
import os
import platform
from datetime import datetime

class ModelTuner:
    def __init__(self, config_path: str):
        """Initialize ModelTuner with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.load_data()
    
    def setup_logging(self):
        """Configure logging for the tuning process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_data(self):
        """Load and prepare the feature-engineered data."""
        # Load data and sort by date
        self.df = pd.read_parquet(self.config['processed_data_with_features'])
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df["gender"] = (self.df["gender"] == "M").astype(int)
        self.df = self.df.sort_values('date')

        # Define feature columns
        self.feature_cols = [col for col in self.df.columns if col not in [
            'date', 'home_team', 'away_team', 'home_team_score', 'away_team_score',
            'home_team_won'
        ]]
        
        # Get season boundaries
        last_date = self.df['date'].max()
        current_season_start = last_date - pd.DateOffset(months=6)
        previous_season_end = current_season_start
        previous_season_start = previous_season_end - pd.DateOffset(months=12)
        
        # Create masks for different time periods
        current_season_mask = self.df['date'] > current_season_start
        previous_season_mask = (self.df['date'] > previous_season_start) & (self.df['date'] <= previous_season_end)
        training_mask = self.df['date'] <= previous_season_start
        
        # Split the data
        train_df = self.df[training_mask]  # All data before previous season
        val_df = self.df[previous_season_mask]  # Previous season as validation
        current_df = self.df[current_season_mask]  # Current season (excluded from training/validation)
        
        # Prepare feature matrices
        self.X_train = train_df[self.feature_cols]
        self.y_train = train_df['home_team_won']
        self.X_val = val_df[self.feature_cols]
        self.y_val = val_df['home_team_won']
        
        # Convert to DMatrix for faster XGBoost training
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dval = xgb.DMatrix(self.X_val, label=self.y_val)
        
        logging.info(f"Training data (before {previous_season_start.date()}): {len(self.X_train)} samples")
        logging.info(f"Validation data ({previous_season_start.date()} to {previous_season_end.date()}): {len(self.X_val)} samples")
        logging.info(f"Current season data (excluded): {len(current_df)} samples")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for XGBoost optimization."""
        param = {
            # Suggested broader ranges
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0, log=True),
            'eta': trial.suggest_float('eta', 0.001, 0.5, log=True),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'min_split_loss': trial.suggest_float('min_split_loss', 0.0, 5.0),
            'max_leaves': trial.suggest_int('max_leaves', 0, 128),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': 42
        }
        
        # Train with early stopping
        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial, "validation-auc"
        )
        
        evals_result = {}
        bst = xgb.train(
            param,
            self.dtrain,
            num_boost_round=5000,
            evals=[(self.dval, "validation")],
            early_stopping_rounds=50,
            callbacks=[pruning_callback],
            evals_result=evals_result,
            verbose_eval=False
        )
        
        # Return the best validation score
        return evals_result["validation"]["auc"][-1]
    
    def tune_model(self, n_trials: int = 50) -> dict:
        """Run hyperparameter optimization."""
        logging.info("Starting hyperparameter optimization...")
        
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=10
            )
        )
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Save results
        BASE_DIRECTORY = os.path.dirname(__file__) + "/"
        results_dir = BASE_DIRECTORY + "model_artifacts"
        os.makedirs(results_dir, exist_ok=True)
        
        optimization_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        joblib.dump(
            optimization_results,
            f"{results_dir}/xgb_tuning_results_{optimization_results['timestamp']}.joblib"
        )
        
        logging.info(f"Best ROC-AUC: {study.best_value:.4f}")
        logging.info("Best parameters:")
        for param, value in study.best_params.items():
            logging.info(f"{param}: {value}")
        
        return study.best_params

def main():
    # Clear the terminal screen
    os.system("cls" if platform.system() == "Windows" else "clear")

    BASE_DIRECTORY = os.path.dirname(__file__) + "/"
    CONFIG_PATH = BASE_DIRECTORY + "config.yaml"
    tuner = ModelTuner(CONFIG_PATH)

    best_params = tuner.tune_model(n_trials=200)
    print("\nOptimization completed successfully!")

if __name__ == "__main__":
    main()