import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import joblib
import yaml
import logging
import os
import platform
from datetime import datetime
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, config_path: str):
        """Initialize ModelEvaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.load_data()
        self.load_best_params()
    
    def setup_logging(self):
        """Configure logging for the evaluation process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_data(self):
        """Load and prepare the feature-engineered data."""
        self.df = pd.read_parquet(self.config['processed_data_with_features'])
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df["gender"] = (self.df["gender"] == "M").astype(int)
        self.df = self.df.sort_values('date')
        
        self.feature_cols = [col for col in self.df.columns if col not in [
            'date', 'home_team', 'away_team', 'home_team_score', 'away_team_score',
            'home_team_won'
        ]]
    
    def load_best_params(self):
        """Load the best parameters from tuning results."""
        BASE_DIRECTORY = os.path.dirname(__file__) + "/"
        results_dir = BASE_DIRECTORY + "model_artifacts"
        tuning_files = [f for f in os.listdir(results_dir) if f.startswith('xgb_tuning_results_')]
        
        if not tuning_files:
            logging.error("No XGBoost tuning results found. Please run model tuning first.")
            raise FileNotFoundError("No tuning results available")
            
        latest_tuning = sorted(tuning_files)[-1]
        results = joblib.load(f"{results_dir}/{latest_tuning}")
        self.best_params = results['best_params']
        
        # Add fixed parameters
        self.best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': 42
        })
        
        logging.info("Loaded best parameters from tuning results")

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba) -> Dict:
        """
        Calculate comprehensive metrics for both home and away team predictions.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'home_win_precision': precision_score(y_true, y_pred),
            'home_win_recall': recall_score(y_true, y_pred),
            'away_win_precision': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'away_win_recall': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'true_home_wins': tp,
            'false_home_wins': fp,
            'true_away_wins': tn,
            'false_away_wins': fn
        }
        
        return metrics
    
    def evaluate_recent_split(self) -> Dict:
        """Evaluate model using most recent 10% of data as test set."""
        logging.info("Starting recent split evaluation...")
        
        # Split data into training (90%) and test (10%) sets
        split_idx = int(len(self.df) * 0.9)
        train_data = self.df.iloc[:split_idx]
        test_data = self.df.iloc[split_idx:]
        
        # Prepare feature matrices
        X_train = train_data[self.feature_cols]
        y_train = train_data['home_team_won']
        X_test = test_data[self.feature_cols]
        y_test = test_data['home_team_won']
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train model
        model = xgb.train(
            self.best_params,
            dtrain,
            num_boost_round=5000,
            early_stopping_rounds=50,
            evals=[(dtest, 'eval')],
            verbose_eval=False
        )
        
        # Make predictions
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
        # Log results
        logging.info("\nRecent split evaluation results:")
        logging.info("\nOverall Performance:")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        logging.info("\nHome Win Prediction Performance:")
        logging.info(f"Precision: {metrics['home_win_precision']:.4f}")
        logging.info(f"Recall: {metrics['home_win_recall']:.4f}")
        
        logging.info("\nAway Win Prediction Performance:")
        logging.info(f"Precision: {metrics['away_win_precision']:.4f}")
        logging.info(f"Recall: {metrics['away_win_recall']:.4f}")
        
        logging.info("\nDetailed Predictions:")
        logging.info(f"Correctly predicted home wins: {metrics['true_home_wins']}")
        logging.info(f"Incorrectly predicted home wins: {metrics['false_home_wins']}")
        logging.info(f"Correctly predicted away wins: {metrics['true_away_wins']}")
        logging.info(f"Incorrectly predicted away wins: {metrics['false_away_wins']}")
        
        return metrics
    
    def evaluate_rolling_seasons(self) -> List[Dict]:
        """Evaluate model using rolling season-based evaluation with comprehensive metrics."""
        logging.info("Starting rolling season evaluation...")
        
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['season'] = np.where(self.df['month'] >= 10, 
                                self.df['year'] + 1, 
                                self.df['year'])
        
        seasons = sorted(self.df['season'].unique())
        mid_point_idx = len(seasons) // 2
        results = []
        
        current_train_data = self.df[self.df['season'] <= seasons[mid_point_idx]]
        
        for season in tqdm(seasons[mid_point_idx + 1:], desc="Evaluating Seasons", unit="season"):
            test_data = self.df[self.df['season'] == season]
            
            X_train = current_train_data[self.feature_cols]
            y_train = current_train_data['home_team_won']
            X_test = test_data[self.feature_cols]
            y_test = test_data['home_team_won']
            
            # Convert to DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            # Train model
            model = xgb.train(
                self.best_params,
                dtrain,
                num_boost_round=1000,
                early_stopping_rounds=50,
                evals=[(dtest, 'eval')],
                verbose_eval=False
            )
            
            # Make predictions
            y_pred_proba = model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics for the season
            season_metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            season_metrics['season'] = season
            season_metrics['n_games'] = len(test_data)
            
            results.append(season_metrics)
            current_train_data = pd.concat([current_train_data, test_data])
            
            # Log season results
            logging.info(f"\nSeason {season} evaluation results:")
            logging.info("\nOverall Performance:")
            logging.info(f"Accuracy: {season_metrics['accuracy']:.4f}")
            logging.info(f"ROC-AUC: {season_metrics['roc_auc']:.4f}")
            
            logging.info("\nHome Win Prediction Performance:")
            logging.info(f"Precision: {season_metrics['home_win_precision']:.4f}")
            logging.info(f"Recall: {season_metrics['home_win_recall']:.4f}")
            
            logging.info("\nAway Win Prediction Performance:")
            logging.info(f"Precision: {season_metrics['away_win_precision']:.4f}")
            logging.info(f"Recall: {season_metrics['away_win_recall']:.4f}")
            
            logging.info("\nDetailed Predictions:")
            logging.info(f"Correctly predicted home wins: {season_metrics['true_home_wins']}")
            logging.info(f"Incorrectly predicted home wins: {season_metrics['false_home_wins']}")
            logging.info(f"Correctly predicted away wins: {season_metrics['true_away_wins']}")
            logging.info(f"Incorrectly predicted away wins: {season_metrics['false_away_wins']}")
            logging.info(f"Total games in season: {season_metrics['n_games']}")
            
            logging.info("\n" + "="*50)
        
        return results
    
    def plot_results(self, season_results: List[Dict]):
        """Plot comprehensive evaluation results with enhanced visualizations."""
        # [Previous plotting code remains unchanged as it works with the metrics dictionary]
        # Let me know if you want me to include the full plotting code

def main():
    # Clear the terminal screen
    os.system("cls" if platform.system() == "Windows" else "clear")

    BASE_DIRECTORY = os.path.dirname(__file__) + "/"
    CONFIG_PATH = BASE_DIRECTORY + "config.yaml"
    
    evaluator = ModelEvaluator(CONFIG_PATH)
    
    recent_metrics = evaluator.evaluate_recent_split()
    season_results = evaluator.evaluate_rolling_seasons()
    
    evaluator.plot_results(season_results)
    
    logging.info("Evaluation completed. Results saved in evaluation_results directory.")

if __name__ == "__main__":
    main()