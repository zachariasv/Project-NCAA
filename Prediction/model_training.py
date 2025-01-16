import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import yaml
import logging
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, config_path: str):
        """Initialize ModelTrainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.load_data()
        self.load_best_params()
    
    def setup_logging(self):
        """Configure logging for the training process."""
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
        
        logging.info(f"Loaded data with {len(self.df)} rows and {len(self.feature_cols)} features")
    
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
        
        # Add fixed parameters that weren't part of tuning
        self.best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': 42
        })
        
        logging.info(f"Loaded best parameters from tuning results from {latest_tuning}")
    
    def train_model(self):
        """Train the final model using full dataset."""
        logging.info("Starting final model training...")
        
        # Prepare full dataset
        X = self.df[self.feature_cols]
        y = self.df['home_team_won']
        
        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        
        # Train model
        model = xgb.train(
            self.best_params,
            dtrain,
            num_boost_round=1000,  # Can be adjusted based on tuning results
            verbose_eval=100
        )
        
        logging.info("Model training completed")
        
        # Save model with timestamp and parameters
        BASE_DIRECTORY = os.path.dirname(__file__) + "/"
        model_dir = BASE_DIRECTORY + "model_artifacts"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_metadata = {
            'timestamp': timestamp,
            'parameters': self.best_params,
            'feature_columns': self.feature_cols,
            'training_data_end_date': str(self.df['date'].max()),
            'n_training_samples': len(self.df),
            'model_version': f'v1.0_{timestamp}',
            'model_type': 'xgboost'
        }
        
        # Save model and metadata
        model_path = f"{model_dir}/production_model_{timestamp}.json"
        metadata_path = f"{model_dir}/model_metadata_{timestamp}.joblib"
        
        # Save model in JSON format for better compatibility
        model.save_model(model_path)
        joblib.dump(model_metadata, metadata_path)
        
        logging.info(f"Saved production model to {model_path}")
        logging.info(f"Saved model metadata to {metadata_path}")
        logging.info("\nModel training completed successfully!")
        
        # Log key information about the trained model
        logging.info("\nModel Training Summary:")
        logging.info(f"Model Version: v1.0_{timestamp}")
        logging.info(f"Training Data End Date: {model_metadata['training_data_end_date']}")
        logging.info(f"Number of Training Samples: {model_metadata['n_training_samples']}")
        logging.info("\nModel Parameters:")
        for param, value in self.best_params.items():
            logging.info(f"{param}: {value}")
        
        # Save feature importance
        importance_scores = model.get_score(importance_type='gain')
        importance_df = pd.DataFrame(
            list(importance_scores.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        logging.info("\nTop 10 Most Important Features:")
        for _, row in importance_df.head(10).iterrows():
            logging.info(f"{row['Feature']}: {row['Importance']:.4f}")
        
        return model, model_metadata

def main():
    BASE_DIRECTORY = os.path.dirname(__file__) + "/"
    CONFIG_PATH = BASE_DIRECTORY + "config.yaml"
    
    trainer = ModelTrainer(CONFIG_PATH)
    model, metadata = trainer.train_model()
    
    logging.info("\nTraining process completed successfully!")

if __name__ == "__main__":
    main()