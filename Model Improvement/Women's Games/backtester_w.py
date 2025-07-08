import os
import logging
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from matplotlib import pyplot as plt

@dataclass
class BacktestConfig:
    # Path to merged data containing features, odds, and actual results
    merged_data_path: str = "/Users/zacharias/Dropbox/Python/Project-NCAA/Model Improvement/Women's Games/merged_data.parquet"
    # Directory containing trained model artifacts
    model_dir: str = "/Users/zacharias/Dropbox/Python/Project-NCAA/Prediction/model_artifacts"
    # Output path for backtest results
    output_path: str = "/Users/zacharias/Dropbox/Python/Project-NCAA/Model Improvement/Women's Games/backtest_results.parquet"
    # Backtesting parameters
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    minimum_kelly: float = 0.01
    kelly_multiplier: float = 1/40
    kelly_rounding_interval: float = 0.005
    minimum_prediction_confidence: float = 0.6
    odds_slippage: float = 0.01

class MergedDataBacktester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.setup_logging()
        self.load_data()
        self.load_model()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def load_data(self):
        logging.info(f"Loading merged data from {self.cfg.merged_data_path}")
        df = pd.read_parquet(self.cfg.merged_data_path)
        df['date'] = pd.to_datetime(df['date'])
        # filter date range
        df = df[(df['date'] >= self.cfg.start_date) & (df['date'] <= self.cfg.end_date)]
        self.data = df.reset_index(drop=True)
        logging.info(f"Loaded {len(self.data)} records for backtesting")

    def load_model(self):
        logging.info(f"Loading model from {self.cfg.model_dir}")
        files = [f for f in os.listdir(self.cfg.model_dir) if f.startswith('production_model_')]
        if not files:
            raise FileNotFoundError("No production_model_ found in model_dir")
        model_file = sorted(files)[-1]
        self.model = xgb.Booster()
        self.model.load_model(os.path.join(self.cfg.model_dir, model_file))
        logging.info(f"Loaded model: {model_file}")

    def predict(self):
        logging.info("Generating predictions...")
        # Exclude non-feature columns
        exclude = [
            'date', 'event_date','event_time',
            'home_team','away_team', 'home_team_won',
            'home_team_proc','away_team_proc',
            'home_odds','away_odds',
            'home_team_score','away_team_score',
            'event_date_x', 'event_date_y'
        ]
        feature_cols = [c for c in self.data.columns if c not in exclude]
        X = self.data[feature_cols].copy()
        if 'gender' in X.columns:
            X['gender'] = (X['gender']=='m').astype(int)
        dmat = xgb.DMatrix(X)
        self.data['predicted_home_win_probability'] = self.model.predict(dmat)
        self.data['predicted_away_win_probability'] = 1 - self.data['predicted_home_win_probability']
        self.data['predicted_outcome'] = (self.data['predicted_home_win_probability'] > 0.5).astype(int)
        logging.info("Predictions added to data")

    def run_backtest(self):
        self.predict()
        bankroll = 10000.0
        running_pnl = 0.0
        records = []
        for _, row in tqdm(self.data.iterrows()):
            ph = row['predicted_home_win_probability']
            pa = row['predicted_away_win_probability']
            # confidence filter
            if max(ph, pa) < self.cfg.minimum_prediction_confidence:
                continue
            oh = row['home_odds'] * (1 - self.cfg.odds_slippage)
            oa = row['away_odds'] * (1 - self.cfg.odds_slippage)
            # Kelly fraction
            qh = 1 - ph
            qa = 1 - pa
            kf_home = self.cfg.kelly_multiplier * ((ph*(oh-1) - qh) / (oh-1))
            kf_away = self.cfg.kelly_multiplier * ((pa*(oa-1) - qa) / (oa-1))
            # Round and threshold
            kf_home = max(0, round(kf_home / self.cfg.kelly_rounding_interval) * self.cfg.kelly_rounding_interval)
            kf_away = max(0, round(kf_away / self.cfg.kelly_rounding_interval) * self.cfg.kelly_rounding_interval)
            if max(kf_home, kf_away) < self.cfg.minimum_kelly:
                continue
                
            kf_home = min(kf_home, 0.03)
            kf_away = min(kf_away, 0.03)

            bet_home = kf_home * bankroll
            bet_away = kf_away * bankroll
            hw = int(row['home_team_score'] > row['away_team_score'])
            pnl = bet_home*(oh-1) - bet_away if hw else bet_away*(oa-1) - bet_home
            bankroll += pnl
            running_pnl += pnl
            records.append({
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'home_odds': oh,
                'away_odds': oa,
                'predicted_home_win_probability': ph,
                'predicted_away_win_probability': pa,
                'bet_home': bet_home,
                'bet_away': bet_away,
                'pnl': pnl,
                'bankroll': bankroll
            })
        results = pd.DataFrame(records)
        logging.info(f"Backtest complete: {len(results)} bets placed, total P&L={results['pnl'].sum():.2f}")
        self.results = results
        return results

    def save_results(self):
        logging.info(f"Saving backtest results to {self.cfg.output_path}")
        self.results.to_parquet(self.cfg.output_path, index=False)
        logging.info("Results saved")

if __name__=='__main__':
    cfg = BacktestConfig()
    bt = MergedDataBacktester(cfg)
    res = bt.run_backtest()
    plt.plot(res['date'], res['bankroll'])
    plt.title('Backtest Bankroll Over Time')
    plt.xlabel('Date')
    plt.ylabel('Bankroll')
    plt.grid()
    plt.show()
    bt.save_results()
