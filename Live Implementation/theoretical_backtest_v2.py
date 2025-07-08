import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import yaml
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    """Configuration class for backtest parameters"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_bankroll: float = 10000.0
    minimum_kelly: float = 0.01
    kelly_rounding_interval: float = 0.005
    minimum_prediction_confidence: float = 0.6
    odds_slippage: float = 0.01
    allowed_bookmakers: Optional[List[str]] = None

class BettingBacktest:
    def __init__(self, config_path: str, start_date: str, bankroll: float = 10000.0):
        """Initialize backtesting with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.start_date = pd.to_datetime(start_date)
        self.bankroll = bankroll

        self.setup_logging()
        self.load_data()

    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        
    def load_data(self):
        """Load historical match results and team name mappings."""
        # Load match results
        self.match_results = pd.read_parquet(self.config["processed_data_path"])
        self.match_results["date"] = pd.to_datetime(self.match_results["date"])
        self.match_results = self.match_results.loc[self.match_results["date"] >= self.start_date]
        
        # Load team name mappings
        team_mappings_df = pd.read_csv(self.config["name_matching_path"])
        self.team_mappings = dict(zip(team_mappings_df["original_name"], team_mappings_df["best_match"]))
        
        logging.info("Loaded historical match results and team mappings")

    def get_prediction_files(self, folder_path: str) -> pd.DataFrame:
        """Get all prediction files and organize by date."""
        prediction_files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
        if not prediction_files:
            raise ValueError("No prediction files found in folder")
        
        # Parse dates from filenames and create dataframe
        file_info = []
        for filename in prediction_files:
            try:
                # Extract date from filename (adjust this based on your filename format)
                date_str = filename.split("_")[3]  # Assuming format like "predictions_20240317_123456.parquet"
                date = pd.to_datetime(date_str).date()
                timestamp = filename.split("_")[4].replace(".parquet", "")  # Get the time part
                
                file_info.append({
                    "filename": filename,
                    "date": date,
                    "timestamp": timestamp
                })
            except (IndexError, ValueError) as e:
                logging.warning(f"Could not parse date from filename {filename}: {e}")
                continue
        
        files_df = pd.DataFrame(file_info)
        
        # Get latest file for each date
        latest_files = files_df.sort_values("timestamp").groupby("date").last()
        logging.info(f"Found {len(latest_files)} unique dates of predictions")
        
        return latest_files["filename"].tolist()

    def load_predictions(self, folder_path: str) -> pd.DataFrame:
        """Load and combine prediction files, keeping latest for each date."""
        filenames = self.get_prediction_files(folder_path)
        
        all_predictions = []
        for filename in filenames:
            try:
                file_path = os.path.join(folder_path, filename)
                df = pd.read_parquet(file_path)
                df["commence_time"] = pd.to_datetime(df["commence_time"]).dt.tz_localize(None)
                df["prediction_date"] = pd.to_datetime(df["prediction_date"])
                all_predictions.append(df)
                logging.info(f"Loaded predictions from {filename}")
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("No prediction data could be loaded")
        
        combined_df = pd.concat(all_predictions, ignore_index=True)
        return combined_df
    
    def apply_slippage(self, odds: float, slippage: float) -> float:
        """Apply slippage to odds to simulate market impact."""
        return odds * (1 - slippage)

    def calculate_kelly_criterion(self, probability: float, odds: float, kelly_fraction: float = 1/40) -> float:
        """Calculate Kelly Criterion bet size."""
        if odds <= 1:
            return 0
        q = 1 - probability
        return (probability * (odds - 1) - q) / (odds - 1) * kelly_fraction

    def round_bet_size(self, bet_size: float, interval: float) -> float:
        """Round the bet size to the nearest interval."""
        if interval <= 0:
            return bet_size
        return interval * round(bet_size / interval)
        
    def get_match_result(self, bet: pd.Series) -> tuple:
        """Get the actual match result for a bet."""
        try:
            home_team = self.team_mappings[bet["home_team"]]
            away_team = self.team_mappings[bet["away_team"]]
            match_date = bet["commence_time"].date()
            
            # Look for match result within 1 day of commence time
            result = self.match_results.loc[
                ((self.match_results["date"].dt.date - match_date).abs() <= pd.Timedelta(days=1)) &
                (self.match_results["home_team"] == home_team) &
                (self.match_results["away_team"] == away_team)
            ].iloc[0]
            
            home_win = int(result["home_team_score"] > result["away_team_score"])
            return home_win, True
            
        except (IndexError, KeyError):
            logging.warning(f"No match result found for {bet['home_team']} vs {bet['away_team']} on {match_date}")
            return None, False

    def run_backtest(self, predictions_folder: str, backtest_config: BacktestConfig) -> pd.DataFrame:
        """Run the backtest with given configuration."""
        predictions = self.load_predictions(predictions_folder)
        
        # Apply date filters
        if backtest_config.start_date:
            predictions = predictions[
                predictions["commence_time"] >= pd.to_datetime(backtest_config.start_date)
            ]
        if backtest_config.end_date:
            predictions = predictions[
                predictions["commence_time"] <= pd.to_datetime(backtest_config.end_date)
            ]
            
        # Apply bookmaker filter
        if backtest_config.allowed_bookmakers:
            # Filter odds columns for allowed bookmakers
            odds_cols = [col for col in predictions.columns if any(
                bm in col for bm in backtest_config.allowed_bookmakers
            ) and "odds" in col]
            predictions = predictions[
                ["commence_time", "home_team", "away_team", "predicted_outcome", "predicted_home_win_probability", "predicted_away_win_probability"] + odds_cols
            ]
        else:
            odds_cols = [col for col in predictions.columns if "odds" in col]
            predictions = predictions[
                ["commence_time", "home_team", "away_team", "predicted_outcome", "predicted_home_win_probability", "predicted_away_win_probability"] + odds_cols
            ]

        results = []
        bankroll = backtest_config.initial_bankroll
        running_pnl = 0

        for _, bet in predictions.iterrows():
            # Apply minimum prediction confidence
            home_prob = bet["predicted_home_win_probability"]
            away_prob = bet["predicted_away_win_probability"]
            
            if max(home_prob, away_prob) < backtest_config.minimum_prediction_confidence:
                continue
            
            # Calculate best available odds
            best_home_odds = bet[[col for col in odds_cols if "home" in col]].max()
            best_away_odds = bet[[col for col in odds_cols if "away" in col]].max()

            # Get best available odds after applying slippage
            best_home_odds = self.apply_slippage(best_home_odds, backtest_config.odds_slippage)
            best_away_odds = self.apply_slippage(best_away_odds, backtest_config.odds_slippage)

            # Calculate Kelly sizes
            home_kelly = self.calculate_kelly_criterion(home_prob, best_home_odds)
            away_kelly = self.calculate_kelly_criterion(away_prob, best_away_odds)

            # Apply minimum Kelly and rounding
            home_kelly = max(0, home_kelly)
            away_kelly = max(0, away_kelly)
            
            if backtest_config.kelly_rounding_interval > 0:
                home_kelly = self.round_bet_size(home_kelly, backtest_config.kelly_rounding_interval)
                away_kelly = self.round_bet_size(away_kelly, backtest_config.kelly_rounding_interval)

            if max(home_kelly, away_kelly) < backtest_config.minimum_kelly:
                continue

            # Calculate bet amounts
            home_bet = home_kelly * bankroll
            away_bet = away_kelly * bankroll

            # Get actual match result
            home_win, match_found = self.get_match_result(bet)
            
            if not match_found:
                logging.warning(f"No match result found for {bet['home_team']} vs {bet['away_team']} on {bet['commence_time']}")
                continue
            
            if True in np.isnan([best_home_odds, best_away_odds, home_bet, away_bet]):
                logging.warning(f"No bets found for {bet['home_team']} vs {bet['away_team']}, skipping due to NaN values")
                continue

            # Calculate P&L based on actual result
            if home_win == 1:
                pnl = (home_bet * (best_home_odds - 1)) - away_bet
            else:
                pnl = (away_bet * (best_away_odds - 1)) - home_bet
                
            # Update running P&L
            running_pnl += pnl

            # Record bet details
            bet_record = {
                "date": bet["commence_time"],
                "home_team": bet["home_team"],
                "away_team": bet["away_team"],
                "predicted_outcome": bet["predicted_outcome"],
                "home_probability": home_prob,
                "away_probability": away_prob,
                "home_win": home_win,
                "home_odds": best_home_odds,
                "away_odds": best_away_odds,
                "home_kelly": home_kelly,
                "away_kelly": away_kelly,
                "home_bet": home_bet,
                "away_bet": away_bet,
                "home_win": home_win,
                "pnl": pnl,
                "running_pnl": running_pnl
            }

            results.append(bet_record)

        results_df = pd.DataFrame(results)
        self.log_backtest_summary(results_df, backtest_config)
        return results_df

    def calculate_performance_metrics(self, results: pd.DataFrame, config: BacktestConfig) -> Dict:
        """Calculate comprehensive performance metrics."""
        metrics = {
            "total_bets": len(results),
            "accuracy": (results["predicted_outcome"] == results["home_win"]).mean(),
            "win_rate": (results["pnl"] > 0).sum() / (results["pnl"] != 0).sum(),
            "total_pnl": results["pnl"].sum(),
            "roi": results["pnl"].sum() / config.initial_bankroll,
            "sharpe_ratio": self.calculate_sharpe_ratio(results.set_index("date")["pnl"]),
            "max_drawdown": self.calculate_max_drawdown(results["pnl"]),
        }
        return metrics

    def calculate_sharpe_ratio(self, pnl_series: pd.Series) -> float:
        """Calculate the annualized Sharpe ratio."""
        daily_returns = pnl_series.groupby(pd.Grouper(freq="D")).sum()
        
        if len(daily_returns) > 1:
            return np.sqrt(150) * (daily_returns.mean() / daily_returns.std()) # Roughly 150 days in a basketball season, making this the equivalent to annualizing
        return 0

    def calculate_max_drawdown(self, pnl_series: pd.Series) -> float:
        """Calculate the maximum drawdown."""
        cumulative = pnl_series.cumsum() + self.bankroll
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min()

    def plot_equity_curve(self, results: pd.DataFrame, config: BacktestConfig):
        """Plot the equity curve with drawdowns."""
        plt.figure(figsize=(12, 6))
        #cumulative_pnl = results.set_index("date").resample("1D").sum()["pnl"].cumsum() + config.initial_bankroll
        cumulative_pnl = results["running_pnl"] + config.initial_bankroll
        plt.plot(cumulative_pnl, label="Equity Curve")
        for i, date in enumerate(results["date"]):
            print(f"{i+1}: {date} - {cumulative_pnl[i]:.2f}")
        plt.title("Backtest Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Bankroll")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def log_backtest_summary(self, results: pd.DataFrame, config: BacktestConfig):
        """Log comprehensive backtest results."""
        metrics = self.calculate_performance_metrics(results, config)
        
        logging.info("\nBacktest Summary:")
        logging.info(f"Period: {results['date'].min().date()} to {results['date'].max().date()}")
        logging.info(f"Total Bets: {metrics['total_bets']}")
        logging.info(f"Prediction Accuracy: {metrics['accuracy']:.2%}")
        logging.info(f"Win Rate: {metrics['win_rate']:.2%}")
        logging.info(f"Total P&L: ${metrics['total_pnl']:,.2f}")
        logging.info(f"ROI: {metrics['roi']:.2%}")
        logging.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logging.info(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")

        self.plot_equity_curve(results, config)

def main():
    BASE_DIRECTORY = os.path.dirname(__file__) + "/"
    CONFIG_PATH = BASE_DIRECTORY + "config.yaml"
    
    # Example configuration
    backtest_config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2025-12-31",
        initial_bankroll=10_000.0,
        minimum_kelly=0.01,
        #kelly_rounding_interval=0.005,
        minimum_prediction_confidence=0.6,
        odds_slippage=0,
        allowed_bookmakers=["marathonbet", "mybookieag", "nordicbet", "betonlineag", "sport888", "pinnacle", "everygame"]
    )
    
    # Initialize and run backtest
    backtester = BettingBacktest(CONFIG_PATH, start_date = backtest_config.start_date)
    results = backtester.run_backtest(
        predictions_folder=BASE_DIRECTORY + "Historical Predictions/",
        backtest_config=backtest_config
    )
    
    # Save results
    results.to_parquet(BASE_DIRECTORY + "backtest_results.parquet")
    logging.info("Results saved to backtest_results.parquet")

if __name__ == "__main__":
    main()