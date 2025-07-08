import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import yaml
from matplotlib import pyplot as plt

class BettingBacktest:
    def __init__(self, config_path: str, bankroll: float = 10_000):
        """Initialize backtesting with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
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
        """Load the necessary datasets."""
        # Load match results
        self.match_results = pd.read_parquet(self.config["processed_data_path"])
        self.match_results = self.match_results.loc[self.match_results.date >= pd.to_datetime("2024-01-01")] 
        self.match_results["date"] = pd.to_datetime(self.match_results["date"])
        
        # Load team name mappings
        team_mappings_df = pd.read_csv(self.config["name_matching_path"])
        self.team_mappings = dict(zip(team_mappings_df["original_name"], team_mappings_df["best_match"]))
    
    def load_historical_bets(self, bets_folder: str) -> pd.DataFrame:
        """Load and preprocess historical bets data from multiple files."""
        all_bets = []
        for filename in os.listdir(bets_folder):
            if filename.endswith(".parquet"):
                file_path = os.path.join(bets_folder, filename)
                try:
                    bets = pd.read_parquet(file_path)
                    bets["commence_time"] = pd.to_datetime(bets["commence_time"])
                    bets["prediction_date"] = pd.to_datetime(bets["prediction_date"])
                    all_bets.append(bets)
                    logging.info(f"Loaded bets from {filename}")
                except Exception as e:
                    logging.error(f"Error loading {filename}: {str(e)}")
        
        if not all_bets:
            raise ValueError("No valid bet files found in folder")
            
        combined_bets = pd.concat(all_bets, ignore_index=True)
        combined_bets["commence_time"] = pd.to_datetime(combined_bets["commence_time"]).dt.tz_localize(None)
        combined_bets.sort_values("commence_time", inplace=True, ascending=True)
        combined_bets.reset_index(drop=True, inplace=True)
        logging.info(f"Total bets loaded: {len(combined_bets)}")
        return combined_bets
    
    def get_match_result(self, row: pd.Series) -> tuple:
        """Get the actual match result for a bet."""
        try:
            home_team = self.team_mappings[row["home_team"]]
            away_team = self.team_mappings[row["away_team"]]
            match_date = row["commence_time"].date()

            result = self.match_results.loc[
                ((self.match_results["date"].dt.date - match_date).abs() <= pd.Timedelta(days = 1)) &
                (self.match_results["home_team"] == home_team) &
                (self.match_results["away_team"] == away_team)
            ].iloc[0]
            
            home_win = int(result["home_team_score"] > result["away_team_score"])
            return home_win, True
            
        except (IndexError, KeyError):
            logging.warning(f"No match result found for {row['home_team']} vs {row['away_team']} on {match_date}. Search parameters: {home_team}, {away_team}, {match_date}")
            return None, False
    
    def calculate_bet_outcome(self, row: pd.Series, home_win: int, slippage: float = 0.0) -> float:
        """Calculate the profit/loss for a bet."""
        initial_bankroll = self.bankroll  # Starting bankroll for percentage calculations
        
        # Calculate bet amounts based on Kelly criterion
        home_bet = row["home_kelly"] * initial_bankroll
        away_bet = row["away_kelly"] * initial_bankroll
        
        if home_win == 1:
            return (home_bet * (row["best_home_odds"]*(1-slippage) - 1)) - away_bet
        else:
            return (away_bet * (row["best_away_odds"]*(1-slippage) - 1)) - home_bet
        
    def round_bet_size(self, bet_size: float, rounding_interval: float = None) -> float:
        """Round the bet size to the nearest increment"""
        return rounding_interval * round(bet_size / rounding_interval)
    
    def run_backtest(self, bets_folder: str, start_date=None, end_date=None, minimum_kelly_threshold: float = 0.0, kelly_rounding_interval: float = 0.0, minimum_prediction_confidence: float = 0.5, slippage: float = 0.0) -> pd.DataFrame:
        """Run the backtest over the specified period."""
        bets = self.load_historical_bets(bets_folder)
        
        # Filter by date range if specified
        if start_date:
            bets = bets[bets["commence_time"] >= pd.to_datetime(start_date)]
        if end_date:
            bets = bets[bets["commence_time"] <= pd.to_datetime(end_date)]
        
        results = []
        running_pnl = 0
        
        for _, bet in bets.iterrows():
            home_win, found = self.get_match_result(bet)
            
            if kelly_rounding_interval and kelly_rounding_interval > 0:
                bet["home_kelly"] = self.round_bet_size(bet["home_kelly"], kelly_rounding_interval)
                bet["away_kelly"] = self.round_bet_size(bet["away_kelly"], kelly_rounding_interval)            

            if found and ((bet["home_kelly"] >= minimum_kelly_threshold) or (bet["away_kelly"] >= minimum_kelly_threshold)) and ((bet["predicted_home_win_probability"] >= minimum_prediction_confidence) or (bet["predicted_away_win_probability"] >= minimum_prediction_confidence)):
                pnl = self.calculate_bet_outcome(bet, home_win, slippage)
                running_pnl += pnl
                
                results.append({
                    "date": bet["commence_time"],
                    "bet_time": bet["prediction_date"],
                    "home_team": bet["home_team"],
                    "away_team": bet["away_team"],
                    "home_odds": bet["best_home_odds"],
                    "away_odds": bet["best_away_odds"],
                    "home_kelly": bet["home_kelly"],
                    "away_kelly": bet["away_kelly"],
                    "bookmaker": bet["best_home_bookmaker"] if bet["home_kelly"] > bet["away_kelly"] else bet["best_away_bookmaker"],
                    "match_found": True,
                    "prediction": int(bet["predicted_outcome"]),
                    "prediction_probability": bet["predicted_home_win_probability"],
                    "home_win": home_win,
                    "bet_size": max([bet["home_kelly"]*self.bankroll, bet["away_kelly"]*self.bankroll]),
                    "pnl": pnl,
                    "running_pnl": running_pnl
                })
            else:
                results.append({
                    "date": bet["commence_time"],
                    "bet_time": bet["prediction_date"],
                    "home_team": bet["home_team"],
                    "away_team": bet["away_team"],
                    "home_odds": bet["best_home_odds"],
                    "away_odds": bet["best_away_odds"],
                    "home_kelly": bet["home_kelly"],
                    "away_kelly": bet["away_kelly"],
                    "match_found": False,
                    "home_win": None,
                    "bet_size": max([bet["home_kelly"]*self.bankroll, bet["away_kelly"]*self.bankroll]),
                    "pnl": 0,
                    "running_pnl": running_pnl
                })
        
        results_df = pd.DataFrame(results).drop_duplicates(subset = ["home_team", "away_team", "date"]).reset_index(drop=True)
        self.log_backtest_summary(results_df)
        return results_df
    
    def calculate_sharpe_ratio(self, pnl: pd.Series, days_of_period: float = 150) -> float:
        '''Calculate the yearly Sharpe ratio'''
        pnl = pnl / self.bankroll # Make P&L relative to bankroll, equivalent to percentage returns
        pnl_mean = pnl.mean()
        pnl_std = pnl.std()

        if pnl_std > 0:
            sharpe_ratio = (pnl_mean / pnl_std) * np.sqrt(days_of_period)
            return sharpe_ratio
        else:
            return None
    
    def calculate_max_drawdown(self, pnl: pd.Series) -> float:
        '''Calculate the maximum drawdown'''

        cumulative_pnl = pnl.cumsum() + self.bankroll
        drawdown = (cumulative_pnl - cumulative_pnl.cummax())# / cumulative_pnl.cummax()
        max_drawdown = drawdown.min()
        
        return max_drawdown

    def log_backtest_summary(self, results: pd.DataFrame):
        """Log summary statistics for the backtest."""
        matched_bets = results[results["match_found"]].reset_index(drop=True)

        plt.figure(figsize=(12, 6))
        plt.plot(matched_bets.index, matched_bets["running_pnl"])
        plt.xticks(ticks=matched_bets.index[::max(1, len(matched_bets)//10)], labels=matched_bets["date"].dt.strftime('%Y-%m-%d')[::max(1, len(matched_bets)//10)], rotation=45, ha='right')
        plt.xlabel("Match")
        plt.ylabel("Profit / Loss")
        plt.title("P&L")
        plt.tight_layout()
        plt.show()
        
        logging.info("\nBacktest Summary:")
        logging.info(f"Total bets analyzed: {len(results)}")
        logging.info(f"Matches found: {len(matched_bets)}")
        logging.info(f"Total return: {(matched_bets['pnl'].sum()+self.bankroll)/self.bankroll -1:.2%}")
        logging.info(f"Total P&L: {int(matched_bets['pnl'].sum())} SEK")
        logging.info(f"Prediction accuracy: {(matched_bets['prediction'] == matched_bets['home_win']).astype(int).mean():.2%}"),
        logging.info(f"Win rate: {(matched_bets['pnl'] > 0).mean():.2%}")
        logging.info(f"Sharpe ratio: {self.calculate_sharpe_ratio(matched_bets[['date', 'pnl']].set_index('date').resample('1D').sum()['pnl']):.2f}")
        logging.info(f"Max drawdown: {int(self.calculate_max_drawdown(matched_bets['pnl']))} SEK")
        logging.info(f"Bookmaker frequency:\n{matched_bets['bookmaker'].value_counts()}")
    
    def generate_pnl_histograms(self, results: pd.DataFrame, bins: int = 10):
        """
        Generate histograms of PNL contributions for various ranges of Kelly criterion, 
        prediction probabilities, and bookmakers.
        :param results: Backtest results DataFrame.
        :param bins: Number of bins for the histograms.
        """
        matched_bets = results.loc[results["match_found"] & (results["pnl"] != 0)].copy()

        # Kelly criterion ranges
        kelly_max = matched_bets[["home_kelly", "away_kelly"]].max(axis=1)
        matched_bets["kelly_range"] = pd.cut(
            kelly_max,
            bins=bins,
            include_lowest=True,
        )
        
        # Prediction probability ranges
        matched_bets["probability_range"] = pd.cut(
            matched_bets["prediction_probability"],
            bins=bins,
            include_lowest=True,
        )

        # Aggregate PNL by Kelly ranges
        kelly_pnl = matched_bets.groupby("kelly_range", observed=False)["pnl"].sum()

        # Aggregate PNL by probability ranges
        probability_pnl = matched_bets.groupby("probability_range", observed=False)["pnl"].sum()

        # Aggregate PNL by bookmaker
        bookmaker_pnl = matched_bets.groupby("bookmaker")["pnl"].sum()

        # Plot histograms
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=False)

        # Histogram for Kelly criterion
        axes[0].bar(
            [str(interval) for interval in kelly_pnl.index], kelly_pnl.values
        )
        axes[0].set_title("PNL Contribution by Kelly Criterion Range")
        axes[0].set_xlabel("Kelly Criterion Range")
        axes[0].set_ylabel("Total PNL")
        axes[0].tick_params(axis="x", rotation=45)

        # Histogram for prediction probabilities
        axes[1].bar(
            [str(interval) for interval in probability_pnl.index], probability_pnl.values
        )
        axes[1].set_title("PNL Contribution by Prediction Probability Range")
        axes[1].set_xlabel("Prediction Probability Range")
        axes[1].set_ylabel("Total PNL")
        axes[1].tick_params(axis="x", rotation=45)

        # Histogram for bookmaker
        axes[2].bar(bookmaker_pnl.index, bookmaker_pnl.values)
        axes[2].set_title("PNL Contribution by Bookmaker")
        axes[2].set_xlabel("Bookmaker")
        axes[2].set_ylabel("Total PNL")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def generate_hourly_profit_histogram(self, results: pd.DataFrame):
        """
        Generate a histogram of profit per bet grouped by the hour of the day.
        :param results: Backtest results DataFrame.
        """
        matched_bets = results.loc[results["match_found"] & (results["pnl"] != 0)].copy()

        # Extract the hour from the commence_time
        matched_bets["bet_hour"] = pd.to_datetime(matched_bets["bet_time"]).dt.hour

        # Aggregate PNL and number of bets per hour
        hourly_stats = matched_bets.groupby("bet_hour").agg(
            total_pnl=("pnl", "sum"),
            total_bets=("pnl", "count"),
            total_bet_size=("bet_size", "sum")
        )

        # Calculate profit per bet
        hourly_stats["profit"] = hourly_stats["total_pnl"] / hourly_stats["total_bet_size"]

        # Plot the histogram
        plt.figure(figsize=(12, 6))
        plt.bar(hourly_stats.index, hourly_stats["profit"])
        plt.title("Profit by Hour of the Day")
        plt.xlabel("Hour of the Day (24-hour format)")
        plt.ylabel("Profit")
        plt.xticks(range(24))  # Ensure all hours are shown on x-axis
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


def main():
    # Get the directory of the current script
    BASE_DIRECTORY = os.path.dirname(__file__) + "/"
    CONFIG_PATH = BASE_DIRECTORY + "config.yaml"
    
    # Initialize backtester
    backtester = BettingBacktest(CONFIG_PATH)
    
    # Example usage with date range

    results = backtester.run_backtest(
        bets_folder=BASE_DIRECTORY + "Historical Bet Opportunities/",
        start_date="2024-01-01",
        end_date="2025-12-31",
        minimum_kelly_threshold = 0.015,
        kelly_rounding_interval = 0.005,
        minimum_prediction_confidence = 0,
        slippage = 0
        )
    #backtester.generate_pnl_histograms(results, bins = 10)
    backtester.generate_hourly_profit_histogram(results)

    # Save results
    #results.to_csv(BASE_DIRECTORY + "backtest_results.csv", index=False)
    #logging.info("Results saved to backtest_results.csv")

if __name__ == "__main__":
    main()