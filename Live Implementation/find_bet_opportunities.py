import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
import subprocess
import os
import logging
import platform
import sys
from fetch_matchups_today import fetch_todays_games

# Import classes from feature engineering script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from Prediction.feature_engineering import *

class UpcomingMatchPredictor:
    def __init__(self, odds_df: pd.DataFrame):
        """Initialize the predictor with necessary components"""
        self.feature_engineer = FeatureEngineering()
        self.games_to_predict = len(odds_df)
        self.sportsreference_matchups = fetch_todays_games()
        self.load_config()
        self.load_model()
        self.load_data(odds_df)
        self.generate_features()

    def load_config(self):
        self.base_directory = os.path.dirname(__file__) + "/"
        with open(self.base_directory + "config.yaml", "r") as f: 
            self.config = yaml.safe_load(f)

    def assign_team_gender(self, home_team: str, away_team: str) -> str:
        """Assign the correct genders to teams playing"""
        home_team = home_team[:-2]
        away_team = away_team[:-2]

        try: match_gender = self.sportsreference_matchups.loc[(self.sportsreference_matchups.home_team == home_team) & (self.sportsreference_matchups.away_team == away_team), "gender"].values[0]
        except Exception as e: 
            logging.error(f"Match {home_team} vs {away_team} not found on Sports Reference. Swapping home and away team and trying again...\nError code: {e}")
            try:
                match_gender = self.sportsreference_matchups.loc[(self.sportsreference_matchups.home_team == away_team) & (self.sportsreference_matchups.away_team == home_team), "gender"].values[0]
            except Exception as e2:
                logging.error(f"Match {home_team} vs {away_team} not found on Sports Reference. Defaulting to gender M. Error code: {e2}")
                match_gender = "m"

        if match_gender not in ["m", "f"]: logging.warning(f"Gender not found for {home_team} vs {away_team}. Found gender: {match_gender}")

        home_team = f"{home_team}_{match_gender.upper()}"
        away_team = f"{away_team}_{match_gender.upper()}"

        return home_team, away_team

    def load_data(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Load the latest odds data"""
        # Fetch stored data
        logging.info("Loading team scores, trends, and name matching...")
        team_scores = pd.read_parquet(self.config["team_scores_path"]).iloc[-1].to_dict()
        team_trends = pd.read_parquet(self.config["team_trends_path"]).iloc[-1].to_dict()
        self.team_name_matching = pd.read_csv(self.config["name_matching_path"])[["original_name", "best_match"]].set_index("original_name")["best_match"].to_dict()

        # Prepare prediction dataframe
        logging.info("Preparing prediction dataframe...")
        prediction_df = odds_df[["commence_time", "home_team", "away_team"]].copy()
        prediction_df["date"] = pd.to_datetime(prediction_df["commence_time"]).dt.tz_localize(None)
        prediction_df.drop("commence_time", axis = 1, inplace = True)

        # Store invalid teams
        invalid_teams_index = []
        self.invalid_team_names = []

        for i in prediction_df.index:
            if prediction_df.loc[i, "home_team"] not in self.team_name_matching or prediction_df.loc[i, "away_team"] not in self.team_name_matching:
                if prediction_df.loc[i, "home_team"] not in self.team_name_matching:
                    logging.error(f"Team name match not found for {prediction_df.loc[i, 'home_team']}, team will be dropped")
                    self.invalid_team_names.append(prediction_df.loc[i, "home_team"])

                if prediction_df.loc[i, "away_team"] not in self.team_name_matching:
                    logging.error(f"Team name match not found for {prediction_df.loc[i, 'away_team']}, team will be dropped")
                    self.invalid_team_names.append(prediction_df.loc[i, "away_team"])

                invalid_teams_index.append(i)
            else:
                home_team, away_team = self.assign_team_gender(self.team_name_matching[prediction_df.loc[i, "home_team"]], self.team_name_matching[prediction_df.loc[i, "away_team"]])

                prediction_df.at[i, "home_team"] = home_team
                prediction_df.at[i, "away_team"] = away_team
                prediction_df.at[i, "home_team_score_z"] = team_scores[home_team]
                prediction_df.at[i, "away_team_score_z"] = team_scores[away_team]
                prediction_df.at[i, "home_team_trend_z"] = team_trends[home_team]
                prediction_df.at[i, "away_team_trend_z"] = team_trends[away_team]
                prediction_df.at[i, "home_team_score"] = np.nan
                prediction_df.at[i, "away_team_score"] = np.nan
                prediction_df.at[i, "home_team_ranking"] = np.nan
                prediction_df.at[i, "away_team_ranking"] = np.nan
                prediction_df.at[i, "gender"] = "m"
        
        prediction_df = prediction_df[["date", "home_team", "home_team_ranking", "home_team_score", "away_team", "away_team_ranking", "away_team_score", "gender", "home_team_score_z", "away_team_score_z", "home_team_trend_z", "away_team_trend_z"]] # Rearrange columns to match historical data
        
        # Drop invalid teams
        prediction_df.drop(invalid_teams_index, inplace = True)
        prediction_df.reset_index(drop = True, inplace = True)

        # Load processed data
        processed_data_df = pd.read_parquet(self.config["processed_data_path"])
        latest_game_date = processed_data_df["date"].max()
        processed_data_df = processed_data_df.loc[processed_data_df["date"] > latest_game_date - timedelta(days = 2*365)]
        
        # Concatinate games to predict for feature generation
        #prediction_df = pd.concat([processed_data_df, prediction_df], ignore_index = True)
        
        self.prediction_data = prediction_df
        self.historical_data = processed_data_df

    def generate_features(self):
        """Generate features for upcoming matches"""
        logging.info("Generating features...")
        self.prediction_data_with_features = self.feature_engineer.process_dataframe(self.prediction_data, self.historical_data)[-self.games_to_predict:].reset_index(drop=True)

    
    def load_model(self):
        """Load the latest trained model"""
        model_dir = os.path.dirname(os.path.dirname(self.base_directory)) + "/Prediction/model_artifacts"
        model_files = [f for f in os.listdir(model_dir) 
                      if f.startswith("production_model_")]
        
        if not model_files:
            raise FileNotFoundError("No trained model found")
            
        latest_model = sorted(model_files)[-1]
        self.model = xgb.Booster()
        self.model.load_model(os.path.join(model_dir, latest_model))
        logging.info(f"Loaded model: {latest_model}")
    
    
    def swap_team_matching_dict(self):
        """Swap the keys and values of a dictionary"""
        return {v: k for k, v in self.team_name_matching.items()}

    def predict(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for upcoming matches"""
        logging.info("Starting prediction process...")
        
        # Prepare feature matrix
        feature_cols = [col for col in self.prediction_data_with_features.columns if col not in ["date", "home_team", "away_team", "home_team_score", "away_team_score", "home_team_won"]]
        X = self.prediction_data_with_features[feature_cols].copy()
        X["gender"] = (X["gender"] == "m").astype(int)
        
        # Make predictions
        dmatrix = xgb.DMatrix(X)
        predictions = self.model.predict(dmatrix)
        
        # Combine predictions with match info
        results = odds_df.copy()
        results = results.loc[~results.home_team.isin(self.invalid_team_names)].reset_index(drop = True)

        # Swap team matching dictionary to "go back" to original team names
        opposite_team_name_matching = self.swap_team_matching_dict()

        # Initialize predictions dataframe to join back to odds data
        predictions_df = pd.DataFrame(predictions, columns = ["predicted_home_win_probability"])
        predictions_df["home_team"] = self.prediction_data_with_features["home_team"].map(opposite_team_name_matching)
        
        # Join predictions to odds data
        results = results.join(predictions_df.set_index("home_team", drop = True), on = "home_team", how = "inner")
        
        # Add away win probability
        results["predicted_away_win_probability"] = 1 - results["predicted_home_win_probability"]
        
        # Add outcome prediction
        results["predicted_outcome"] = results["predicted_home_win_probability"].round(0).astype(int)
        
        # Add prediction date
        results["prediction_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add common odds conversion
        results["predicted_home_odds"] = 1 / results["predicted_home_win_probability"]
        results["predicted_away_odds"] = 1 / results["predicted_away_win_probability"]
        
        logging.info(f"Generated {len(results)} predictions")   

        return results

class BetEvaluator:
    def __init__(self, predictions: pd.DataFrame):
        self.predictions = predictions.copy()

    def kelly_criterion(self, odds: float, probability: float, kelly_multiplier: float = float(1/40), edge: float = 0.005) -> float:
        b = odds - 1
        q = 1 - probability
        kelly_fraction = kelly_multiplier * ((b * probability - q) / b)
        
        # Only place the bet if the expected value exceeds the edge
        if kelly_fraction > edge:
            return kelly_fraction
        else:
            return 0  # Do not place the bet
        
    def filter_odds(self):
        """Filter out the best odds for each match"""
        logging.info("Finding best odds for each match...")
        for i in self.predictions.index:
            home_odds_row = self.predictions.loc[i, [col for col in self.predictions.columns if (("odds" in col) and ("home" in col) and not ("predicted" in col))]]
            away_odds_row = self.predictions.loc[i, [col for col in self.predictions.columns if (("odds" in col) and ("away" in col) and not ("predicted" in col))]]
            
            self.predictions.at[i, "best_home_odds"] = home_odds_row.max()
            self.predictions.at[i, "best_away_odds"] = away_odds_row.max()
        
            self.predictions.at[i, "best_home_bookmaker"] = np.nan#"".join([s for s in home_odds_row.idxmax(skipna = True).split("_")[:-3]])
            self.predictions.at[i, "best_away_bookmaker"] = np.nan#"".join([s for s in away_odds_row.idxmax(skipna = True).split("_")[:-3]])

    def evaluate_bets(self) -> pd.DataFrame:
        """Evaluate the bets based on the predictions"""
        logging.info("Evaluating bets...")
        
        # Filter odds
        self.filter_odds()

        # Filter out bets with positive expected value
        bets = self.predictions.copy().reset_index(drop = True)
        for i in bets.index:
            bets.at[i, "home_kelly"] = self.kelly_criterion(bets.at[i, "best_home_odds"], bets.at[i, "predicted_home_win_probability"])
            bets.at[i, "away_kelly"] = self.kelly_criterion(bets.at[i, "best_away_odds"], bets.at[i, "predicted_away_win_probability"])
        
        # Filter out bets with positive expected value
        bets = bets[(bets["home_kelly"] > 0) | (bets["away_kelly"] > 0)]
        
        logging.info(f"Found {len(bets)} potential bets")
        
        return bets

def run_script(script_path):
    result = subprocess.run(["python", script_path], check=True)
    if result.returncode != 0:
        logging.info(f"Error running {script_path.split('/')[-1]}")
        return
    return result

def main():
    # Clear the terminal screen
    os.system("cls" if platform.system() == "Windows" else "clear")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Get script directory and move up one level to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    script_paths = {"fetch_odds":os.path.join(project_dir, "The Odds API/the_odds_api_odds_data.py")}
    
    # Fetch latest odds
    logging.info("Fetching latest odds data...")
    run_script(script_paths["fetch_odds"])
    odds_df = pd.read_parquet(os.path.join(project_dir, "The Odds API/data/most_recent_odds_data.parquet"))

    # Initialize predictor
    predictor = UpcomingMatchPredictor(odds_df)

    # Generate predictions
    predictions = predictor.predict(odds_df)

    # Save predictions
    predictions.to_parquet(os.path.join(script_dir, f"Historical Predictions/upcoming_match_predictions_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.parquet"))
    
    evaluator = BetEvaluator(predictions)
    bets = evaluator.evaluate_bets()

    bets.to_parquet(os.path.join(script_dir, f"Historical Bet Opportunities/bet_opportunities_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.parquet"))
    bets.to_parquet(os.path.join(script_dir, "most_recent_bet_opportunities.parquet"))
    
if __name__ == "__main__":
    main()
    