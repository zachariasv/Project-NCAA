import pandas as pd
from thefuzz import fuzz
import numpy as np
import logging
import re
import os
from typing import List, Dict, Any
import yaml

class ManualBacktester:
    """
    A simple class that:
      - Reads your weird CSV text and extracts "match blocks"
      - Iterates through each block, letting you manually match
        it to historical results & predictions
      - Logs the final data into a "manual_backtest.csv" file
    """

    def __init__(self,
                 historical_results: pd.DataFrame,
                 predictions: pd.DataFrame,
                 output_file: str = "manual_backtest_results.csv"):
        """
        Args:
            historical_results: A DataFrame containing final outcomes (to be manually searched)
            predictions: A DataFrame containing your model predictions (to be manually searched)
            output_file: Where we store the final manual backtest info.
        """
        self.historical_results = historical_results
        self.predictions = predictions
        self.output_file = output_file

        # If output_file doesn't exist yet, create it with headers
        if not os.path.exists(self.output_file):
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write("csv_block_index,parsed_date,parsed_time,home_team,home_odds,away_team,away_odds,"
                        "hist_result_index,hist_final_outcome,prediction_index,predicted_outcome\n")

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def parse_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses a single-column DataFrame containing match data and splits it into individual games.

        Each game is identified by the line "Winner (Incl. Overtime)".
        For each such line, the function takes the following 4 rows as:
            - Home Team
            - Home Odds
            - Away Team
            - Away Odds
        It also includes the time from 2 rows prior to the "Winner..." line and the current date.

        Args:
            df (pd.DataFrame): DataFrame with one column containing the raw CSV lines.

        Returns:
            pd.DataFrame: Parsed DataFrame with each row representing a game, containing:
                        ['date', 'time', 'home_team', 'home_odds', 'away_team', 'away_odds']
        """
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        
        # Initialize variables
        current_date = None
        games = []
        lines = df.iloc[:, 0].tolist()
        total_lines = len(lines)
        
        # Define regex patterns
        date_pattern = re.compile(r"^[A-Za-z]+ \d{1,2}, \d{4}$")
        winner_pattern = re.compile(r"^Winner(?: \(Incl\. Overtime\))?$", re.IGNORECASE)
        time_pattern = re.compile(r"^\d{1,2}:\d{2} (AM|PM)$")
        starts_in_pattern = re.compile(r"^Starts in \d+ minutes$", re.IGNORECASE)
        
        for i in range(total_lines):
            line = lines[i].strip()
            
            # Check for date line
            if date_pattern.match(line):
                try:
                    current_date = pd.to_datetime(line).date()
                    logging.info(f"Detected date: {current_date}")
                except ValueError:
                    logging.warning(f"Unable to parse date from line: '{line}'")
                continue
            
            # Check for "Winner (Incl. Overtime)" line
            if winner_pattern.match(line):
                # Ensure there are enough lines before and after
                if i < 2 or i + 4 >= total_lines:
                    logging.warning(f"Insufficient lines around index {i} for a complete match block. Skipping.")
                    continue
                
                # Get time from 2 lines prior
                time_line = lines[i - 2].strip()
                if time_pattern.match(time_line) or starts_in_pattern.match(time_line):
                    parsed_time = time_line
                    logging.info(f"Detected time: {parsed_time}")
                else:
                    parsed_time = None
                    logging.warning(f"Invalid or missing time at index {i - 2}: '{time_line}'.")
                
                # Extract match details from the next 4 lines
                try:
                    home_team = lines[i + 1].strip()
                    home_odds_str = lines[i + 2].strip()
                    away_team = lines[i + 3].strip()
                    away_odds_str = lines[i + 4].strip()
                    
                    # Convert odds to float
                    home_odds = float(home_odds_str)
                    away_odds = float(away_odds_str)
                    
                    logging.info(f"Parsed match - Home: {home_team} (Odds: {home_odds}), Away: {away_team} (Odds: {away_odds})")
                    
                    # Append the game to the list if date and time are available
                    if current_date:
                        games.append({
                            "date": current_date,
                            "time": parsed_time,
                            "home_team": home_team,
                            "home_odds": home_odds,
                            "away_team": away_team,
                            "away_odds": away_odds
                        })
                    else:
                        logging.warning(f"No date available for match at index {i}. Skipping match.")
                        
                except ValueError as ve:
                    logging.error(f"Error converting odds to float at indices {i+2} or {i+4}: {ve}. Skipping match.")
                except Exception as e:
                    logging.error(f"Unexpected error parsing match at index {i}: {e}. Skipping match.")
        
        # Create DataFrame from games list
        parsed_df = pd.DataFrame(games, columns=["date", "time", "home_team", "home_odds", "away_team", "away_odds"])
        logging.info(f"Parsed {len(parsed_df)} games from the CSV.")
        return parsed_df

    def search_match(self, date, home_team: str, away_team: str, df: pd.DataFrame, threshold: int = 70):
        """
        Searches for the best matching game in the DataFrame based on home and away team names using fuzzy matching.

        Args:
            home_team (str): Name of the home team to search for.
            away_team (str): Name of the away team to search for.
            df (pd.DataFrame): DataFrame containing game data with 'home_team' and 'away_team' columns.
            threshold (int, optional): Minimum fuzzy match score to consider a valid match. Defaults to 80.

        Returns:
            pd.Series or None: The row with the highest combined fuzzy match score if above threshold, else None.
        """
        best_score = 0
        best_match = None

        for idx, row in df.iterrows():
            # Calculate fuzzy scores for home and away teams
            home_score = fuzz.token_set_ratio(home_team.lower(), row['home_team'].lower())
            away_score = fuzz.token_set_ratio(away_team.lower(), row['away_team'].lower())
            
            # Combine the scores (you can adjust weighting if needed)
            if date:
                date_score = fuzz.token_set_ratio(date, row['date'])
                combined_score = (home_score + away_score + date_score) / 3
            else:
                combined_score = (home_score + away_score) / 2
            
            logging.debug(f"Checking row {idx}: Home Score={home_score}, Away Score={away_score}, Combined Score={combined_score}")
            
            # Update best match if current combined score is higher
            if combined_score > best_score:
                best_score = combined_score
                best_match = row
        
        if best_score >= threshold:
            logging.info(f"Best match found with score {best_score}: Home Team='{best_match['home_team']}', Away Team='{best_match['away_team']}'")
            return best_match
        else:
            logging.warning(f"No suitable match found. Highest score was {best_score} for {best_match.home_team} vs {best_match.away_team} which is below the threshold of {threshold}.")
            return None

    def run_backtest(self, odds_data: pd.DataFrame):
        for i in odds_data.index:
            date = odds_data.loc[i, "date"]
            time = odds_data.loc[i, "time"]
            home_team = odds_data.loc[i, "home_team"]
            away_team = odds_data.loc[i, "away_team"]

            best_match = self.search_match(date, home_team, away_team, df = self.historical_results)
            if best_match != None:
                home_team_won = int(best_match.home_team_score > best_match.away_team_score)



def main():
    """
    Example usage. In reality, you'd:
      1) Load your historical_results DataFrame
      2) Load your predictions DataFrame
      3) Load or read your weird CSV text
      4) Pass them all to ManualBacktester
    """
    # -- 1) Load historical results (placeholder) --
    # Suppose your historical DF has columns like ['team_home','team_away','final_outcome','date']
    # You handle this however you want
    BASE_DIRECTORY = os.path.dirname(__file__) + "/"
    CONFIG_PATH = BASE_DIRECTORY + "config.yaml"
    
    with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

    historical_results = pd.read_parquet(config["processed_data_path"])
    historical_results = historical_results.loc[historical_results.date >= pd.to_datetime("2024-01-01")]

    # -- 2) Load predictions (placeholder) --
    predictions = pd.DataFrame()
    for file in os.listdir(BASE_DIRECTORY+"Historical Predictions/"):
        predictions = pd.concat([predictions.copy(), pd.read_parquet((BASE_DIRECTORY) + "Historical Predictions/" + file)])

    predictions.drop_duplicates(inplace = True)
    predictions.sort_values("commence_time", inplace = True)
    predictions.reset_index(inplace = True, drop = True)

    # -- 3) Read the weird CSV text (placeholder) --
    odds_data = pd.DataFrame()
    for file in os.listdir(BASE_DIRECTORY+"Stake Data/"):
        try: odds_data = pd.concat([odds_data, pd.read_csv(BASE_DIRECTORY+"Stake Data/"+file, header=None)])
        except Exception as e: print(f"Failed loading {file}. Error code: {e}")

    # 4) Instantiate and run
    backtester = ManualBacktester(historical_results, predictions)
    odds_data = backtester.parse_csv(odds_data)

    backtester.run_backtest(odds_data)

if __name__ == "__main__":
    main()
