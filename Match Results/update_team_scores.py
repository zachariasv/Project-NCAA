# Imports
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import logging
import os
import platform
from time import timedelta

def calculate_goal_utility(home_goals, away_goals, alpha=0.025, norm=11):
    """
    Calculates the goal utility based on the number of goals scored by home and away teams.
    """
    home_goal_utility = ((1.0 / (1.0 + math.exp(-alpha * home_goals))) - 0.5) * norm
    away_goal_utility = ((1.0 / (1.0 + math.exp(-alpha * -away_goals))) - 0.5) * norm
    goal_utility = home_goal_utility + away_goal_utility
    return goal_utility

def calculate_team_score(prev_score, prev_trend, latest_goal_utility, alpha=0.97):
    """
    Updates a team's score based on previous score, trend, and latest goal utility.
    """
    team_score = alpha * (prev_score + prev_trend) + (1.0 - alpha) * latest_goal_utility
    return team_score

def calculate_team_trend(prev_score, prev_trend, latest_score, beta=0.99):
    """
    Updates a team's trend based on previous score, trend, and latest score.
    """
    team_trend = beta * prev_trend + (1 - beta) * (latest_score - prev_score)
    return team_trend

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the historical match data.
    """
    df = pd.read_parquet(file_path).drop_duplicates().sort_values("date").reset_index(drop=True)
    df["home_team"] = df["home_team"] + "_" + df["gender"].str.capitalize()
    df["away_team"] = df["away_team"] + "_" + df["gender"].str.capitalize()
    df["date"] = pd.to_datetime(df["date"])
    return df

def calculate_team_scores(df, historical_scores=None, historical_trends=None):
    """
    Calculates team scores and trends over time with progress bars.
    """
    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()

    # Initialize previous scores and trends
    prev_scores = {}
    prev_trends = {}

    if historical_scores is not None and historical_trends is not None:
        # Get the last date
        last_date = historical_scores.index.max()
        # Filter df to only include matches after last_date
        df = df[df["date"] > last_date]
        # If there are no new matches, quit
        if df.empty:
            logging.info("No new matches to process.")
            quit()
        # Get the last known scores and trends
        last_scores = historical_scores.loc[last_date].to_dict()
        last_trends = historical_trends.loc[last_date].to_dict()
        prev_scores.update(last_scores)
        prev_trends.update(last_trends)
    else:
        # Initialize scores and trends to 0
        prev_scores = {team: 0 for team in all_teams}
        prev_trends = {team: 0 for team in all_teams}
        historical_scores = pd.DataFrame()
        historical_trends = pd.DataFrame()

    # Sort the matches by date
    df = df.sort_values("date")

    # Initialize lists to store records
    team_scores_records = []
    team_trends_records = []

    # Initialize progress bar
    total_matches = len(df)
    with tqdm(total=total_matches, desc="Calculating Team Scores", unit="match") as pbar:
        for idx, match in df.iterrows():
            date = match["date"]
            next_day = date + timedelta(days=1)
            home_team = match["home_team"]
            away_team = match["away_team"]
            home_goals = match["home_team_score"]
            away_goals = match["away_team_score"]

            # Ensure teams are in prev_scores and prev_trends
            for team in [home_team, away_team]:
                if team not in prev_scores:
                    prev_scores[team] = 0
                    prev_trends[team] = 0

            # Calculate goal utility
            goal_utility = calculate_goal_utility(home_goals, away_goals)

            # Home team calculations
            latest_home_score = calculate_team_score(prev_scores[home_team], prev_trends[home_team], goal_utility)
            latest_home_trend = calculate_team_trend(prev_scores[home_team], prev_trends[home_team], latest_home_score)

            # Away team calculations (negative goal utility)
            latest_away_score = calculate_team_score(prev_scores[away_team], prev_trends[away_team], -goal_utility)
            latest_away_trend = calculate_team_trend(prev_scores[away_team], prev_trends[away_team], latest_away_score)

            # Update previous scores and trends
            prev_scores[home_team] = latest_home_score
            prev_trends[home_team] = latest_home_trend
            prev_scores[away_team] = latest_away_score
            prev_trends[away_team] = latest_away_trend

            # Store the scores and trends for this date
            team_scores_records.extend([
                {"date": next_day, "team": home_team, "score": latest_home_score},
                {"date": next_day, "team": away_team, "score": latest_away_score},
            ])
            team_trends_records.extend([
                {"date": next_day, "team": home_team, "trend": latest_home_trend},
                {"date": next_day, "team": away_team, "trend": latest_away_trend},
            ])

            # Update progress bar
            pbar.update(1)

    # Convert records to DataFrames
    new_team_scores_df = pd.DataFrame(team_scores_records)
    new_team_trends_df = pd.DataFrame(team_trends_records)

    # Pivot to get team scores and trends over time
    new_team_scores_historical = new_team_scores_df.pivot_table(index="date", columns="team", values="score")
    new_team_trends_historical = new_team_trends_df.pivot_table(index="date", columns="team", values="trend")

    # Combine with existing historical data
    historical_scores = pd.concat([historical_scores, new_team_scores_historical])
    historical_trends = pd.concat([historical_trends, new_team_trends_historical])

    # Sort index and forward-fill missing values
    historical_scores = historical_scores.sort_index().ffill()
    historical_trends = historical_trends.sort_index().ffill()

    return historical_scores, historical_trends

def save_team_scores(historical_scores, historical_trends, TEAM_SCORES_FILE, TEAM_TRENDS_FILE):
    """
    Saves the team scores and trends to Parquet files.
    """
    logging.info("Saving team scores and trends...")
    historical_scores.to_parquet(TEAM_SCORES_FILE)
    historical_trends.to_parquet(TEAM_TRENDS_FILE)

def load_team_scores(TEAM_SCORES_FILE, TEAM_TRENDS_FILE):
    """
    Loads the team scores and trends from Parquet files if they exist.
    """
    logging.info("Loading team scores and trends...")
    if os.path.exists(TEAM_SCORES_FILE) and os.path.exists(TEAM_TRENDS_FILE):
        historical_scores = pd.read_parquet(TEAM_SCORES_FILE)
        historical_trends = pd.read_parquet(TEAM_TRENDS_FILE)
        return historical_scores, historical_trends
    else:
        return None, None

def assign_team_data(df, historical_scores, historical_trends):
    """
    Assigns team scores and trends to the match data and calculates home team advantage.
    """
    # Ensure date columns are datetime
    df["date"] = pd.to_datetime(df["date"])
    historical_scores.index = pd.to_datetime(historical_scores.index)
    historical_trends.index = pd.to_datetime(historical_trends.index)

    # Stack the historical_scores and historical_trends
    team_scores_df_long = historical_scores.stack().reset_index().rename(columns={"level_1": "team", 0: "team_score"})
    team_trends_df_long = historical_trends.stack().reset_index().rename(columns={"level_1": "team", 0: "team_trend"})

    # Merge team scores with the main DataFrame
    df = df.merge(
        team_scores_df_long.rename(columns={"team": "home_team", "team_score": "home_team_score_z"}),
        on=["date", "home_team"], how="left"
    )
    df = df.merge(
        team_scores_df_long.rename(columns={"team": "away_team", "team_score": "away_team_score_z"}),
        on=["date", "away_team"], how="left"
    )

    # Merge team trends with the main DataFrame
    df = df.merge(
        team_trends_df_long.rename(columns={"team": "home_team", "team_trend": "home_team_trend_z"}),
        on=["date", "home_team"], how="left"
    )
    df = df.merge(
        team_trends_df_long.rename(columns={"team": "away_team", "team_trend": "away_team_trend_z"}),
        on=["date", "away_team"], how="left"
    )

    return df

def main():
    # Clear the terminal screen
    os.system("cls" if platform.system() == "Windows" else "clear")

    # Get the directory of the current script
    BASE_DIRECTORY = os.path.dirname(__file__) + "/"

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Configuration
    WEBSCRAPED_MATCHES_FILE = BASE_DIRECTORY + "webscraped_ncaa_games_history.parquet"
    TEAM_SCORES_FILE = BASE_DIRECTORY + "team_scores.parquet"
    TEAM_TRENDS_FILE = BASE_DIRECTORY + "team_trends.parquet"
    PROCESSED_DATA_FILE = BASE_DIRECTORY + "processed_data.parquet"

    logging.info("Starting team scores calculation.")

    # Load and preprocess historical data
    df = load_and_preprocess_data(WEBSCRAPED_MATCHES_FILE)

    # Load existing team scores and trends
    historical_scores, historical_trends = load_team_scores(TEAM_SCORES_FILE, TEAM_TRENDS_FILE)

    # Calculate team scores and trends
    historical_scores, historical_trends = calculate_team_scores(df, historical_scores=historical_scores, historical_trends=historical_trends)

    # Save updated team scores and trends
    save_team_scores(historical_scores, historical_trends, TEAM_SCORES_FILE, TEAM_TRENDS_FILE)

    # Assign team scores and calculate home team advantage
    df = assign_team_data(df, historical_scores, historical_trends)

    # Save processed data
    df.to_parquet(PROCESSED_DATA_FILE)
    logging.info("Team scores calculated and assigned successfully.")

if __name__ == "__main__":
    main()
