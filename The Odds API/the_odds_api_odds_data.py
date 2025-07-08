import numpy as np
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import logging
import platform

def fetch_odds_data(BASE_DIRECTORY):
    logging.info("Fetching odds data...")
    # Load the API key from the .env file
    load_dotenv()
    api_key = os.getenv("API_KEY")

    # Fetch the odds data
    sport_key = "basketball_ncaab"
    regions = "eu"  # Regions: us, us2, uk, eu, au
    markets = "h2h"  # Betting markets
    odds_format = "decimal"  # Odds formats: decimal, american

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        
        # Add timestamp and parameters to the data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for game in data:
            game["timestamp"] = timestamp
        
        # Generate a unique filename using a timestamp
        filename = f"odds_data_{datetime.now().strftime('%Y%m%d%H%M%S')}_{sport_key}_{regions}_{markets}_{odds_format}.json"
        
        # Save the JSON response to a file
        with open(BASE_DIRECTORY + "json_requests/"+filename, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
    else:
        logging.info(f"Error: {response.status_code} - {response.text}")
    return data


def format_odds_data(data):
    logging.info("Formatting odds data...")
    # Prepare list for flattened data entries
    flattened_data = []
    for game in data:
        base_info = {
            "id": game["id"],
            "sport_key": game["sport_key"],
            "sport_title": game["sport_title"],
            "commence_time": game["commence_time"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "timestamp": game["timestamp"]
        }
        
        # Process each bookmaker and its markets
        for bookmaker in game["bookmakers"]:
            bookmaker_key = bookmaker["key"]
            for market in bookmaker["markets"]:
                market_key = market["key"]
                
                entry = base_info.copy()
                for outcome in market["outcomes"]:
                    team = "home" if outcome["name"] == game["home_team"] else "away"
                    # Generalized column name based on market type and team
                    column_name = f"{bookmaker_key}_{team}_{market_key}_odds"
                    entry[column_name] = outcome["price"]
                
                # Add the entry to flattened_data
                flattened_data.append(entry)

    # Convert the flattened data into a DataFrame
    df = pd.DataFrame(flattened_data).drop_duplicates()

    # Collapse rows for each fixture by using `groupby` and `agg`
    df = df.groupby(
        ["id", "sport_key", "sport_title", "commence_time", "home_team", "away_team", "timestamp"],
        as_index=False
    ).agg("first")  # Use "first" to take the first non-null value in each group
    
    return df

def main():
    # Clear the terminal screen
    os.system("cls" if platform.system() == "Windows" else "clear")

    # Set base directory
    BASE_DIRECTORY = os.path.dirname(__file__) + "/"
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Retrieve odds data
    data = fetch_odds_data(BASE_DIRECTORY)
    df = format_odds_data(data)
    logging.info(f"Formatted data with {len(df)} rows.")

    df.to_parquet(BASE_DIRECTORY + "data/most_recent_odds_data.parquet")

if __name__ == "__main__":
    main()