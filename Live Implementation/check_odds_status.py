import os
import logging
import platform
import pandas as pd
import subprocess

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
    if odds_df.empty:
        logging.info("No odds data found, exiting...")
        return
    
    odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"])
    odds_df.sort_values("commence_time", inplace=True)

    logging.info(f"==================== Found {len(odds_df)} odds for today ====================")
    for index, row in odds_df.iterrows():
        logging.info(f"{row['commence_time'].date()} - {row['commence_time'].time()} - {row['home_team']} vs {row['away_team']}")

if __name__ == "__main__":
    main()