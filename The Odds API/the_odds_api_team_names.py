import requests
import os
from dotenv import load_dotenv
import json
import time
import logging
from datetime import datetime
import pandas as pd
from fuzzywuzzy import fuzz, process
import re
import platform

class TeamNameMatcher:
    def __init__(self):
        """Initialize the TeamNameMatcher with configuration and setup."""
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        
        # Configuration
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport_keys = {
            "men": "basketball_ncaab",
        }
        
        # Set base directory
        self.base_directory = os.path.dirname(__file__) + "/"
        
        # Create necessary directories
        os.makedirs(self.base_directory + "json_requests", exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def fetch_events(self, sport_key: str, gender: str) -> tuple:
        """Fetch events from The Odds API for a specific sport and gender."""
        url = f"{self.base_url}/sports/{sport_key}/events"
        params = {"apiKey": self.api_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Save response to JSON file
            filename = f"team_name_data_{datetime.now().strftime('%Y%m%d%H%M%S')}_{sport_key}.json"
            with open(self.base_directory + "json_requests/" + filename, "w") as json_file:
                json.dump(data, json_file, indent=4)
            
            return data, gender
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching events for {sport_key}: {str(e)}")
            return [], gender

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize team names by removing punctuation and converting to lowercase."""
        return re.sub(r"[^a-zA-Z0-9 ]", "", name).lower().replace(" ", "_")

    @staticmethod
    def create_name_variations(name: str) -> list:
        """Create variations of team names by removing words from the end."""
        words = name.split()
        variations = [name]  # Original name
        if len(words) > 1:
            variations.append(" ".join(words[:-1]))  # Drop last word
        return variations

    def fetch_and_process_teams(self) -> list:
        """Fetch and process team data from all configured sports."""
        all_teams = []
        
        for gender, sport_key in self.sport_keys.items():
            logging.info(f"Fetching events for {gender}'s NCAA basketball")
            events, gender = self.fetch_events(sport_key, gender)
            
            for event in events:
                home_team = event.get("home_team")
                away_team = event.get("away_team")
                if home_team:
                    all_teams.append({"team": home_team, "gender": gender})
                if away_team:
                    all_teams.append({"team": away_team, "gender": gender})
            
            time.sleep(1)  # Respect API rate limits
        
        return all_teams

    def load_existing_teams(self) -> list:
        """Load existing teams from JSON file if it exists."""
        teams_file = self.base_directory + 'ncaa_basketball_teams.json'
        if os.path.exists(teams_file):
            with open(teams_file, 'r') as f:
                return json.load(f)
        return []

    def match_teams(self, unique_teams: list, prediction_teams: list) -> tuple:
        """Match teams against prediction dataset teams."""
        master_file = self.base_directory + "matched_teams_master.csv"
        
        # Load existing matched teams
        if os.path.exists(master_file):
            master_df = pd.read_csv(master_file)
            already_matched_teams = set(master_df["original_name"].values)
        else:
            master_df = pd.DataFrame(columns=["original_name", "expected_name", "best_match", "match_score"])
            already_matched_teams = set()
        
        matched_teams = []
        unmatched_teams = []
        
        for team_entry in unique_teams:
            original_name = team_entry['team']
            
            # Skip if already matched
            if original_name in already_matched_teams:
                continue
            
            normalized_variations = [self.normalize_name(var) for var in self.create_name_variations(original_name)]
            gender_suffix = "_M" if team_entry['gender'] == 'men' else "_F"
            
            # Collect matches across variations
            all_matches = []
            for name_variation in normalized_variations:
                full_team_name = f"{name_variation}{gender_suffix}"
                matches = process.extract(full_team_name, prediction_teams, limit=5, scorer=fuzz.token_sort_ratio)
                all_matches.extend(matches)
            
            # Sort and deduplicate matches
            unique_matches = sorted(set(all_matches), key=lambda x: x[1], reverse=True)[:5]
            
            if unique_matches and unique_matches[0][1] >= 100:
                matched_teams.append({
                    "original_name": original_name,
                    "expected_name": f"{self.normalize_name(original_name)}{gender_suffix}",
                    "best_match": unique_matches[0][0],
                    "match_score": unique_matches[0][1]
                })
            else:
                unmatched_teams.append({
                    "original_name": original_name,
                    "expected_name": f"{self.normalize_name(original_name)}{gender_suffix}",
                    "top_matches": unique_matches
                })
        
        return matched_teams, unmatched_teams, master_df

    def save_results(self, matched_teams: list, unmatched_teams: list, master_df: pd.DataFrame) -> None:
        """Save matched and unmatched teams to their respective files."""
        # Save matched teams
        new_matches_df = pd.DataFrame(matched_teams)
        master_df = pd.concat([master_df, new_matches_df], ignore_index=True).drop_duplicates(subset="original_name")
        master_df.to_csv(self.base_directory + "matched_teams_master.csv", index=False)
        
        # Save unmatched teams
        with open(self.base_directory + "unmatched_teams_with_suggestions.json", "w") as f:
            json.dump(unmatched_teams, f, indent=4)
        
        # Calculate and log success rate
        newly_matched_count = len(matched_teams)
        unmatched_count = len(unmatched_teams)
        
        if newly_matched_count + unmatched_count > 0:
            match_success_rate = (newly_matched_count / (newly_matched_count + unmatched_count)) * 100
        else:
            match_success_rate = 100
        
        logging.info(f"\nMatch Success Rate: {match_success_rate:.2f}%")
        logging.info(f"{newly_matched_count} matched teams saved to 'matched_teams_master.csv'")
        logging.info(f"{unmatched_count} unmatched teams with suggestions saved to 'unmatched_teams_with_suggestions.json'")

def main():
    # Clear the terminal screen
    os.system("cls" if platform.system() == "Windows" else "clear")
    
    matcher = TeamNameMatcher()
    
    # Fetch new teams
    new_teams = matcher.fetch_and_process_teams()
    
    # Load and combine with existing teams
    existing_teams = matcher.load_existing_teams()
    all_teams = new_teams + existing_teams
    
    # Remove duplicates
    unique_teams = [dict(t) for t in set(tuple(team.items()) for team in all_teams)]
    
    # Save updated team list
    with open(matcher.base_directory + 'ncaa_basketball_teams.json', 'w') as f:
        json.dump(unique_teams, f, indent=4)
    
    # Load prediction teams
    prediction_teams = pd.read_parquet(os.getenv("PREDICTION_TEAMS_PATH")).columns
    
    # Match teams
    matched_teams, unmatched_teams, master_df = matcher.match_teams(unique_teams, prediction_teams)
    
    # Save results
    matcher.save_results(matched_teams, unmatched_teams, master_df)

if __name__ == "__main__":
    main()