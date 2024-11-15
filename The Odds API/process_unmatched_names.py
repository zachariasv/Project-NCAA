import pandas as pd
import re
from fuzzywuzzy import fuzz, process
import json
import os
import platform
import yaml

# Function to clear the terminal screen
def clear_terminal():
    os.system("cls" if platform.system() == "Windows" else "clear")

# Get the directory of the current script
base_directory = os.path.dirname(__file__)

# Load configuration from YAML file
with open(base_directory +"/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load or create the master file for matched teams
master_file = "matched_teams_master.csv"
if os.path.exists(master_file):
    master_df = pd.read_csv(base_directory + "/" + master_file)
else:
    master_df = pd.DataFrame(columns=["original_name", "expected_name", "best_match", "match_score"])

# Load unmatched teams with suggestions, if available, or load unique teams from the source file
unmatched_file = "unmatched_teams_with_suggestions.json"
if os.path.exists(base_directory + "/" + unmatched_file):
    with open(base_directory + "/" + unmatched_file, "r") as f:
        unmatched_teams = json.load(f)
else:
    with open(base_directory + "/" + "ncaa_basketball_teams.json", "r") as f:
        unmatched_teams = json.load(f)

# Prediction dataset team names in the "TEAMNAME_M" or "TEAMNAME_F" format
prediction_teams = pd.read_parquet(config["prediction_dataset_path"]).columns

# Function to normalize team names by removing punctuation and converting to lowercase
def normalize_name(name):
    return re.sub(r"[^a-zA-Z0-9 ]", "", name).lower().replace(" ", "_")

# Function to create name variations by removing the last one or two words
def create_name_variations(name):
    words = name.split()
    variations = [name]  # Original name
    if len(words) > 1:
        variations.append(" ".join(words[:-1]))  # Drop last word
    return variations

# Function to save unmatched teams
def save_unmatched_teams():
    with open(base_directory + "/" + unmatched_file, "w") as f:
        json.dump(unmatched_teams, f, indent=4)

# Function to save matched teams to the master file
def save_master_df():
    master_df.to_csv(base_directory + "/" + master_file, index=False)

# Processing unmatched teams
try:
    print("Processing unmatched teams...")
    for team_entry in unmatched_teams.copy():  # Copy to modify while iterating
        clear_terminal()
        
        original_name = team_entry["original_name"]
        expected_name = team_entry["expected_name"]
        
        # Check if the team has already been matched to avoid re-processing
        if master_df["original_name"].str.contains(original_name).any():
            continue

        # Generate all variations and collect top matches
        normalized_variations = [normalize_name(var) for var in create_name_variations(original_name)]
        gender_suffix = "_M" if team_entry.get("gender") == "men" else "_F"
        
        all_matches = []
        for name_variation in normalized_variations:
            full_team_name = f"{name_variation}{gender_suffix}"
            matches_for_variation = process.extract(full_team_name, prediction_teams, limit=5, scorer=fuzz.token_sort_ratio)
            all_matches.extend(matches_for_variation)

        # Sort all matches across variations by score and select the top 5 unique matches
        all_matches = sorted(all_matches, key=lambda x: x[1], reverse=True)
        unique_matches = []
        seen_matches = set()
        for match in all_matches:
            if match[0] not in seen_matches:
                unique_matches.append(match)
                seen_matches.add(match[0])
            if len(unique_matches) >= 5:
                break

        # Show options to the user
        print(f"Original Team: {original_name} | Expected Format: {expected_name}")
        print("Top 5 Suggested Matches:")
        for i, (match, score) in enumerate(unique_matches, start=1):
            print(f"{i}. {match} (Score: {score})")

        print("0. None match - Enter a custom search string")
        choice = input("Select the correct match (1-5) or enter 0 to search manually: ").strip().lower()
        
        if choice.lower() == "exit":
            print("Exiting program. Saving progress...")
            save_unmatched_teams()
            save_master_df()
            exit()

        # Handle manual search if selected
        if choice not in ["1", "2", "3", "4", "5", "exit"]:
            custom_search = input("Enter a custom search string: ").strip()
            search_matches = process.extract(custom_search, prediction_teams, limit=10, scorer=fuzz.token_sort_ratio)
            
            # Display manual search results
            print("\nCustom Search Results:")
            for i, (match, score) in enumerate(search_matches, start=1):
                print(f"{i}. {match} (Score: {score})")
            
            choice = input("Select the correct match (1-10), or press Enter if none match: ").strip()
            try:
                choice = int(choice)
                if 1 <= choice <= 10:
                    selected_match = search_matches[choice - 1][0]
                    selected_score = search_matches[choice - 1][1]
                    
                    # Add to master file
                    new_row = pd.DataFrame([{
                        "original_name": original_name,
                        "expected_name": expected_name,
                        "best_match": selected_match,
                        "match_score": selected_score
                    }])
                    master_df = pd.concat([master_df, new_row], ignore_index=True)
                    unmatched_teams.remove(team_entry)
                    print(f"Match '{selected_match}' selected and saved.")
            except ValueError:
                print("No match selected. Keeping team in unmatched.")
        
        elif choice in ["1", "2", "3", "4", "5"]:
            choice = int(choice)
            selected_match = unique_matches[choice - 1][0]
            selected_score = unique_matches[choice - 1][1]
            
            # Add to master file
            new_row = pd.DataFrame([{
                "original_name": original_name,
                "expected_name": expected_name,
                "best_match": selected_match,
                "match_score": selected_score
            }])
            master_df = pd.concat([master_df, new_row], ignore_index=True)
            unmatched_teams.remove(team_entry)
            print(f"Match '{selected_match}' selected and saved.")
        else:
            print("No match selected. Keeping team in unmatched.")

        # Save progress after each unmatched team is processed
        save_unmatched_teams()
        save_master_df()

except KeyboardInterrupt:
    print("\nProcess interrupted. Saving progress...")

# Final save
save_unmatched_teams()
save_master_df()

print("\nAll progress saved. Process completed or interrupted.")
print(f"Matched teams are saved in '{master_file}' and remaining unmatched teams in '{unmatched_file}'.")
