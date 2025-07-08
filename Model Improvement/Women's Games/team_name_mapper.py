import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import yaml
import os
import tqdm
import time  # Added for sleep functionality

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Paths (adjust as needed)
SCRAPED_ODDS_PATH = "/Users/zacharias/Dropbox/Python/Project-NCAA/Model Improvement/Women's Games/ncaa_women_odds.csv"
PROCESSED_DATA_PATH = '/Users/zacharias/Dropbox/Python/Project-NCAA/Prediction/processed_data_with_features.parquet'
OUTPUT_MAPPING_CSV = "/Users/zacharias/Dropbox/Python/Project-NCAA/Model Improvement/Women's Games/team_name_mapping.csv"

# Load scraped team names
odds_df = pd.read_csv(SCRAPED_ODDS_PATH)
scraped_names = set(odds_df['home_team'].unique()).union(odds_df['away_team'].unique())

# Load processed data team names
proc_df = pd.read_parquet(PROCESSED_DATA_PATH)
db_names = set([name for name in proc_df['home_team'].unique() if name.endswith('_F')]).union(
    [name for name in proc_df['away_team'].unique() if name.endswith('_F')])

# Load existing mappings if available
mappings = []
existing_mapped_names = set()
if os.path.exists(OUTPUT_MAPPING_CSV):
    print(f"Loading existing mappings from {OUTPUT_MAPPING_CSV}")
    existing_df = pd.read_csv(OUTPUT_MAPPING_CSV)
    mappings = existing_df.to_dict('records')
    existing_mapped_names = set(existing_df['original_name'])
    print(f"Loaded {len(existing_mapped_names)} existing mappings")

# Determine which names still need processing
names_to_process = sorted(scraped_names - existing_mapped_names)
print(f"Found {len(names_to_process)} new team names to process")

print("Team Name Matcher - Suggest top candidate matches from processed DB for each scraped name")
print("Type the number 1-5 to choose, 0 for None, or 'q' to quit and save progress.")
print("Matches with score >= 85% will be auto-mapped")

time.sleep(2)
# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

for original in tqdm.tqdm(names_to_process):
    # Skip if already mapped
    if original in existing_mapped_names:
        print(f"Skipping '{original}' - already mapped")
        continue
        
    # Compute top matches
    choices = process.extract(
        original,
        db_names,
        scorer=fuzz.token_sort_ratio,
        limit=10
    )  # returns list of (match, score, idx)

    # Check if the top choice has a score above 85%
    if choices[0][1] >= 85:
        # Auto-mapping high-confidence matches
        best_match = choices[0][0]
        print(f"\nAuto-mapping '{original}' to '{best_match}' (score: {choices[0][1]:.2f}%)")
        mappings.append({
            'original_name': original,
            'best_match': best_match
        })
    else:
        print(f"\nOriginal Name: '{original}'")
        for i, (match, score, _) in enumerate(choices, start=1):
            print(f"  {i}. {match} ({score:.2f}%)")
        print("  0. None of the above")

        # Prompt user
        while True:
            sel = input(f"Select best match for '{original}': ")
            if sel.lower() == 'q':
                break
            if sel.isdigit() and int(sel) in range(0, len(choices)+1):
                sel = int(sel)
                break
            print("Invalid selection. Enter 1-9, 0, or 'q'.")

        if isinstance(sel, str) and sel.lower() == 'q':
            print("Quitting and saving mapping so far...")
            break

        best_match = None
        if sel > 0:
            best_match = choices[sel-1][0]
        mappings.append({
            'original_name': original,
            'best_match': best_match
        })

    # Save to CSV (append to existing data)
    mapping_df = pd.DataFrame(mappings)
    mapping_df.to_csv(OUTPUT_MAPPING_CSV, index=False)
    
    # Print save message and clear terminal with delay to allow reading
    print(f"Saved {len(mappings)} mappings to {OUTPUT_MAPPING_CSV}")
    time.sleep(0.1)  # Brief pause to see the message
    os.system('cls' if os.name == 'nt' else 'clear')

print("Mapping process completed.")
print(f"Total mappings: {len(mappings)}")
print(f"Saved to: {OUTPUT_MAPPING_CSV}")
