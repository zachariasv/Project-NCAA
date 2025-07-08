import pandas as pd
import logging
from typing import Dict, List, Tuple
import os
import platform
import json

class DatabaseMatcher:
    def __init__(self, file_path: str):
        """Initialize the database matcher with the CSV file."""
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.rejected_entries = []
        self.setup_logging()

    def setup_logging(self):
        """Configure logging with proper formatting."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def find_duplicates(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Find entries where either:
        1. One original_name maps to multiple best_matches
        2. One best_match is mapped to by multiple original_names
        """
        # Group by original_name and find those with multiple best_matches
        original_duplicates = {}
        for name, group in self.df.groupby("original_name"):
            if len(group["best_match"].unique()) > 1:
                original_duplicates[name] = group["best_match"].unique().tolist()

        # Group by best_match and find those with multiple original_names
        match_duplicates = {}
        for match, group in self.df.groupby("best_match"):
            if len(group["original_name"].unique()) > 1:
                match_duplicates[match] = group["original_name"].unique().tolist()

        return original_duplicates, match_duplicates

    def get_duplicate_indices(self, original_duplicates: Dict[str, List[str]], match_duplicates: Dict[str, List[str]]) -> List[int]:
        """Get indices of all rows involved in duplicates."""
        duplicate_indices = []
        
        # Get rows where original_name has multiple matches
        for name in original_duplicates:
            indices = self.df[self.df["original_name"] == name].index
            duplicate_indices.extend(indices)
            
        # Get rows where best_match has multiple original names
        for match in match_duplicates:
            indices = self.df[self.df["best_match"] == match].index
            duplicate_indices.extend(indices)
            
        return sorted(list(set(duplicate_indices)))  # Remove duplicates and sort

    def review_matches(self, review_mode: str = "all", original_duplicates: Dict = None, match_duplicates: Dict = None):
        """Interactive interface for reviewing matches one by one."""
        os.system("cls" if platform.system() == "Windows" else "clear")
        logging.info("Starting match review process...")
        
        # Determine which indices to review
        if review_mode == "duplicates":
            indices_to_review = self.get_duplicate_indices(original_duplicates, match_duplicates)
            if not indices_to_review:
                logging.info("No duplicates to review.")
                return
            logging.info(f"Found {len(indices_to_review)} entries to review.")
        else:
            indices_to_review = self.df.index
        review_counter = 1
        for idx in indices_to_review:
            row = self.df.loc[idx]
            while True:
                os.system("cls" if platform.system() == "Windows" else "clear")
                print(f"\nReviewing entry [{review_counter}/{len(indices_to_review)}]\n")
                print("\nMatch Review:")
                print("=" * 50)
                print(f"Original Name: {row['original_name']}")
                print(f"Expected Name: {row['expected_name']}")
                print(f"Best Match:    {row['best_match']}")
                print(f"Match Score:   {row['match_score']}")
                if review_mode == "duplicates":
                    print("\nThis entry is part of a duplicate set.")
                print("=" * 50)
                print("\nOptions:")
                print("Press Enter - Accept match")
                print("n - Reject match")
                print("s - Skip")
                print("q - Quit review")
                
                choice = input("\nYour choice: ").lower().strip()
                
                review_counter += 1
                
                if choice == "":
                    logging.info(f"Accepted: {row['original_name']} -> {row['best_match']}")
                    break
                elif choice == "n":
                    self.rejected_entries.append(idx)
                    logging.info(f"Rejected: {row['original_name']} -> {row['best_match']}")
                    break
                elif choice == "s":
                    logging.info(f"Skipped: {row['original_name']} -> {row['best_match']}")
                    break
                elif choice == "q":
                    return
                else:
                    print("Invalid choice. Please try again.")

    def save_rejected_to_unmatched(self):
        """Format rejected entries and save them to unmatched_teams_with_suggestions.json"""
        script_directory = os.path.dirname(os.path.abspath(__file__))
        unmatched_file = os.path.join(script_directory, "unmatched_teams_with_suggestions.json")
        
        # Read existing unmatched teams if file exists
        if os.path.exists(unmatched_file):
            with open(unmatched_file, "r") as f:
                unmatched_teams = json.load(f)
        else:
            unmatched_teams = []
        
        # Format rejected entries
        for idx in self.rejected_entries:
            row = self.df.loc[idx]
            entry = {
                "original_name": row["original_name"],
                "expected_name": row["expected_name"],
                "top_matches": [
                    [row["best_match"], int(row["match_score"])]
                ]
            }
            unmatched_teams.append(entry)
        
        # Save updated unmatched teams file
        with open(unmatched_file, "w") as f:
            json.dump(unmatched_teams, f, indent=4)
        
        logging.info(f"Added {len(self.rejected_entries)} rejected entries to {unmatched_file}")

    def confirm_and_remove_rejected(self):
        """Show summary of rejected entries and confirm deletion."""
        if not self.rejected_entries:
            logging.info("No entries were rejected.")
            return

        os.system("cls" if platform.system() == "Windows" else "clear")
        print("\nRejected Entries Summary:")
        print("=" * 50)
        for idx in self.rejected_entries:
            row = self.df.loc[idx]
            print(f"Original: {row['original_name']}")
            print(f"Best Match: {row['best_match']}")
            print("-" * 30)

        while True:
            choice = input(f"\nDelete these {len(self.rejected_entries)} entries and add them to unmatched teams? (y/n): ").lower().strip()
            if choice == "y":
                # Save rejected entries to unmatched teams file
                self.save_rejected_to_unmatched()
                
                # Remove rejected entries and save
                self.df = self.df.drop(self.rejected_entries)
                self.df.to_csv(self.file_path, index=False)
                logging.info(f"Removed {len(self.rejected_entries)} entries and saved changes.")
                break
            elif choice == "n":
                logging.info("No changes were made to the database.")
                break
            else:
                print("Please type 'y' for yes or 'n' for no.")

def main():
    # Clear the terminal screen
    os.system("cls" if platform.system() == "Windows" else "clear")
    
    # Initialize matcher with the CSV file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    matcher = DatabaseMatcher(os.path.join(script_directory, "matched_teams_master.csv"))
    
    # Find duplicates
    original_duplicates, match_duplicates = matcher.find_duplicates()
    
    # Display duplicate findings
    if original_duplicates or match_duplicates:
        print("\nDuplicate Matches Found:")
        print("=" * 50)
        
        if original_duplicates:
            print("\nOriginal names with multiple matches:")
            for original, matches in original_duplicates.items():
                print(f"\n{original}:")
                for match in matches:
                    print(f"  → {match}")
        
        if match_duplicates:
            print("\nBest matches with multiple original names:")
            for match, originals in match_duplicates.items():
                print(f"\n{match}:")
                for original in originals:
                    print(f"  → {original}")
    else:
        print("\nNo duplicates found in the database.")
    
    # Ask if user wants to review matches
    while True:
        print("\nReview options:")
        print("1 - Review all matches")
        print("2 - Review only duplicates")
        print("3 - Skip review")
        choice = input("\nYour choice (1/2/3): ").strip()
        
        if choice in ["1", "2", "3"]:
            break
        print("Please enter 1, 2, or 3.")
    
    if choice == "1":
        matcher.review_matches(review_mode="all")
        matcher.confirm_and_remove_rejected()
    elif choice == "2":
        matcher.review_matches(review_mode="duplicates", 
                             original_duplicates=original_duplicates,
                             match_duplicates=match_duplicates)
        matcher.confirm_and_remove_rejected()

if __name__ == "__main__":
    main()