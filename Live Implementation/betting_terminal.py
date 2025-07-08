import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from pathlib import Path

# Constants and global variables
SETTINGS_FILE = "settings.yaml"
COMPLETED_BETS_FOLDER = "Completed_Bets"
console = Console()

# Load or initialize settings
def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        default_settings = {
            "bankroll": 10_000,
            "rounding_interval": 0.005,
            "minimum_kelly_threshold": 0.015,
            "kelly_fraction": 1 / 40,
            "data_file_path": Prompt.ask("Enter the path to the bets file")
        }
        with open(SETTINGS_FILE, "w") as f:
            yaml.dump(default_settings, f)
    with open(SETTINGS_FILE, "r") as f:
        return yaml.safe_load(f)

# Save updated settings
def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        yaml.dump(settings, f)

# Load bets data
def load_bets():
    settings = load_settings()
    file_path = settings["data_file_path"]
    return pd.read_parquet(file_path).reset_index(drop=True)

# Save completed bets
def save_bets(data, date):
    os.makedirs(COMPLETED_BETS_FOLDER, exist_ok=True)
    file_name = f"{date}_bets.csv"
    file_path = os.path.join(COMPLETED_BETS_FOLDER, file_name)
    data.to_csv(file_path, index=False)
    console.print(f"[green]Bets saved to {file_path}[/green]")

# Main menu
def main_menu():
    while True:
        console.clear()
        console.print("[bold cyan]Main Menu[/bold cyan]")
        console.print("1. Place Bets")
        console.print("2. View All Bets")
        console.print("3. Settings")
        console.print("4. Exit")

        choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"])

        if choice == "1":
            place_bets()
        elif choice == "2":
            view_all_bets()
        elif choice == "3":
            settings_menu()
        elif choice == "4":
            exit_app()

# Place bets functionality
def place_bets():
    settings = load_settings()
    bets = load_bets()

    # Sort and filter bets
    bets = bets.sort_values(by=["home_kelly", "away_kelly"], ascending=False)

    bets["home_kelly"] = (bets["home_kelly"] / settings["rounding_interval"]).round() * settings["rounding_interval"]
    bets["away_kelly"] = (bets["away_kelly"] / settings["rounding_interval"]).round() * settings["rounding_interval"]

    bets = bets[(bets["home_kelly"] >= settings["minimum_kelly_threshold"]) | 
                (bets["away_kelly"] >= settings["minimum_kelly_threshold"])]

    placed_bets = []
    skipped_bets = []

    total_bets = len(bets)
    bets = bets.reset_index(drop=True)

    for idx, bet in bets.iterrows():
        console.clear()
        console.print(f"[bold cyan]Placing Bets[/bold cyan]")
        console.print(f"[bold]Bet {idx + 1}/{total_bets}[/bold]")

        # Display bet details with alignment
        console.print(f"[bold]Date:[/bold]: {pd.to_datetime(bet['commence_time']):%Y-%m-%d}", justify="left")
        console.print(f"[bold]Time:[/bold]: {pd.to_datetime(bet['commence_time']):%H:%M}", justify="left")
        console.print()
        
        table = Table(show_header=False, box=None)
        table.add_column("Home", justify="right", style="cyan")
        table.add_column("", justify="center")
        table.add_column("Away", justify="left", style="magenta")

        table.add_row(
            bet["home_team"], "vs", bet["away_team"]
        )
        table.add_row(
            f"{bet['predicted_home_win_probability']:.2%}", "Prediction", f"{bet['predicted_away_win_probability']:.2%}"
        )
        table.add_row(
            f"{1/bet['predicted_home_win_probability']:.2f}", "Fair Odds", f"{1/bet['predicted_away_win_probability']:.2f}"
        )
        table.add_row(
            f"{bet['home_kelly']:.2%}", "Kelly", f"{bet['away_kelly']:.2%}"
        )
        table.add_row(
            f"({bet['best_home_bookmaker']}) {bet['best_home_odds']:.2f}", "Max Odds", f"{bet['best_away_odds']:.2f} ({bet['best_away_bookmaker']})"
        )
        table.add_row(
            f"{max(bet['best_home_odds']*0.75, 1):.2f}", "Min Playable Odds", f"{max(bet['best_away_odds']*0.75, 1):.2f}"
        )

        console.print(table)
        console.print()

        choice = Prompt.ask("1. Record Bet 2. Skip Bet 3. Exit", choices=["1", "2", "3"])

        if choice == "1":
            bet_data = {
                "team": Prompt.ask("Team bet on (home/away)", choices=["home", "away"]),
                "amount": float(Prompt.ask("Bet amount")),
                "odds": float(Prompt.ask("Final odds")),
                "bookmaker": Prompt.ask("Bookmaker"),
                "time": Prompt.ask("Time of bet (HH:MM)")
            }
            placed_bets.append({**bet, **bet_data})
        elif choice == "2":
            skipped_bets.append(bet)
        elif choice == "3":
            break

    # Create final table
    review_bets = pd.DataFrame(placed_bets + skipped_bets)
    date = datetime.now().strftime("%Y-%m-%d")
    save_bets(review_bets, date)

# View all bets functionality
def view_all_bets():
    bets = load_bets()

    # Filter relevant columns
    bets = bets[[
        "commence_time", "home_team", "away_team", 
        "predicted_outcome", "predicted_home_win_probability",
        "best_home_odds", "best_home_bookmaker", "best_away_odds", "best_away_bookmaker", "home_kelly", "away_kelly"
    ]]

    console.clear()
    table = Table(title="All Bets")

    # Add headers
    table.add_column("Date/Time", justify="center")
    table.add_column("Home Team", justify="right", style="cyan")
    table.add_column("Away Team", justify="left", style="magenta")
    table.add_column("Prediction", justify="center")
    table.add_column("Probability", justify="center")
    table.add_column("Max Home Odds", justify="center")
    table.add_column("Best Home Bookmaker", justify="center")
    table.add_column("Best Away Bookmaker", justify="center")
    table.add_column("Max Away Odds", justify="center")
    table.add_column("Home Kelly", justify="center")
    table.add_column("Away Kelly", justify="center")

    # Add rows
    for _, row in bets.iterrows():
        table.add_row(
            f"{pd.to_datetime(row['commence_time']):%Y-%m-%d | %H:%M}",
            row["home_team"],
            row["away_team"],
            "Home" if row["predicted_outcome"] == 1 else "Away",
            f"{max(row['predicted_home_win_probability'], 1-row['predicted_home_win_probability']):.2%}",
            f"{row['best_home_odds']:.2f}",
            row["best_home_bookmaker"],
            f"{row['best_away_odds']:.2f}",
            row["best_away_bookmaker"],
            f"{row['home_kelly']:.2%}",
            f"{row['away_kelly']:.2%}"
        )

    console.print(table)
    Prompt.ask("Press Enter to return to the main menu")

# Settings menu
def settings_menu():
    settings = load_settings()

    while True:
        console.clear()
        console.print("[bold cyan]Settings[/bold cyan]")
        console.print(f"1. Bankroll: {settings['bankroll']}")
        console.print(f"2. Rounding Interval: {settings['rounding_interval']}")
        console.print(f"3. Minimum Kelly Threshold: {settings['minimum_kelly_threshold']}")
        console.print(f"4. Kelly Fraction: {settings['kelly_fraction']}")
        console.print(f"5. Data File Path: {settings['data_file_path']}")
        console.print("6. Return to Main Menu")

        choice = Prompt.ask("Choose a setting to update or return", choices=["1", "2", "3", "4", "5", "6"])

        if choice == "1":
            settings["bankroll"] = float(Prompt.ask("Enter new bankroll"))
        elif choice == "2":
            settings["rounding_interval"] = float(Prompt.ask("Enter new rounding interval"))
        elif choice == "3":
            settings["minimum_kelly_threshold"] = float(Prompt.ask("Enter new minimum Kelly threshold"))
        elif choice == "4":
            settings["kelly_fraction"] = float(Prompt.ask("Enter new Kelly fraction"))
        elif choice == "5":
            settings["data_file_path"] = Prompt.ask("Enter new data file path")
        elif choice == "6":
            break

        save_settings(settings)

# Exit functionality
def exit_app():
    console.clear()
    console.print("[bold]Exiting the application...[/bold]")
    exit()

if __name__ == "__main__":
    main_menu()
