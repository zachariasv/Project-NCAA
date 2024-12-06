import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import os
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import platform

@dataclass
class GameStats:
    date: datetime
    opponent: str
    points_scored: int
    points_conceded: int
    win: int
    is_home: bool
    point_diff: float

class TeamStats:
    def __init__(self):
        self.history: List[GameStats] = []
        self._cached_stats: Dict = {}
        self._last_update_idx = -1

    def add_game(self, game: GameStats) -> None:
        self.history.append(game)
        cutoff_date = game.date - timedelta(days=30*18) # Keep only last 18 months of data
        self.history = [g for g in self.history if g.date > cutoff_date]
        self._cached_stats = {}  # Invalidate cache

    def get_recent_stats(self, before_date: datetime, n_games: int = 5) -> Dict:
        cache_key = (before_date, n_games)
        if cache_key in self._cached_stats:
            return self._cached_stats[cache_key]

        recent_games = [g for g in self.history if g.date < before_date][-n_games:]

        stats = {
            "wins": sum(g.win for g in recent_games),
            "avg_points_scored": np.mean([g.points_scored for g in recent_games]) if recent_games else 0.0,
            "avg_points_conceded": np.mean([g.points_conceded for g in recent_games]) if recent_games else 0.0,
            "total_points_scored": sum(g.points_scored for g in recent_games),
            "total_points_conceded": sum(g.points_conceded for g in recent_games),
            "avg_point_diff": np.mean([g.point_diff for g in recent_games]) if recent_games else 0.0,
            "total_point_diff": sum(g.point_diff for g in recent_games),
        }

        self._cached_stats[cache_key] = stats
        return stats

    def get_venue_stats(self, is_home: bool, before_date: datetime) -> Tuple[float, float]:
        cache_key = f"venue_stats_{is_home}"
        if cache_key in self._cached_stats:
            return self._cached_stats[cache_key]

        venue_games = [g for g in self.history if (g.is_home == is_home and g.date < before_date)]
        if not venue_games:
            return 0.0, 0.0

        win_rate = sum(g.win for g in venue_games) / len(venue_games)
        avg_points = np.mean([g.points_scored for g in venue_games])

        stats = (win_rate, avg_points)
        self._cached_stats[cache_key] = stats
        return stats

    def get_time_since_last_game(self, current_date: datetime) -> int:
        last_game_dates = [g.date for g in self.history if g.date < current_date]
        if not last_game_dates:
            return -1  # Indicates no previous games
        last_game_date = max(last_game_dates)
        return (current_date - last_game_date).days

    def get_away_game_streak(self, current_date: datetime) -> int:
        streak = 0
        for game in reversed([g for g in self.history if g.date > (current_date - timedelta(days=30*6))]): # Only look at current season
            if game.date >= current_date:
                continue
            if not game.is_home:
                streak += 1
            else:
                break
        return streak

    def get_overall_stats(self, before_date: datetime) -> Dict:
        games = [g for g in self.history if (g.date < before_date )]
        if not games:
            return {"avg_points_scored": 0.0, "avg_points_conceded": 0.0}
        avg_points_scored = np.mean([g.points_scored for g in games])
        avg_points_conceded = np.mean([g.points_conceded for g in games])
        return {"avg_points_scored": avg_points_scored, "avg_points_conceded": avg_points_conceded}

class FeatureEngineering:
    def __init__(self):
        self.teams: Dict[str, TeamStats] = {}
        self.head_to_head_history: Dict[Tuple[str, str], List[GameStats]] = {}

    def _update_head_to_head_history(self, home_team: str, away_team: str, home_game: GameStats, away_game: GameStats) -> None:
        """Update head-to-head history between two teams"""
        key = tuple(sorted([home_team, away_team]))
        if key not in self.head_to_head_history:
            self.head_to_head_history[key] = []
        self.head_to_head_history[key].extend([home_game, away_game])

    def get_head_to_head_stats(self, home_team: str, away_team: str, before_date: datetime, n_games: int = 10) -> Dict:
        """Calculate head-to-head statistics between two teams"""
        key = tuple(sorted([home_team, away_team]))

        # Filter games before the current date
        h2h_games = [
            g for g in self.head_to_head_history.get(key, [])
            if g.date < before_date
        ][-n_games:]  # Adjusted for both home and away perspectives

        if not h2h_games:
            return {
                "h2h_total_games": 0,
                "h2h_home_win_percentage": 0.0,
                "h2h_away_win_percentage": 0.0,
                "h2h_avg_home_points": 0.0,
                "h2h_avg_away_points": 0.0,
                "h2h_home_win_rate": 0.0,
                "h2h_avg_points_scored_home_team": 0.0,
                "h2h_avg_points_conceded_home_team": 0.0,
                "h2h_avg_point_diff_home_team": 0.0,
                "h2h_avg_points_scored_away_team": 0.0,
                "h2h_avg_points_conceded_away_team": 0.0,
                "h2h_avg_point_diff_away_team": 0.0,
            }

        # Split games into home_team and away_team perspectives
        home_team_games = [g for g in h2h_games if g.opponent == away_team]
        away_team_games = [g for g in h2h_games if g.opponent == home_team]

        h2h_total_games = len(home_team_games)

        h2h_home_wins = sum(g.win for g in home_team_games)
        h2h_away_wins = sum(g.win for g in away_team_games)

        h2h_home_win_percentage = h2h_home_wins / len(home_team_games) if home_team_games else 0.0
        h2h_away_win_percentage = h2h_away_wins / len(away_team_games) if away_team_games else 0.0

        h2h_avg_home_points = np.mean([g.points_scored for g in home_team_games]) if home_team_games else 0.0
        h2h_avg_away_points = np.mean([g.points_scored for g in away_team_games]) if away_team_games else 0.0

        # New metrics for home team
        h2h_avg_points_scored_home_team = np.mean([g.points_scored for g in home_team_games]) if home_team_games else 0.0
        h2h_avg_points_conceded_home_team = np.mean([g.points_conceded for g in home_team_games]) if home_team_games else 0.0
        h2h_avg_point_diff_home_team = np.mean([g.point_diff for g in home_team_games]) if home_team_games else 0.0

        # New metrics for away team
        h2h_avg_points_scored_away_team = np.mean([g.points_scored for g in away_team_games]) if away_team_games else 0.0
        h2h_avg_points_conceded_away_team = np.mean([g.points_conceded for g in away_team_games]) if away_team_games else 0.0
        h2h_avg_point_diff_away_team = np.mean([g.point_diff for g in away_team_games]) if away_team_games else 0.0

        return {
            "h2h_total_games": h2h_total_games,
            "h2h_home_win_percentage": h2h_home_win_percentage,
            "h2h_away_win_percentage": h2h_away_win_percentage,
            "h2h_avg_home_points": h2h_avg_home_points,
            "h2h_avg_away_points": h2h_avg_away_points,
            "h2h_home_win_rate": h2h_home_win_percentage,  # Keeping for compatibility
            "h2h_avg_points_scored_home_team": h2h_avg_points_scored_home_team,
            "h2h_avg_points_conceded_home_team": h2h_avg_points_conceded_home_team,
            "h2h_avg_point_diff_home_team": h2h_avg_point_diff_home_team,
            "h2h_avg_points_scored_away_team": h2h_avg_points_scored_away_team,
            "h2h_avg_points_conceded_away_team": h2h_avg_points_conceded_away_team,
            "h2h_avg_point_diff_away_team": h2h_avg_point_diff_away_team,
        }
    
    def calculate_months_into_season(self, date: datetime) -> float:
        month = date.month
        if month >= 11:  # November or December
            season_start = date.replace(month=11, day=1)
        else:  # January through October
            season_start = date.replace(year=date.year-1, month=11, day=1)
        
        # Calculate months as a float for more granular tracking
        days_into_season = (date - season_start).days
        return days_into_season / 30.44  # Average days in a month

    def process_dataframe(self, df: pd.DataFrame, historical_df: pd.DataFrame = None, recent_games: int = 5) -> pd.DataFrame:
        """
        Main processing function with vectorized operations where possible.
        If historical_df is provided, only process new games while maintaining historical context.
        """
        # Verify format compatibility if historical data exists
        if historical_df is not None:
            if set(historical_df.columns) != set(df.columns):
                logging.error("Historical data format doesn't match current data format")
                logging.error("Consider regenerating all features by passing historical_df=None")
                raise ValueError("Column mismatch between historical and new data")
            
            # Initialize teams from historical data first
            for team in pd.concat([historical_df["home_team"], historical_df["away_team"]]).unique():
                if team not in self.teams:
                    self.teams[team] = TeamStats()
            
            # Process historical data to build up team states
            chunk_size = 1000
            for start_idx in tqdm(range(0, len(historical_df), chunk_size), desc="Processing historical data"):
                end_idx = min(start_idx + chunk_size, len(historical_df))
                chunk = historical_df.iloc[start_idx:end_idx]
                self._process_chunk(chunk, pd.DataFrame(), start_idx, recent_games, update_features=False)

        # Initialize any new teams from the new data
        for team in pd.concat([df["home_team"], df["away_team"]]).unique():
            if team not in self.teams:
                self.teams[team] = TeamStats()

        # Pre-allocate feature DataFrame for new data
        feature_cols = self._get_feature_columns(recent_games)
        features = pd.DataFrame(0.0, index=df.index, columns=feature_cols)

        # Process new data in chunks
        chunk_size = 1000
        for start_idx in tqdm(range(0, len(df), chunk_size), desc="Processing new data"):
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]
            self._process_chunk(chunk, features, start_idx, recent_games, update_features=True)

        # Combine original data with features
        result = pd.concat([df, features], axis=1)
        
        if historical_df is not None:
            result = pd.concat([historical_df, result]).sort_values("date")
        
        return result

    def _process_chunk(self, chunk: pd.DataFrame, features: pd.DataFrame, start_idx: int, recent_games: int, update_features: bool = True) -> None:
        """Process a chunk of data, optionally updating features"""
        for idx, row in chunk.iterrows():
            date = row["date"]
            home_team, away_team = row["home_team"], row["away_team"]
            home_score, away_score = row["home_team_score"], row["away_team_score"]

            if update_features:
                # Calculate features
                self._calculate_team_features(
                    features, idx,
                    self.teams[home_team], self.teams[away_team],
                    home_team, away_team,
                    date, recent_games,
                    home_score, away_score
                )

            # Update team histories (always do this)
            game_result = int(home_score > away_score)
            point_diff = home_score - away_score

            home_game = GameStats(
                date=date, opponent=away_team,
                points_scored=home_score, points_conceded=away_score,
                win=game_result, is_home=True, point_diff=point_diff
            )

            away_game = GameStats(
                date=date, opponent=home_team,
                points_scored=away_score, points_conceded=home_score,
                win=1 - game_result, is_home=False, point_diff=-point_diff
            )

            # Update team histories and head-to-head history
            self.teams[home_team].add_game(home_game)
            self.teams[away_team].add_game(away_game)
            self._update_head_to_head_history(home_team, away_team, home_game, away_game)

    def _calculate_team_features(
        self, features: pd.DataFrame, idx: int,
        home_team_stats: TeamStats, away_team_stats: TeamStats,
        home_team_name: str, away_team_name: str,
        date: datetime, recent_games: int,
        home_score: int, away_score: int
        ) -> None:
        # Get recent stats
        home_recent = home_team_stats.get_recent_stats(date, recent_games)
        away_recent = away_team_stats.get_recent_stats(date, recent_games)

        # Get venue stats
        home_venue_rate, home_venue_points = home_team_stats.get_venue_stats(True, before_date=date)
        away_venue_rate, away_venue_points = away_team_stats.get_venue_stats(False, before_date=date)

        # Get head-to-head stats
        h2h_stats = self.get_head_to_head_stats(home_team_name, away_team_name, date, recent_games)

        # Get time since last game
        home_time_since_last_game = home_team_stats.get_time_since_last_game(date)
        away_time_since_last_game = away_team_stats.get_time_since_last_game(date)

        # Get away game streak
        home_away_streak = home_team_stats.get_away_game_streak(date)
        away_away_streak = away_team_stats.get_away_game_streak(date)

        # Get overall stats
        home_overall_stats = home_team_stats.get_overall_stats(date)
        away_overall_stats = away_team_stats.get_overall_stats(date)

        # Set features
        if not (np.isnan(home_score) or np.isnan(away_score)): features.at[idx, "home_team_won"] = int(home_score > away_score)
        features.at[idx, f"home_win_last_{recent_games}"] = home_recent["wins"]
        features.at[idx, f"away_win_last_{recent_games}"] = away_recent["wins"]
        features.at[idx, f"home_total_points_scored_last_{recent_games}"] = home_recent["total_points_scored"]
        features.at[idx, f"away_total_points_scored_last_{recent_games}"] = away_recent["total_points_scored"]
        features.at[idx, f"home_avg_point_diff_last_{recent_games}"] = home_recent["avg_point_diff"]
        features.at[idx, f"away_avg_point_diff_last_{recent_games}"] = away_recent["avg_point_diff"]
        features.at[idx, "home_team_home_win_rate"] = home_venue_rate
        features.at[idx, "away_team_away_win_rate"] = away_venue_rate

        # Add head-to-head features
        features.at[idx, "h2h_total_games"] = h2h_stats["h2h_total_games"]
        features.at[idx, "h2h_home_win_percentage"] = h2h_stats["h2h_home_win_percentage"]
        features.at[idx, "h2h_away_win_percentage"] = h2h_stats["h2h_away_win_percentage"]
        features.at[idx, "h2h_avg_home_points"] = h2h_stats["h2h_avg_home_points"]
        features.at[idx, "h2h_avg_away_points"] = h2h_stats["h2h_avg_away_points"]
        features.at[idx, "h2h_home_win_rate"] = h2h_stats["h2h_home_win_rate"]
        features.at[idx, "h2h_avg_points_scored_home_team"] = h2h_stats["h2h_avg_points_scored_home_team"]
        features.at[idx, "h2h_avg_points_conceded_home_team"] = h2h_stats["h2h_avg_points_conceded_home_team"]
        features.at[idx, "h2h_avg_point_diff_home_team"] = h2h_stats["h2h_avg_point_diff_home_team"]
        features.at[idx, "h2h_avg_points_scored_away_team"] = h2h_stats["h2h_avg_points_scored_away_team"]
        features.at[idx, "h2h_avg_points_conceded_away_team"] = h2h_stats["h2h_avg_points_conceded_away_team"]
        features.at[idx, "h2h_avg_point_diff_away_team"] = h2h_stats["h2h_avg_point_diff_away_team"]

        # Add time since last game
        features.at[idx, "home_time_since_last_game"] = home_time_since_last_game
        features.at[idx, "away_time_since_last_game"] = away_time_since_last_game

        # Add away game streak
        features.at[idx, "home_away_game_streak"] = home_away_streak
        features.at[idx, "away_away_game_streak"] = away_away_streak

        # Add average points scored and conceded
        features.at[idx, "home_avg_points_scored"] = home_overall_stats["avg_points_scored"]
        features.at[idx, "home_avg_points_conceded"] = home_overall_stats["avg_points_conceded"]
        features.at[idx, "away_avg_points_scored"] = away_overall_stats["avg_points_scored"]
        features.at[idx, "away_avg_points_conceded"] = away_overall_stats["avg_points_conceded"]

        # Add time into season
        features.at[idx, "months_into_season"] = self.calculate_months_into_season(date)

    @staticmethod
    def _get_feature_columns(recent_games: int) -> List[str]:
        """Return list of feature column names"""
        return [
            "home_team_won",
            f"home_win_last_{recent_games}",
            f"away_win_last_{recent_games}",
            f"home_total_points_scored_last_{recent_games}",
            f"away_total_points_scored_last_{recent_games}",
            f"home_avg_point_diff_last_{recent_games}",
            f"away_avg_point_diff_last_{recent_games}",
            "home_team_home_win_rate",
            "away_team_away_win_rate",
            "h2h_total_games",
            "h2h_home_win_percentage",
            "h2h_away_win_percentage",
            "h2h_avg_home_points",
            "h2h_avg_away_points",
            "h2h_home_win_rate",
            "h2h_avg_points_scored_home_team",
            "h2h_avg_points_conceded_home_team",
            "h2h_avg_point_diff_home_team",
            "h2h_avg_points_scored_away_team",
            "h2h_avg_points_conceded_away_team",
            "h2h_avg_point_diff_away_team",
            "home_time_since_last_game",
            "away_time_since_last_game",
            "home_away_game_streak",
            "away_away_game_streak",
            "home_avg_points_scored",
            "home_avg_points_conceded",
            "away_avg_points_scored",
            "away_avg_points_conceded",
            "months_into_season"
        ]

def main():
    # Clear the terminal screen
    os.system("cls" if platform.system() == "Windows" else "clear")

    # Load configuration
    BASE_DIRECTORY = os.path.dirname(__file__) + "/"
    with open(BASE_DIRECTORY + "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Process data
    df = pd.read_parquet(config["processed_data_file"])
    
    # Initialize feature engineering
    feature_engineering = FeatureEngineering()
    output_file = BASE_DIRECTORY + "processed_data_with_features.parquet"
    
    # Determine feature columns to be generated
    expected_feature_columns = feature_engineering._get_feature_columns(5)  # Assuming 5 recent games

    # Check if processed features file exists
    if os.path.exists(output_file):
        existing_features = pd.read_parquet(output_file)
        
        # Check if existing feature columns match expected columns
        existing_feature_cols = [col for col in existing_features.columns if col not in df.columns]
        
        # Determine if we need to regenerate all features
        need_full_regeneration = set(existing_feature_cols) != set(expected_feature_columns)
        
        if need_full_regeneration:
            logging.info("Feature columns do not match. Regenerating entire feature dataset.")
            df_with_features = feature_engineering.process_dataframe(df)
        else:
            # Find dates not yet processed
            processed_dates = set(existing_features["date"])
            new_dates_df = df[~df["date"].isin(processed_dates)]
            
            if len(new_dates_df) == 0:
                logging.info("No new dates to process. Skipping feature generation.")
                return
            
            logging.info(f"Found {len(new_dates_df)} new dates to process")
            
            # Add features for new dates
            new_features = feature_engineering.process_dataframe(df = new_dates_df, historical_df = existing_features)
            
            # Combine existing and new features
            df_with_features = pd.concat([existing_features, new_features], ignore_index=True)
            
            # Sort by date to maintain chronological order
            df_with_features = df_with_features.sort_values("date").reset_index(drop=True)
    else:
        # If no existing features, process entire dataframe
        df_with_features = feature_engineering.process_dataframe(df)

    # Save results
    df_with_features.to_parquet(output_file)
    logging.info(f"Saved feature-engineered data to {output_file}")
    logging.info(f"Total rows in feature dataset: {len(df_with_features)}")
    logging.info(f"Date range: {df_with_features['date'].min()} to {df_with_features['date'].max()}")

if __name__ == "__main__":
    main()
