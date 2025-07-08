import os
import logging
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class MergeConfig:
    # Path to scraped odds CSV
    scraped_odds_path: str = "/Users/zacharias/Dropbox/Python/Project-NCAA/Model Improvement/Women's Games/ncaa_women_odds.csv"
    # Path to processed data with features
    processed_data_path: str = "/Users/zacharias/Dropbox/Python/Project-NCAA/Prediction/processed_data_with_features.parquet"
    # Team name mapping CSV
    name_mapping_path: str = "/Users/zacharias/Dropbox/Python/Project-NCAA/Model Improvement/Women's Games/team_name_mapping.csv"
    # Output path for merged data
    output_path: str = "/Users/zacharias/Dropbox/Python/Project-NCAA/Model Improvement/Women's Games/merged_data.parquet"
    # Date filter for processed data
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"

class DataMerger:
    def __init__(self, cfg: MergeConfig):
        self.cfg = cfg
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_team_mapping(self):
        logging.info(f"Loading team name mapping from {self.cfg.name_mapping_path}")
        df = pd.read_csv(self.cfg.name_mapping_path)
        # Expect columns: original_name, best_match
        self.name_map = dict(zip(df['original_name'], df['best_match']))
        logging.info(f"Loaded {len(self.name_map)} mappings")

    def load_scraped_odds(self):
        logging.info(f"Loading scraped odds from {self.cfg.scraped_odds_path}")
        df = pd.read_csv(self.cfg.scraped_odds_path)
        # forward-fill missing dates
        df['date'] = df['date'].ffill()
        df['date_clean'] = df['date'].str.split(' - ').str[0]
        # parse event_time
        df['event_time'] = pd.to_datetime(
            df['date_clean'] + ' ' + df['time'],
            format='%d %b %Y %H:%M',
            errors='coerce'
        )
        df = df.dropna(subset=['event_time'])
        df['event_date'] = df['event_time'].dt.date
        # remap suffix
        df['home_team'] = df['home_team'].str.replace(r' W$', '_F', regex=True)
        df['away_team'] = df['away_team'].str.replace(r' W$', '_F', regex=True)
        # map to processed names
        df['home_team_proc'] = df['home_team'].map(self.name_map).fillna(df['home_team'])
        df['away_team_proc'] = df['away_team'].map(self.name_map).fillna(df['away_team'])
        # cast odds
        df['home_odds'] = pd.to_numeric(df['home_odds'], errors='coerce')
        df['away_odds'] = pd.to_numeric(df['away_odds'], errors='coerce')
        # select relevant columns
        self.odds_df = df[[
            'event_date', 'home_team_proc', 'away_team_proc',
            'event_time', 'home_odds', 'away_odds'
        ]].dropna()
        logging.info(f"Processed scraped odds: {len(self.odds_df)} records")

    def load_processed_data(self):
        logging.info(f"Loading processed data from {self.cfg.processed_data_path}")
        df = pd.read_parquet(self.cfg.processed_data_path)
        df['date'] = pd.to_datetime(df['date'])
        # filter date range
        df = df[(df['date'] >= self.cfg.start_date) & (df['date'] <= self.cfg.end_date)]
        df['event_date'] = df['date'].dt.date
        # we keep all feature columns
        self.proc_df = df
        logging.info(f"Processed data: {len(self.proc_df)} records between {self.cfg.start_date} and {self.cfg.end_date}")

    def merge(self):
        logging.info("Merging processed data with odds data allowing ±1 day tolerance")
        # Prepare odds with datetime for date tolerance
        odds = self.odds_df.copy()
        odds['odds_date'] = pd.to_datetime(odds['event_date'])

        # Merge on team names only
        merged = pd.merge(
            self.proc_df,
            odds,
            left_on=['home_team', 'away_team'],
            right_on=['home_team_proc', 'away_team_proc'],
            how='inner'
        )
        # Ensure proc date is datetime
        merged['proc_date'] = pd.to_datetime(merged['date'])
        # Filter matches within ±1 day
        merged['date_diff'] = (merged['proc_date'] - merged['odds_date']).abs()
        tol = pd.Timedelta(days=1)
        merged = merged.loc[merged['date_diff'] <= tol].drop(columns=['date_diff', 'proc_date', 'odds_date'])
        logging.info(f"Merged dataset contains {len(merged)} records after ±1 day tolerance")
        self.merged_df = merged

    def save(self):
        logging.info(f"Saving merged data to {self.cfg.output_path}")
        self.merged_df.to_parquet(self.cfg.output_path, index=False)
        logging.info("Save complete")

if __name__ == '__main__':
    cfg = MergeConfig()
    merger = DataMerger(cfg)
    merger.load_team_mapping()
    merger.load_scraped_odds()
    merger.load_processed_data()
    merger.merge()
    merger.save()