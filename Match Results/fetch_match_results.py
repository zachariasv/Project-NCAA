# Import packages
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from datetime import datetime, timedelta
import platform
import os
import logging

# Clear the terminal screen
os.system("cls" if platform.system() == "Windows" else "clear")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Get the directory of the current script
BASE_DIRECTORY = os.path.dirname(__file__)
WEBSCRAPE_MATCHES_FILE = BASE_DIRECTORY + "/webscraped_ncaa_games_history.parquet"

def initialize_data():
    """
    Load existing data or initialize the data if no existing database is found.
    """
    if not os.path.exists(WEBSCRAPE_MATCHES_FILE):
        logging.info("No existing database found, creating new database...")
        date = pd.to_datetime("1986-1-2").date() # Initialize with first date of data if no data is available
        columns = ["date", "home_team", "home_team_ranking", "home_team_score", "away_team", "away_team_ranking", "away_team_score", "gender"]
        df = pd.DataFrame(columns=columns)
        df.to_parquet(WEBSCRAPE_MATCHES_FILE)
    else:
        logging.info("Fetching data from existing database...")
        df = pd.read_parquet(WEBSCRAPE_MATCHES_FILE) # Load already fetched database
        date = pd.to_datetime(df.date.max()).date()
    return df, date

def setup_webdriver(date):
    """
    Set up the Chrome webdriver with the necessary options.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    browser = webdriver.Chrome(options=options)

    browser.get(f"https://www.sports-reference.com/cbb/boxscores/index.cgi?month={date.month}&day={date.day}&year={date.year}") # Load last fetched date
    return browser

def scrape_data(df, browser, date):
    """
    Scrape data from the website.
    """
    finished_scraping = False # Set up while loop

    while not finished_scraping:
        date = (date + timedelta(days = 1)) # Fetch next date

        if date.month == 5 and date.day == 1: date = date.replace(month=11, day=1) # If the date is May 1st, jump to November 1st of the same year
        
        if date > datetime.now().date(): # If the date is in the future, quit the script
            finished_scraping = True 
            browser.quit()
            continue

        browser.get(f"https://www.sports-reference.com/cbb/boxscores/index.cgi?month={date.month}&day={date.day}&year={date.year}")
        time.sleep(1)

        elements = browser.find_elements(By.CLASS_NAME, "game_summary") # Get all match result elements

        if date.day in [1, 11, 21]: # Periodically save data
            df.to_parquet("webscraped_ncaa_games_history.parquet")
            logging.info(f"Saved data on {date}.")

        if len(elements) > 0: # If there are any match results on the page
            for element in elements:
                try:
                    if "hidden" not in element.get_attribute("class").split():
                        home_team = element.find_elements(By.CSS_SELECTOR, "td")[0].text.split(" (")[0]
                        home_team_score = int(element.find_elements(By.CSS_SELECTOR, "td")[1].text)
                        away_team = element.find_elements(By.CSS_SELECTOR, "td")[3].text.split(" (")[0]
                        away_team_score = int(element.find_elements(By.CSS_SELECTOR, "td")[4].text)
                        gender = element.get_attribute("class")[-1:]
                        try: 
                            home_team_ranking = int(element.find_elements(By.CSS_SELECTOR, "td")[0].text.split(" (")[1].replace(") ", ""))
                            away_team_ranking = int(element.find_elements(By.CSS_SELECTOR, "td")[3].text.split(" (")[1].replace(") ", ""))
                        except:
                            home_team_ranking = np.nan
                            away_team_ranking = np.nan

                        df.loc[df.shape[0]] = [date, home_team, home_team_ranking, home_team_score, away_team, away_team_ranking, away_team_score, gender] # Insert new row into the bottom of the dataframe

                except Exception as e: # Simple error handling, often parsing of integers fails due to empty strings when elements are found but are empty on days with no matches. This is a crude but working fix.
                    pass
            logging.info(f"Finished collecting data for {date}!")
            
    return df

def main():
    df, date = initialize_data() # Load data
    browser = setup_webdriver(date) # Set up webdriver
    df = scrape_data(df, browser, date) # Scrape data

    df.drop_duplicates().reset_index(drop = True).to_parquet(WEBSCRAPE_MATCHES_FILE) # Save data to parquet file once finished
    date = pd.to_datetime(df.date.max()).date()
    logging.info(f"Saved data on {date}.")

if __name__ == "__main__":
    main()