import time
import pandas as pd
import os
import argparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- Command-line argument parsing ---
parser = argparse.ArgumentParser(description="Scrape NCAA basketball odds from OddsPortal.")
parser.add_argument(
    "gender",
    type=str,
    choices=["men", "women"],
    help="Specify whether to scrape men's or women's games."
)
args = parser.parse_args()

# --- Configuration based on gender ---
if args.gender == "women":
    BASE_URL = "https://www.oddsportal.com/basketball/usa/ncaa-women/results/"
    OUTPUT_FILE = "Model Improvement/Women's Games/ncaa_women_odds.csv"
    SEASONS = [f"{year}-{year+1}" for year in range(2024, 2025)] # Only 24/25 is available
else: # men
    BASE_URL = "https://www.oddsportal.com/basketball/usa/ncaa/results/"
    OUTPUT_FILE = "Model Improvement/Men's Games/ncaa_men_odds.csv"
    SEASONS = [f"{year}-{year+1}" for year in range(2015, 2024)]

# Initialize WebDriver
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
#options.add_argument("--headless")
#options.add_argument("--disable-gpu")
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 10)


def extract_event_data(row, page):
    # Extract date
    date_text = None
    try:
        date_text = row.find_element(By.CSS_SELECTOR, "[data-testid='date-header']").text.strip()
    except:
        pass

    # Locate game-row container
    try:
        game_data = row.find_element(By.CSS_SELECTOR, "[data-testid='game-row']")
    except:
        return None

    # Extract teams
    home_team, away_team = None, None
    try:
        names = game_data.find_elements(By.CSS_SELECTOR, "p.participant-name")
        if len(names) >= 2:
            home_team = names[0].text.strip()
            away_team = names[1].text.strip()
    except:
        pass

    # Extract game URL
    game_url = None
    try:
        game_url = game_data.find_element(By.TAG_NAME, "a").get_attribute("href")
    except:
        pass

    # Extract time
    time_text = None
    try:
        time_text = game_data.find_element(By.CSS_SELECTOR, "[data-testid='time-item']").text.strip()
    except:
        pass

    # Extract odds
    odds_elements = game_data.find_elements(
        By.CSS_SELECTOR,
        "[data-testid='odd-container-winning'], [data-testid='odd-container-default']"
    )
    odds = [el.text.strip() for el in odds_elements]

    return {
        "date": date_text,
        "time": time_text,
        "home_team": home_team,
        "away_team": away_team,
        "home_odds": odds[0] if len(odds) > 0 else None,
        "draw_odds": odds[1] if len(odds) > 1 else None,
        "away_odds": odds[2] if len(odds) > 2 else None,
        "game_url": game_url,
        "page": page
    }

# Main scraping logic
if os.path.exists(OUTPUT_FILE):
    print(f"Loading existing data from {OUTPUT_FILE}")
    existing_df = pd.read_csv(OUTPUT_FILE)
    all_games = existing_df.to_dict('records')
    print(f"Loaded {len(all_games)} existing records.")
else:
    all_games = []

for season in SEASONS:
    print(f"Scraping season: {season}")
    # Handle URL format difference for men's recent season
    if args.gender == 'men' and season == '2023-2024':
        season_url = BASE_URL
    else:
        season_url = BASE_URL.replace('/results/', f'-{season}/results/')
    driver.get(season_url)
    
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.eventRow")))
    except:
        print(f"No data found for season {season}. Skipping.")
        continue

    # Get the number of pages for the current season
    links = driver.find_elements(By.CSS_SELECTOR, "a.pagination-link[data-number]")
    pages = [int(link.get_attribute("data-number")) for link in links if link.get_attribute("data-number").isdigit()]
    max_page = max(pages) if pages else 1
    print(f"Detected {max_page} pages to scrape for season {season}.")

    for page in range(1, max_page + 1):
        print(f"Scraping page {page}...")
        if page > 1:
            try:
                # Find and click the pagination link
                pagination_links = driver.find_elements(By.CSS_SELECTOR, "a.pagination-link[data-number]")
                for link in pagination_links:
                    if link.get_attribute("data-number") == str(page):
                        # Scroll only to the pagination container
                        driver.execute_script(
                            "arguments[0].parentNode.scrollIntoView({block:'nearest'});", link
                        )
                        time.sleep(0.5)
                        # Click via JS
                        driver.execute_script("arguments[0].click();", link)
                        break
                # Wait for new events to load
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.eventRow")))
                time.sleep(1)
                # Scroll back up to top of results
                driver.execute_script("window.scrollTo(0,0);")
            except Exception as e:
                print(f"Failed to navigate to page {page}: {e}")
                break

        # Collect event rows
        rows = driver.find_elements(By.CSS_SELECTOR, "div.eventRow")
        print(f"Found {len(rows)} events on page {page}.")
        if not rows:
            break

        # Extract data
        for row in rows:
            data = extract_event_data(row, page)
            if data and data not in all_games:
                all_games.append(data)

        # Save progress
        pd.DataFrame(all_games).to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(all_games)} records so far.")
        time.sleep(1)

# Cleanup
driver.quit()
print(f"Finished scraping {len(all_games)} total events.")