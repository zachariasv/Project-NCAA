import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Base URL for NCAA Women’s Basketball results
BASE_URL = "https://www.oddsportal.com/basketball/usa/ncaa-women/results/"
OUTPUT_FILE = "Model Improvement/Women's Games/ncaa_women_odds.csv"

# Initialize WebDriver
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
#options.add_argument("--headless")
#options.add_argument("--disable-gpu")
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 10)


def get_max_page():
    driver.get(BASE_URL)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.eventRow")))
    links = driver.find_elements(By.CSS_SELECTOR, "a.pagination-link[data-number]")
    pages = [int(link.get_attribute("data-number")) for link in links if link.get_attribute("data-number").isdigit()]
    return max(pages) if pages else 1


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
all_games = []
max_page = get_max_page()
print(f"Detected {max_page} pages to scrape.")

# Load first page
driver.get(BASE_URL)
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.eventRow")))

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
