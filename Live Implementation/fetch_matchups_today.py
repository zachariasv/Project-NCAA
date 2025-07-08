import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_todays_games():
    """Fetch today's NCAA basketball matchups."""
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    import time
    from datetime import datetime, timedelta
    
    dates = [datetime.now(), datetime.now() + timedelta(days=1)]
    
    # Setup headless browser
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    browser = webdriver.Chrome(options=options)
    
    
    # Initialize list for games
    games = []
    
    for date in dates:
        # Get today's games
        browser.get(f"https://www.sports-reference.com/cbb/boxscores/index.cgi?month={date.month}&day={date.day}&year={date.year}")
        time.sleep(1)

        # Find all game summaries
        elements = browser.find_elements(By.CLASS_NAME, "game_summary")
        
        for element in elements:
            if "hidden" not in element.get_attribute("class").split():
                try:
                    away_team = element.find_elements(By.CSS_SELECTOR, "td")[0].text.split(" (")[0]
                    home_team = element.find_elements(By.CSS_SELECTOR, "td")[3].text.split(" (")[0]
                    gender = element.get_attribute("class")[-1:]
                    if len(home_team) > 0 and len(away_team) > 0:
                        games.append({
                            "date": date.date(),
                            "home_team": home_team,
                            "away_team": away_team,
                            "gender": gender
                        })
                except Exception as e:
                    logging.error(f"Error processing game: {str(e)}")
                    continue
        
    browser.quit()
    return pd.DataFrame(games)

if __name__ == "__main__":
    games_df = fetch_todays_games()
    logging.info(f"\nFound {len(games_df)} games for today:")
    logging.info(f"\n{games_df}")