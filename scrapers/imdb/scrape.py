import json
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Config
load_dotenv()
ROOT = Path(os.environ.get("PROJECT_ROOT"))
SCRAPE_DIR = ROOT / "scrapers" / "imdb"
URLS_PATH  = SCRAPE_DIR / "imdb_episode_urls.txt"
OUTPUT_PATH = ROOT / "data/imdb_episode_ratings.json"
API_KEY = os.getenv("SCRAPING_BEE_API_KEY")
API_BASE = "https://app.scrapingbee.com/api/v1"
HISTOGRAM_RATING_REGEX = re.compile(r"(\d+)\s+(\d+)-star\s+reviews", re.I) 
# Extract rules for scrapingbee API 
# Source: I used https://www.scrapingbee.com/blog/how-to-scrape-imdb/
EXTRACT_RULES = {
    "title": {"selector": "h1", "type": "item", "output": "text"},
    "ratings_histogram": {
        "selector": "[data-testid='rating-histogram']",
        "type": "item", "output": "html"
    },
    "season_episode": {
        "selector": "[data-testid='hero-subnav-bar-season-episode-numbers-section']",
        "type": "item", "output": "text"
    },
}


def parse_ratings_histogram(html: str) -> tuple[float, dict]:
    """
    Parse ratings histogram into overall rating and histogram of star counts
    Args:
        html: string of HTML the rating histogram
    Returns:
        Dictionary containing the overall rating and ratings histogram
    """
    parser = BeautifulSoup(html, "html.parser")
    root = parser.select_one("[data-testid='rating-histogram']")
    overall_rating = float(root.select_one("span.ipc-rating-star--rating").get_text())
    ratings_histogram = {}
    
    # Loop through each bar in the rating histogram
    for bar in root.select("a[data-testid^='rating-histogram-bar-']"):
        
        # Get the number of votes and star rating from each bar in histogram
        votes_stars = HISTOGRAM_RATING_REGEX.search(bar.get("aria-label", ""))
        if votes_stars:
            votes, stars = int(votes_stars.group(1)), int(votes_stars.group(2))
            ratings_histogram[stars] = votes
            
    return (overall_rating, ratings_histogram)

def scrape_episode(url: str) -> dict:
    """
    Scrape one URL and return the episode dictionary or error dictionary
    Args:
        url: The URL of the episode to scrape
    Returns:
        dict: Dictionary containing the episode data or error data
    """
    params = {
        "api_key": API_KEY,
        "url": url,
        "render_js": "false",
        "extract_rules": json.dumps(EXTRACT_RULES),
    }
    
    # Make request to scrapingbee API
    try:
        response = requests.get(API_BASE, params=params)
        
        # Check for any errors
        response.raise_for_status()
        
        # Parse JSON response
        episode = response.json()
    
    # If error, return error dictionary
    except (requests.exceptions.RequestException, ValueError) as e:
        return {"url": url, "error": str(e)}

    # Parse rating histogram
    overall_rating, ratings_histogram = parse_ratings_histogram(
        episode.get("ratings_histogram")
    )
    
    # Add overall rating, ratings histogram, and URL to the episode dictionary
    episode["overall_rating"] = overall_rating
    episode["ratings_histogram"] = ratings_histogram
    episode["url"] = url
    
    # Clean episode number to match labels (for example, from S1.E10 to 1_10)
    episode["season_episode"] = re.sub(r"S(\d+)\.E(\d+)", r"\1_\2", episode["season_episode"])
    
    # Return episode dictionary
    return episode


def main():
    """
    Scrape each episode from IMDb and extract information about it
    """
    
    # Input scraped episodes from URLs 
    episodes = []
    with open(URLS_PATH, "r") as f:
        for line in f:
            url = line.strip()
            if url:
                time.sleep(1) # to avoid being rate-limited
                episodes.append(scrape_episode(url))

    # Write the results to the output file
    with open(OUTPUT_PATH, "w") as f:
        json.dump(episodes, f, indent=2)


if __name__ == "__main__":
    main()