import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Config 
FANDOM_API = "https://scrubs.fandom.com/api.php"
ROOT = Path(os.environ.get("PROJECT_ROOT"))
OUTPUT_DIR = ROOT / "data" / "transcripts"
MAX_WORDS = 150 # Maximum words per scene
DELAY = 1.2 # Delay between requests to avoid being rate-limited   
HEADERS    = {"User-Agent": "ScrubsTranscriptScraper"}
IRRELEVANT_TAGS = "table, figure, sup, style, script, .mw-editsection, .toc, #toc, .printfooter, .asst-ad"
IGNORE_PHRASES = [
    "scrubs episode", 
    "transcripts", 
    "browse more", 
    "back to \"", 
    "following is a transcript",
    "editorial note"
]


def query_fandom_api(
    category: str, query_type: str) -> list[dict]:
    """
    Queries the Fandom API for a category and returns the results as a list
    of dictionaries
    Args:
        category: The name of the category to query
        query_type: The type of query to perform ('page', 'subcat', or 'file')
    Returns:
        list[dict]: A list of dictionaries containing the member information
    """
    results = []
    
    # query_parameters defines what we want to ask the fandom API
    query_parameters = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category, # The category to get from the fandom API
        "cmtype": query_type, # 'page', 'subcat', or 'file'
        "cmlimit": "500", # Maximum results per request
        "format": "json", # Return data as JSON
    }

    while True:
        # Perform the network request
        api_response = requests.get(
            FANDOM_API, 
            params=query_parameters, 
            headers=HEADERS, 
            timeout=15
        )
        
        # Check for any errors
        api_response.raise_for_status()
        
        # Parse the JSON data into a Python dictionary
        response_data = api_response.json()
        
        # Get category members (pages or subcategories) from query
        category_members = response_data.get(
            "query", {}).get("categorymembers", []
        )
        results.extend(category_members)
        
        # If more results to get, add the continue token 
        if "continue" in response_data:
            query_parameters["cmcontinue"] = response_data["continue"]["cmcontinue"]
        else:
            # Else no more results to get
            break
            
    return results

def get_episodes_by_season(season_title: str) -> list[tuple[int, str]]:
    """
    Scrapes the season transcripts category page for (episode_number, page_title).
    Table has one row with multiple columns; each cell can list several episodes.
    """
    # Get the list of episodes from the season transcripts page
    html = get_html_from_page(f"Category:{season_title.replace(" ", "_")}")
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output")


    # Loop through table cells in episode list
    episodes = []
    for episode in content.find_all(["td", "li"]):
        
        # Get row of episodes in table cell
        row_text = episode.get_text(strip=True)
        
        # Parse row into groups of (episode number, episode title)
        for episode_number, episode_title in re.findall(
            r"(\d+)\.\s*\"(.+?)\"", row_text
        ):
            episodes.append((int(episode_number), episode_title.strip()))
            
    # Return the list of episodes sorted by number
    return sorted(episodes)

def get_episode_titles_by_season() -> dict[str, list[str]]:
    """
    Gets the episode titles by season
    Returns:
        dict[str, list[str]]: A dictionary of season titles and their 
        corresponding episode titles
    """

    # Get all Season subcategories within the main Transcripts category
    response = query_fandom_api(
        "Category:Transcripts", "subcat"
    )
    
    # Identify Season subcategories, so we can get episode pages
    seasons = [season for season in response if "Season" in season["title"]]

    # Loop through each Season to collect the episode titles
    episodes_by_season = {}
    for season in seasons:
        
        # Remove "Category:" from the season title
        season_title = season["title"].replace("Category:", "")
        
        # Get the transcript pages within the season
        time.sleep(DELAY)
        episodes = query_fandom_api(season["title"], "page")
        
        # Store the list of page titles as part of the season 
        episodes_by_season[season_title] = (
            [episode["title"] for episode in episodes]
        )
        
    # Return the episodes by season dictionary
    return episodes_by_season

def get_html_from_page(page: str) -> str:
    """
    Downloads the HTML content of a wiki page from the fandom API.
    Args:
        page: The title of the page to get the HTML from
    Returns:
        str: The HTML content of the page
    """
    request_settings = {
        "action": "parse",
        "page": page,
        "prop": "text", # Ignore metadata
        "disablelimitreport": "1", # Ignore other irrelevant information
        "format": "json",
    }

    # Make request
    response = requests.get(
        FANDOM_API, params=request_settings, headers=HEADERS, timeout=15
    )
    
    # Check for any errors
    response.raise_for_status()
    
    # Get JSON response from the fandom API
    json_data = response.json()

    # Check for page content in JSON Response
    if "parse" not in json_data:
        raise ValueError(f"Could not find page content for: {page}")

    # Get the HTML string from the text field
    return json_data["parse"]["text"]["*"]

def html_to_lines(html: str) -> list[str]:
    """
    Converts HTML to a list of lines
    Args:
        html: The HTML content of the page
    Returns:
        list[str]: A list of lines
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output")
    
    # Remove irrelevant tags
    for tag in content.select(IRRELEVANT_TAGS):
        tag.decompose()

    # Loop through all paragraphs and list items on the transcript page
    lines = []
    for element in content.find_all(["p", "li"]):
        text = element.get_text(separator="\n", strip=True)
        
        # Split the transcript text into lines
        for line in text.splitlines():
            
            # Remove extra whitespace
            line = line.strip() 
            
            # Skip empty lines
            if not line:
                continue

            # Skip lines that contain an ignored phrase
            if any(phrase in line.lower() for phrase in IGNORE_PHRASES):
                continue
            
            # Skip lines that are just a single quote or punctuation
            if len(line) < 2 and line in "\".!,":
                continue

            # Skip consecutive duplicates 
            # (I did this after noticing some transcripts have duplicates)
            if lines and lines[-1] == line:
                continue
            lines.append(line)

    return lines


def batch_lines(lines: list[str], season_num: int, ep_num: int) -> list[dict]:
    """
    Groups lines into batches of up to MAX_WORDS words.
    Args:
        lines: A list of lines to batch
        season_num: The season number
        ep_num: The episode number
    Returns:
        list[dict]: A list of dictionaries containing the scene information
    """
    scenes = []
    current_scene = []
    current_word_count = 0

    for line in lines:
        
        # Count the words in the line
        line_words = len(line.split())
        
        # If the current scene is full, add it to the list of scenes
        if current_word_count + line_words > MAX_WORDS and current_scene:
            scenes.append("\n".join(current_scene))
            current_scene, current_word_count = [], 0
            
        # Add the line to the scene being built
        current_scene.append(line)
        current_word_count += line_words

    # Add final scene
    if current_scene:
        scenes.append("\n".join(current_scene))

    # Format the scenes into dictionaries with scene id, position, and text
    num_scenes = len(scenes)
    output = []
    for scene_index, text in enumerate(scenes):
        
        # Check far along the episode the batch is (rounded to 4 decimal places)
        pos = round(scene_index / (num_scenes - 1), 4)
        
        # Add the batch to the output list with the scene id, position, and text
        output.append({
            "scene_id": f"{season_num}_{ep_num:02d}_{scene_index + 1}",
            "position": pos,
            "text": text,
        })

    return output

def main():

    # Get the list of season titles
    response = query_fandom_api(
        category="Category:Transcripts", query_type="subcat"
    )
    seasons = [
        season["title"].replace("Category:", "") # Remove "Category:" from title
        for season in response if "Season" in season["title"]
    ]

    # Loop through each season to get the episode titles
    for season_title in seasons:
        
        # Get the season number and create corresponding directory
        season_num = int(re.search(r"\d+", season_title).group())
        season_dir = OUTPUT_DIR / season_title
        os.makedirs(season_dir, exist_ok=True)

        # Loop through each episode in the season to get the transcript
        for episode_number, episode_title in get_episodes_by_season(season_title):
            
            # Print the season, episode number, and title
            print(f"  S{season_num} E{episode_number:02d}: {episode_title}")
            
            # Get the scenes with text from the episode page
            try:
                # Get get html from page and convert to scenes
                time.sleep(DELAY)
                page_title = f"{episode_title} transcript"
                lines = html_to_lines(get_html_from_page(page_title))
                scenes = batch_lines(lines, season_num, episode_number)

                episode_data = {
                    "source": page_title,
                    "episode_id": f"{season_num}_{episode_number:02d}",
                    "scenes": scenes
                }

                # Save the episode data to a JSON file
                filename = f"ep_{episode_number:02d}_{episode_title}.json"
                with open(season_dir / filename, "w", encoding="utf-8") as f:
                    json.dump(episode_data, f, indent=2, ensure_ascii=False)
                    
            # If hit an error getting and saving the episode, print it
            except Exception as e:
                print(f"Error: {e}")

        
if __name__ == "__main__":
    main()