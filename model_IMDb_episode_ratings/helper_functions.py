import json
import os
from pathlib import Path
from dotenv import load_dotenv
import statistics as st

load_dotenv()

# Config
GEMINI_MODEL = "gemini-2.5-flash-lite"
DEBERTA_MODEL = "microsoft_deberta_v3_base_SGD_without_dropout_cls_results"
ROOT = Path(os.environ.get("PROJECT_ROOT"))
DATA_DIR = ROOT / "data"
IMDb_RATINGS_PATH = DATA_DIR / "imdb_episode_ratings.json"
OUTPUT_DIR = DATA_DIR / "modeled_episode_ratings"
DATA_SOURCES = ["labels", "gemini", "deberta"]
DATA_PATHS = {
    "labels": 
        DATA_DIR / "labeled_scenes.json",
    "deberta": 
        DATA_DIR / "DeBERTa_predictions" / "all_scenes_with_transcripts" / f"{DEBERTA_MODEL}.json",
    "gemini":
        DATA_DIR / "gemini_predictions" /  "all_scenes_with_transcripts" / f"{GEMINI_MODEL}.json",
}

def load_episodes() -> dict[dict, dict, dict]:
    """
    Return dict of 3 dicts, one per data source with scene predictions/labels:
    [manual labels, gemini predictions, DeBERTa predictions]
    """
    
    # Read in labels/predictions to data dictionary
    data = {}
   
    for source in DATA_SOURCES:    
        with open(DATA_PATHS[source]) as f:
            data[source] = json.load(f)
    
    # Loop through each data source
    episodes_per_source = {source: {} for source in DATA_SOURCES}
    for source in DATA_SOURCES:
        
        # Initialize list of scenes for the episode
        scenes_per_episode = {}
        
        for scene in data[source]:
            # Add scene to corresponding episode list in dictionary
            episode_id = scene["episode_id"]
            scenes = scenes_per_episode.get(episode_id, [])
            scenes.append(scene)
            scenes_per_episode[episode_id] = scenes
            
        # Add complete dataset to list to return
        episodes_per_source[source] = scenes_per_episode
    
    return episodes_per_source

def calculate_rating_metrics(episodes_per_source) -> dict:
    """
    Calculate mean, variance and oscillation for funny/sad ratings
    """

    # Track metrics on all three data sources
    metrics = {source: {} for source in DATA_SOURCES}
    
    # Calculate metrics for each episode
    for source, episodes_dict in episodes_per_source.items():
        
        source_metrics = {}
        for episode_id, episode in episodes_dict.items():
            
            # Keep track of ratings from curr/prev scene to calculate metrics
            funny_ratings, sad_ratings, funny_change, sad_change = [], [], [], []
            prev_sad, prev_funny = None, None
            
            for scene in episode:
                # Get funny/sad ratings
                funny = int(
                    scene["funny"] if source == "labels" else scene["predicted_funny"]
                )
                sad = int(
                    scene["sad"] if source == "labels" else scene["predicted_sad"]
                )
                
                # If either rating is invalid (could happen from Gemini),
                # then don't add labels
                if not((1 <= funny <= 5) and (1 <= sad <= 5)):
                    prev_sad, prev_funny = None, None # set for next scene
                    continue
                
                # Add ratings to lists
                funny_ratings.append(funny)
                sad_ratings.append(sad)
                
                # Compare to previous scene, adding to tally 
                if prev_sad and prev_funny:
                    funny_change.append(abs(funny - prev_funny))
                    sad_change.append(abs(sad - prev_sad))
                    
                # Set ratings as previous ratings for next scene 
                prev_sad, prev_funny = sad, funny 
         
            # Calculate metrics 
            # I used sample variance instead of population variance since we
            # don't have transcripts for all episodes
            source_metrics[episode_id] = {
                # Funny metrics
                "funny_mean": st.mean(funny_ratings),
                "funny_var": st.variance(funny_ratings), 
                "funny_change": st.mean(funny_change),
                
                # Sad metrics
                "sad_mean": st.mean(sad_ratings),
                "sad_var": st.variance(sad_ratings), 
                "sad_change": st.mean(sad_change),
            }
        
        metrics[source] = source_metrics
    
    return metrics

def calculate_imdb_metrics() -> dict:
    """
    Load imdb ratings and output dictionary mapping episodes to overall rating,
    variance in rating, share of 1 ratings, and share of 10 ratings
    """

    # Load data
    with open(IMDb_RATINGS_PATH) as f:
        episodes = json.load(f)
    
    # Loop through scraped imdb episodes
    metrics = {} 
    for episode in episodes:
        
        # Build list of ratings to calculate metrics
        ratings_lst = []
        for rating, count in episode["ratings_histogram"].items():
            ratings_lst.extend([int(rating)] * count)
        
        # Calculate metrics
        num_ratings = len(ratings_lst)
        metrics[episode["episode_id"]] = {
            "episode_id": episode["episode_id"],
            "imdb_rating_mean": st.mean(ratings_lst),
            "imdb_rating_variance": st.pvariance(ratings_lst),
            "imdb_rating_share_1" : ratings_lst.count(1) / num_ratings,
            "imdb_rating_share_10": ratings_lst.count(10) / num_ratings
        }

    return metrics

def merge_metrics_dicts(all_rating_metrics: dict, imdb_metrics: dict) -> dict:
    """
    Merge calculated IMDb metrics onto our funny/sad rating metrics
    """
    # Loop over three metrics dictionaries
    for metrics in all_rating_metrics.values():
        for episode_id, episode in metrics.items():
            episode.update(imdb_metrics[episode_id])