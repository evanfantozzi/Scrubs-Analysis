from google import genai
import os 
from pathlib import Path
import json
import time
from dotenv import load_dotenv
load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
CLIENT = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash-lite"
ROOT = Path(os.environ.get("PROJECT_ROOT"))
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
GEMINI_PREDICTIONS_DIR = ROOT / "data" / "gemini_predictions"
LABELED_SCENES_PREDICTIONS_PATH = GEMINI_PREDICTIONS_DIR / "labeled_scenes" / f"{MODEL}.json"
OUTPUT_DIR = GEMINI_PREDICTIONS_DIR / "all_scenes_with_transcripts" 

from predict_labeled_scenes import prompt, parse_response

def load_existing_predictions() -> list[dict]:
    """
    Try to load existing predictions for all scenes. If it doesn't exist, load
    existing predictions for labeled scenes. 
    
    Returns:
        Dict of scene_id: scene with predictions
    """
    # Doing this because the first time I tried this tonight, I got a 503 error 
    try:
        with open(OUTPUT_DIR / f"{MODEL}.json", 'r') as f:
            scenes = json.load(f)
            return {scene["scene_id"]: scene for scene in scenes}
    except:
        with open(LABELED_SCENES_PREDICTIONS_PATH, 'r') as f:
            scenes = json.load(f)
            return {scene["scene_id"]: scene for scene in scenes}

def load_scenes_from_transcripts() -> dict[str, dict]:
    """
    Load all scenes in transcripts directory, return as dict of scenes
    """
    scenes = {}
    
    # Loop through seasons
    for season_dir in sorted(TRANSCRIPTS_DIR.iterdir()):
        
        # Loop through episodes
        for episode_transcript in sorted(season_dir.iterdir()):
            
            # Load episode transcript
            with open(episode_transcript, 'r') as f:
                episode = json.load(f)
                
                # Loop through scenes, adding to scenes dict
                for scene_idx, scene in enumerate(episode["scenes"]):
                    
                    # Add previous scene text to scene
                    if scene_idx == 0:
                        scene["prev_scene_text"] = ""
                    else:
                        scene["prev_scene_text"] = (
                            episode["scenes"][scene_idx - 1]["text"]
                        )
                        
                    # Add episode id to scene
                    id_parts = scene["scene_id"].split("_")
                    scene["episode_id"] = f"{id_parts[0]}_{id_parts[1]}"
                    
                    # Add scene to scenes dict
                    scenes[scene["scene_id"]] = scene
    
    return scenes   
    
def main():
    """
    Main function to predict all scenes with transcripts on Scrubs Fandom Wiki
    using the flash lite model
    """
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing predictions
    scene_predictions = load_existing_predictions()
    
    # Load transcripts
    scenes = load_scenes_from_transcripts()
    
    # Loop through transcripts, checking if they've already been predicted
    for scene_id, scene in scenes.items():
        
        # Check if the scene has already been predicted
        if scene_predictions.get(scene_id):
            print(f"Scene {scene_id} already predicted with model {MODEL}")
            continue
        
        # Prompt the model to predict the transcript
        time.sleep(0.5) # Avoid hitting rate limit
        print(f"Predicting scene {scene['scene_id']} with model {MODEL}")
        response = prompt(scene, MODEL)
        predicted_funny, predicted_sad = parse_response(response)
        
        # Add scene with predictions to dict
        scene_predictions[scene_id] = {
            "episode_id": scene["episode_id"],
            "scene_id": scene["scene_id"],
            "position": scene["position"],
            "prev_scene_text": scene["prev_scene_text"],
            "text": scene["text"],
            "predicted_funny": predicted_funny,
            "predicted_sad": predicted_sad,
        }
        
        # Update predictions JSON 
        with open(OUTPUT_DIR / f"{MODEL}.json", 'w') as f:
            json.dump(list(scene_predictions.values()), f)
        
if __name__ == "__main__":
    main()