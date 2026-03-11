import pandas as pd
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(os.environ.get("PROJECT_ROOT"))
OUTPUT_PATH = ROOT / "data" / "labeled_scenes.json"
LABELS_PATH = ROOT / "data" / "labels.json"
TRANSCRIPTS_PATH = ROOT / "data" / "transcripts"

from db import export_labels_to_json

def export_labeled_scenes():
    """
    Load labels, merge with scene text, and save labeled scenes to JSON
    """
    # Load labels
    labels = pd.read_json(LABELS_PATH, dtype=str)
    
    # Load scene text
    scenes = [] 
    for folder in TRANSCRIPTS_PATH.iterdir():
        for transcript in folder.iterdir():
            with open(transcript, 'r') as file:
                
                # Load episode json file 
                episode = json.load(file)
                
                # For each scene in episode, add text and text from prev scene
                for scene_idx, scene in enumerate(episode["scenes"]):
                    if scene_idx == 0:
                        scene["prev_scene_text"] = ""
                    else:
                        scene["prev_scene_text"] = (
                            episode["scenes"][scene_idx-1]["text"]
                        )
                    
                    # Also add episode ID based on scene id
                    id_parts = scene["scene_id"].split("_")
                    scene["episode_id"] = f"{id_parts[0]}_{id_parts[1]}"
                    
                    scenes.append(scene)

    # Convert scenes to pandas df
    scenes_df = pd.DataFrame(scenes)
    
    # Merge scenes with labels
    labeled_scenes_df = labels.merge(
        scenes_df, on="scene_id", how="inner"
    )[["episode_id", "scene_id", "funny", "sad", "text", "prev_scene_text", "position"]]
    
    # Save labeled scenes to JSON
    labeled_scenes_df.to_json(OUTPUT_PATH, indent=2, orient="records")

if __name__ == "__main__":
    export_labels_to_json()
    export_labeled_scenes()
