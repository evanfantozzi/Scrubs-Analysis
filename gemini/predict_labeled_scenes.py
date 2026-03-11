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
MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
ROOT = Path(os.environ.get("PROJECT_ROOT"))
LABELED_SCENES_PATH = ROOT / "data" / "labeled_scenes.json"
OUTPUT_DIR = ROOT / "data" / "gemini_predictions" / "labeled_scenes" 

def prompt(labeled_scene: dict, model: str) -> str:
    ''' 
    Prompt the inputted Gemini model to score the scene, returning text
    '''
    
    prompt_text = f"""
You are scoring scenes from the TV show Scrubs on how funny and sad they are on a 1 (least) to 5 (most) scale.

Example 1 — Scene: Jordan: It's Jack's first birthday. I want it to be special. I got a petting zoo for the kids. Cox: How about a Russian Roulette booth? We put bullets in ALL the chambers. That way everyone wins! J.D.: Will there be a piñata? Because I need to know if I should bring my piñata helmet. Jordan: Would you zip it, nerd? The only reason I invited you is because you have your own Spongebob Squarepants costume.
Funny: 4
Sad: 1

Example 2 — Scene: Dr. Cox: Time's up. Carla, would you do it for him, please? J.D.: Why are you telling her? Dr. Cox: Shut up and watch. Dr. Cox: Why does this GOMER got to try and die everyday during my lunch? J.D.: That's a little insensitive. J.D.'s narration: Mistake.
Funny: 2
Sad: 2

Now rate this scene.
Location in episode (0 = beginning, 1 = end): {round(labeled_scene["position"], 3)}
Previous scene: {labeled_scene["prev_scene_text"]}
Scene: {labeled_scene["text"]}

Respond in exactly this format:
Funny: <number>
Sad: <number>
"""
    response = CLIENT.models.generate_content(
        model=model,
        contents=prompt_text
    )
    return response.text

def parse_response(response: str) -> tuple[int, int]:
    '''
    Parse the response from the model to get the funny and sad scores
    '''
    if not response:
        print("Error: No response from model")
        return 0, 0
    
    # Split into funny and sad lines of text
    lines = response.splitlines()
    funny_line = lines[0].split(": ")
    sad_line = lines[1].split(": ")

    # Get funny and sad scores as integers
    funny = int(funny_line[1])
    sad = int(sad_line[1])
    return funny, sad

def main():
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load labeled scenes
    with open(LABELED_SCENES_PATH, 'r') as f:
        labeled_scenes = json.load(f)
    
    # Loop through different models
    for model in MODELS:
        
        # Loop through scenes and prompt the model to score the scene
        predictions = []
        for scene in labeled_scenes:
            
            # Convert labels to integers and position to float
            scene["funny"] = int(scene["funny"])
            scene["sad"] = int(scene["sad"])
            scene["position"] = float(scene["position"])
            
            # Print progress
            print(f"Predicting for scene {scene['scene_id']} with {model}")
            
            # Avoid hitting rate limit
            time.sleep(1)
            
            # Prompt LLM and parse predictions
            response = prompt(scene, model)
            predicted_funny, predicted_sad = parse_response(response)
            
            # Append to predictions list 
            predictions.append({
                "episode_id": scene["episode_id"],
                "scene_id": scene["scene_id"],
                "position": scene["position"],
                "prev_scene_text": scene["prev_scene_text"],
                "text": scene["text"],
                "true_funny": scene["funny"],
                "true_sad": scene["sad"],
                "predicted_funny": predicted_funny,
                "predicted_sad": predicted_sad,
            })   

        # Save predictions to JSON
        with open(OUTPUT_DIR / f"{model}.json", "w") as f:
            json.dump(predictions, f)
        
if __name__ == "__main__":
    main()