import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from dotenv import load_dotenv
from transformers import AutoModel, DebertaV2Tokenizer
load_dotenv()

# Highest performing DeBERTa model was DeBERTa-v3-base with SGD optimizer,
# no dropout, and using CLS token for encodings
MODEL = "microsoft/deberta-v3-base"
MODEL_RESULT_NAME = "microsoft_deberta_v3_base_SGD_without_dropout_cls_results"

# Config
ROOT = Path(os.environ.get("PROJECT_ROOT"))
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
MODEL_PATH = ROOT / "DeBERTa" / "models" / "microsoft_deberta_v3_base_SGD_without_dropout_best_model.pt"
DEBERTA_PREDICTIONS_DIR = ROOT / "data" / "DeBERTa_predictions"
LABELED_SCENES_PREDICTIONS_PATH = DEBERTA_PREDICTIONS_DIR / "labeled_scenes" / f"{MODEL_RESULT_NAME}.json"
OUTPUT_DIR = DEBERTA_PREDICTIONS_DIR / "all_scenes_with_transcripts" 


class ScrubsDeBERTa(nn.Module):
    """
    Simplified version from the fine tuning, notebook specified for highest
    performing model
    """
    def __init__(self):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(MODEL)
        hidden_dim = self.pretrained_model.config.hidden_size
        self.funny_head = nn.Linear(hidden_dim * 3 + 1, 5)
        self.sad_head = nn.Linear(hidden_dim * 3 + 1, 5)

    def forward(
        self,
        prev_input_ids,
        prev_attention_mask,
        curr_input_ids,
        curr_attention_mask,
        positions,
    ):
        prev_outputs = self.pretrained_model(
            prev_input_ids,
            attention_mask=prev_attention_mask,
        )
        curr_outputs = self.pretrained_model(
            curr_input_ids,
            attention_mask=curr_attention_mask,
        )

        # Get encoding from CLS token, create combined embedding of 
        # previous scene text, current scene text, difference between the two,
        # and position of scene in ep
        prev_encoding = prev_outputs.last_hidden_state[:, 0, :]
        curr_encoding = curr_outputs.last_hidden_state[:, 0, :]
        diff = curr_encoding - prev_encoding
        combined = torch.cat([prev_encoding, curr_encoding, diff, positions], dim=1)

        # Calculate funny and sad logits, returning them
        funny_logits = self.funny_head(combined)
        sad_logits = self.sad_head(combined)
        return funny_logits, sad_logits


def load_existing_predictions() -> dict[str, dict]:
    """
    Load existing predictions for labeled scenes
    """
    with open(LABELED_SCENES_PREDICTIONS_PATH, "r") as f:
        predictions = json.load(f)
    return {prediction["scene_id"]: prediction for prediction in predictions}


def load_scenes() -> dict[str, dict]:
    """
    Load all scenes from transcripts
    """
    scenes = {}

    # Loop through seasons
    for season_dir in sorted(TRANSCRIPTS_DIR.iterdir()):
        
        # Loop through episodes
        for episode_transcript in sorted(season_dir.iterdir()):
            
            # Load episode transcript
            with open(episode_transcript, "r") as f:
                episode = json.load(f)

            # Loop through scenes
            for scene_idx, scene in enumerate(episode["scenes"]):
                
                # Add previous scene text to scene
                if scene_idx == 0:
                    scene["prev_scene_text"] = ""
                else:
                    scene["prev_scene_text"] = episode["scenes"][scene_idx - 1]["text"]

                # Calculate episode and scene IDs
                id_parts = scene["scene_id"].split("_")
                scene["episode_id"] = f"{id_parts[0]}_{id_parts[1]}"
                scenes[scene["scene_id"]] = scene

    return scenes


def load_model():
    """
    Load model for use with inference, along with the tokenizer
    """
    # Load tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL)
    
    # Load model
    model = ScrubsDeBERTa()
    
    # Load checkpoint and set model in eval mode
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    
    
    return model, tokenizer


def predict_scene(model, tokenizer, scene: dict) -> tuple[int, int]:
    """
    Predict scene labels using loaded model

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        scene: Scene to predict labels for
        
    Returns:
        Predicted funny and sad labels
    """
    
    # Tokenize text from previous scene and current scene
    prev_tokens = tokenizer(
        [scene["prev_scene_text"]],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    curr_tokens = tokenizer(
        [scene["text"]],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Get position of scene
    positions = torch.tensor([[float(scene["position"])]], dtype=torch.float32)

    # Calculate funny and sad logits 
    with torch.no_grad():
        funny_logits, sad_logits = model(
            prev_input_ids=prev_tokens["input_ids"],
            prev_attention_mask=prev_tokens["attention_mask"],
            curr_input_ids=curr_tokens["input_ids"],
            curr_attention_mask=curr_tokens["attention_mask"],
            positions=positions,
        )

    # Get argmax of logits, adjust to 1-5 scale and return predictions
    predicted_funny = funny_logits.argmax(dim=-1).cpu().item() + 1
    predicted_sad = sad_logits.argmax(dim=-1).cpu().item() + 1
    return predicted_funny, predicted_sad


def main():
    # Make output directory if doesn't exist yet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Get existing predictions and scenes
    scene_predictions = load_existing_predictions()
    scenes = load_scenes()

    # Loop through all scenes
    for scene_id, scene in scenes.items():
        
        # If a scene already has predictions, skip it
        if scene_id in scene_predictions:
            print(f"Scene {scene_id} already predicted with {MODEL_RESULT_NAME}")
            continue
        
        # Else, perform inference on the scene
        print(f"Predicting scene {scene_id} with {MODEL_RESULT_NAME}")
        predicted_funny, predicted_sad = predict_scene(model, tokenizer, scene)


        # Add the scene and its predictions to the dictionary for output 
        scene_predictions[scene_id] = {
            "episode_id": scene["episode_id"],
            "scene_id": scene["scene_id"],
            "position": scene["position"],
            "prev_scene_text": scene["prev_scene_text"],
            "text": scene["text"],
            "predicted_funny": predicted_funny,
            "predicted_sad": predicted_sad,
        }

    # Save predictions to JSON
    with open(OUTPUT_DIR / f"{MODEL_RESULT_NAME}.json", "w") as f:
        json.dump(list(scene_predictions.values()), f, indent=2)


if __name__ == "__main__":
    main()
