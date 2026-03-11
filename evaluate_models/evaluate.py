import csv
import json
import os
from pathlib import Path
from itertools import chain
import altair as alt
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    precision_recall_fscore_support, 
    accuracy_score,
    recall_score
)

load_dotenv()

ROOT = Path(os.environ.get("PROJECT_ROOT"))
OUTPUT_DIR = ROOT / "data" / "model_evaluation"
CHARTS_DIR = OUTPUT_DIR / "charts"
EXAMPLES_DIR = OUTPUT_DIR / "examples"
GEMINI_PREDICTIONS_DIR = ROOT / "data" / "gemini_predictions" / "labeled_scenes" 
DEBERTA_PREDICTIONS_DIR = ROOT / "data" / "DeBERTa_predictions" / "labeled_scenes"
RESULTS_ORDER = ["Classified Correctly", "Classified Too High", "Classified Too Low", "Not Classified"]
LABELS = [1, 2, 3, 4, 5]
# Source for colors: https://colorhunt.co/palette/5a9cb5face68faac68fa6868
RESULTS_COLORS = ["#5A9CB5", "#FACE68", "#FAAC68", "#FA6868"]

def gen_breakdown_dict() -> dict:
    """
    Create a dictionary with breakdown in each category of correct/incorrect pred
    """
    
    return {
        "funny": {
            label: {
                "Classified Too High": [], 
                "Classified Too Low": [],
                "Classified Correctly": [],
                "Not Classified": [],
            }  
            for label in range(1, 6)
        },
        
        "sad": {
            label: {
                "Classified Too High": [], 
                "Classified Too Low": [],
                "Classified Correctly": [],
                "Not Classified": [],
            }  
            for label in range(1, 6)
        },
    }
    

def get_model_breakdown(scenes: dict) -> dict:
    
    # Initiate metrics dict to track how well the model did
    breakdown = gen_breakdown_dict()
    
    # Loop through scenes, adding to breakdown
    for scene in scenes:
        for funny_or_sad in ["funny", "sad"]:
            
            # Get result of prediction (correct, too high/low, not classified)
            label = scene[f"true_{funny_or_sad}"]
            prediction = scene[f"predicted_{funny_or_sad}"]
            if label == prediction:
                result = "Classified Correctly"
            elif prediction == 0:
                result = "Not Classified"
            elif prediction > label:
                result = "Classified Too High"
            else:
                result = "Classified Too Low"
            
            # Add to [funny/sad][1-5][correct/not classified/incorrect] list
            breakdown[funny_or_sad][label][result].append(scene)

    return breakdown


def plot_model(breakdown, model_name):
    """
    Saves a bar chart of a model's prediction accuracy breakdown
    Args:
        breakdown: dict of breakdowns per category and label
        model_name: name of the model to plot
    """
    # Build a list where each elem is one breakdown combination and num scenes in it
    long_rows = []
    for funny_or_sad in ["Funny", "Sad"]:
        for label in range(1, 6):
            for result in RESULTS_ORDER:
                long_rows.append({
                    "funny_or_sad": funny_or_sad,
                    "label": label,
                    "result": result,
                    "count": len(breakdown[funny_or_sad.lower()][label][result]),
                })
    chart_df = pd.DataFrame(long_rows)

    # Use a smaller y-axis range for DeBERTa models
    y_max = 80 if "deberta" in model_name.lower() else 300

    # Match scale with colors
    color_scale = alt.Scale(domain=RESULTS_ORDER, range=RESULTS_COLORS)

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        # X = true label 1–5, Y = number of scenes, 
        # bar color = [correct / too high / too low / not classified]
        .encode(
            x=alt.X("label:O", title="True label", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count:Q", title="Count", scale=alt.Scale(domain=[0, y_max])),
            color=alt.Color(
                "result:N",
                scale=color_scale,
                legend=alt.Legend(title=None),
            ),
            order=alt.Order("result:N", sort="ascending"),
        )
        
        # Chart size 
        .properties(width=220, height=220)
        
        # Two panels side by side: "Funny" and "Sad" 
        .facet(
            column=alt.Column(
                "funny_or_sad:N",
                header=alt.Header(title=None, labelFontSize=12, labelFontWeight="bold"),
            ),
        )

        # Main title
        .properties(title=f"Predictions with model: {model_name.replace('_', ' ')}")
        .configure_title(anchor="middle")
    )

    # Save chart
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(CHARTS_DIR / f"{model_name}.svg")

def get_model_summary_rows(scenes: list[dict], model_name: str) -> list[dict]:
    """
    Return metrics for funny and sad labels, for 1-5 labels, and overall 
    """
    summary_rows = []

    # Loop through funny and sad labels
    for funny_or_sad in ["funny", "sad"]:

        # Get labels and predictions
        labels = [scene[f"true_{funny_or_sad}"] for scene in scenes]
        predictions = [scene[f"predicted_{funny_or_sad}"] for scene in scenes]

        # Add overall metrics
        summary_rows.append({
            "model": model_name,
            "funny_or_sad": funny_or_sad,
            "true_label": "overall",
            
            # Add accuracy score
            "accuracy": round(accuracy_score(labels, predictions), 4),

            # Add precision score
            "precision": round(
                precision_score(
                    labels, 
                    predictions, 
                    labels=LABELS, 
                    average="macro", 
                    zero_division=0
                ),
                4,
            ),

            # Add recall score
            "recall": round(
                recall_score(
                    labels,
                    predictions,
                    labels=LABELS,
                    average="macro",
                    zero_division=0
                ),
                4,
            ),
            
            # Add f1 score
            "f1": round(
                f1_score(
                    labels, 
                    predictions, 
                    labels=LABELS, 
                    average="macro", 
                    zero_division=0),
                4,
            ),
        })

        # Loop through each label and calculate metrics per 1-5 label
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            labels=LABELS,
            zero_division=0
        )

        for label_index, label in enumerate(LABELS):
            summary_rows.append({
                "model": model_name,
                "funny_or_sad": funny_or_sad,
                "true_label": label,
                
                # Add accuracy score per 1-5 label value
                "accuracy": round(
                    accuracy_score(
                        [label_val == label for label_val in labels],
                        [predict_val == label for predict_val in predictions]
                    ),
                    4
                ),

                # Add precision score per 1-5 label
                "precision": round(precision[label_index], 4),

                # Add recall score per 1-5 label
                "recall": round(recall[label_index], 4),

                # Add f1 score per 1-5 label
                "f1": round(f1[label_index], 4),
            })

    return summary_rows

def export_breakdown_jsons(breakdown: dict, model_name: str) -> None:
    """
    Write one JSON per (category, label, result) with the scene list.
    """
    for funny_or_sad in ["funny", "sad"]:
        for label in LABELS:

            # Loop through results and write JSON files with examples
            for result in RESULTS_ORDER:
                
                # Get scenes to output
                scenes = breakdown[funny_or_sad][label][result]
                result_safe = result.replace(" ", "_").replace("(", "").replace(")", "")
                
                # Create examples directory if it doesn't exist
                os.makedirs(EXAMPLES_DIR / model_name, exist_ok=True)

                # Write JSON file
                with open(
                    EXAMPLES_DIR / model_name / f"{funny_or_sad}_{label}_{result_safe}.json", "w"
                ) as f:
                    json.dump(scenes, f, indent=2)


def main():
    """
    Main function to evaluate models
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_rows = []

    # Loop through each LLM model and DeBERTa model
    for model in sorted(chain(
        GEMINI_PREDICTIONS_DIR.iterdir(), DEBERTA_PREDICTIONS_DIR.iterdir())
        ):

        # Load predictions from JSON file
        with open(model) as f:
            predictions = json.load(f)

        # Get breakdown of model predictions
        breakdown = get_model_breakdown(predictions)

        # Plot model accuracy breakdown
        plot_model(breakdown, model.stem)
        
        # Export summary metrics and JSONs with examples
        summary_rows.extend(get_model_summary_rows(predictions, model.stem))
        export_breakdown_jsons(breakdown, model.stem)

    # Write summary CSV for all models
    with open(OUTPUT_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "funny_or_sad",
                "true_label",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()