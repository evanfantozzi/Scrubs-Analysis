"""
Flask backend for the Scrubs scene labeling app.
Serves the single-page frontend (static/index.html) and API for scene list and labels.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder="static")

# -----------------------------------------------------------------------------
# Paths and in-memory scene state
# -----------------------------------------------------------------------------
load_dotenv()
ROOT = Path(os.environ.get("PROJECT_ROOT"))
TRANSCRIPTS_DIR = ROOT / "data" / "transcripts"
_scenes_by_id = None
_structure = None


def _title_from_wiki_source(source: str, stem: str) -> str:
    """Episode title from JSON source (e.g. 'My Lunch transcript') or filename stem (ep_20_My_Lunch_transcript)."""
    if source:
        t = source.replace(" transcript", "").strip()
        if t:
            return t

    m = re.match(r"ep_\d+_(.+)", stem, re.I)
    if m:
        return m.group(1).replace("_transcript", "").replace("_", " ").strip()
    return stem


def _resolve_scene(scene_id: str):
    """Return scene dict from in-memory JSON, or None."""
    if _scenes_by_id is not None:
        return _scenes_by_id.get(scene_id)
    return None


def _season_dir_match(name: str) -> re.Match | None:
    """Return match if name is 'Season N Transcripts' (case-insensitive)."""
    return re.match(r"Season (\d+) Transcripts", name, re.I)


def _parse_episode_file(jf: Path, data: dict, fallback_season: int) -> tuple[int, int, str, list[dict]]:
    """Return (season, episode, title, scenes) for one episode JSON. Scenes get scene_id set."""
    season = data.get("season")
    season = int(season) if season is not None else fallback_season
    episode = data.get("episode")
    if episode is None:
        ep_m = re.search(r"ep_(\d+)_", jf.stem, re.I)
        episode = int(ep_m.group(1)) if ep_m else 0
    else:
        episode = int(episode) if not isinstance(episode, dict) else 0
    title = _title_from_wiki_source(data.get("source") or "", jf.stem)
    scenes = data.get("scenes", [])
    for i, scene in enumerate(scenes, start=1):
        scene["scene_id"] = f"{season}_{episode:02d}_{i}"
    return season, episode, title, scenes


def _load_scenes():
    """Load scene JSONs from data/transcripts/Season N Transcripts/*.json and build scene lookup and structure."""
    global _scenes_by_id, _structure

    scenes_by_id = {}
    by_season = {}

    for season_dir in sorted(TRANSCRIPTS_DIR.iterdir(), key=lambda p: p.name):
        if not season_dir.is_dir():
            continue
        season_m = _season_dir_match(season_dir.name)
        if not season_m:
            continue
        fallback_season = int(season_m.group(1))

        for jf in sorted(season_dir.glob("*.json"), key=lambda f: f.stem):
            data = json.loads(jf.read_text(encoding="utf-8"))
            season, episode, title, scenes = _parse_episode_file(jf, data, fallback_season)
            for scene in scenes:
                scenes_by_id[scene["scene_id"]] = scene
            by_season.setdefault(season, []).append({
                "episode": episode,
                "title": title,
                "scene_ids": [s["scene_id"] for s in scenes],
            })

    for season in by_season:
        by_season[season].sort(key=lambda x: x["episode"])

    _scenes_by_id = scenes_by_id
    _structure = {
        "seasons": sorted(by_season.keys()),
        "by_season": by_season,
    }


# -----------------------------------------------------------------------------
# Static and structure
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/structure", methods=["GET"])
def api_structure():
    """Returns season/episode list and scene_ids for the labeling UI (from scrape_wiki/transcripts)."""
    _load_scenes()
    return jsonify(_structure)


# -----------------------------------------------------------------------------
# Scene and label endpoints
# -----------------------------------------------------------------------------
@app.route("/api/scenes/<scene_id>", methods=["GET"])
def api_scene(scene_id):
    _load_scenes()
    scene = _resolve_scene(scene_id)
    if not scene:
        return jsonify({"error": "Scene not found"}), 404
    return jsonify(scene)


@app.route("/api/scenes/<scene_id>/labels", methods=["GET"])
def api_scene_labels(scene_id):
    """Return labels for a scene (funny, sad, updated_at only; no identity)."""
    from db import get_conn, get_all_labels_for_scene
    conn = get_conn()
    try:
        labels = get_all_labels_for_scene(conn, scene_id)
        return jsonify({"labels": labels})
    finally:
        conn.close()


@app.route("/api/labels/<scene_id>", methods=["GET"])
def api_get_label(scene_id):
    from db import get_conn, get_label
    conn = get_conn()
    try:
        label = get_label(conn, scene_id)
        if label is None:
            return jsonify({"scene_id": scene_id, "labeled": False}), 200
        out = {"scene_id": scene_id, "funny": label["funny"], "sad": label["sad"], "labeled": True}
        if label.get("updated_at"):
            out["updated_at"] = label["updated_at"]
        return jsonify(out)
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# Save rating (funny 1–5, sad 1–5) for scene
# -----------------------------------------------------------------------------
@app.route("/api/labels", methods=["POST"])
def api_set_label():
    from db import get_conn, set_label
    data = request.get_json()
    if not data or "scene_id" not in data or "funny" not in data or "sad" not in data:
        return jsonify({"error": "scene_id, funny, sad required"}), 400
    scene_id = data["scene_id"]
    funny = int(data["funny"])
    sad = int(data["sad"])
    if not (1 <= funny <= 5 and 1 <= sad <= 5):
        return jsonify({"error": "funny and sad must be 1-5"}), 400
    conn = get_conn()
    try:
        set_label(conn, scene_id, funny, sad)
        return jsonify({"scene_id": scene_id, "funny": funny, "sad": sad})
    finally:
        conn.close()


_load_scenes()


if __name__ == "__main__":
    app.run(port=5001, debug=True)
