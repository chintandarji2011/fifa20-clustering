"""
FIFA 20 Player Clustering - Flask web app
-----------------------------------------
Serves an interactive page where players are shown in their K-Means cluster.
Changing a player's skill sliders re-runs the saved model; if the predicted
cluster changes, the page animates the player riding a football into the new
cluster (with particle burst, shockwave ring and GOAL! flash).

Run:  python app.py   ->  http://127.0.0.1:5000
"""
import os
import urllib.request
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fifa20_player_clustering_pipeline.joblib")
DATA_PATH = os.path.join(BASE_DIR, "players_20.csv")
PHOTO_CACHE = os.path.join(BASE_DIR, "photo_cache")

# ---------------------------------------------------------------- model + data
artifact = joblib.load(MODEL_PATH)
PIPELINE = artifact["pipeline"]                 # imputer -> scaler -> KMeans
FEATURES = artifact["features"]                 # 34 skill columns, in order
CLUSTER_NAMES = artifact["cluster_names"]       # raw KMeans label -> name
CLUSTER_ORDER = artifact["cluster_order"]       # display order

KEEP = ["sofifa_id", "short_name", "long_name", "age", "nationality", "club",
        "overall", "potential", "player_positions"] + FEATURES
df = pd.read_csv(DATA_PATH, usecols=KEEP)
df["club"] = df["club"].fillna("Free agent")
df["cluster"] = pd.Series(PIPELINE.predict(df[FEATURES])).map(CLUSTER_NAMES).values
df = df.sort_values("overall", ascending=False).reset_index(drop=True)
df_by_id = df.set_index("sofifa_id")


def photo_url(sofifa_id: int) -> str:
    s = f"{int(sofifa_id):06d}"
    return f"https://cdn.sofifa.net/players/{s[:3]}/{s[3:]}/20_120.png"


def card(row) -> dict:
    return {
        "id": int(row["sofifa_id"]),
        "name": row["short_name"],
        "overall": int(row["overall"]),
        "position": row["player_positions"],
        "club": row["club"],
        "photo": photo_url(row["sofifa_id"]),
    }


def predict_cluster(skills: dict) -> str:
    """Predict a cluster name from a {feature: value} dict (missing -> median imputed)."""
    row = pd.DataFrame([{f: skills.get(f, np.nan) for f in FEATURES}], columns=FEATURES).astype(float)
    return CLUSTER_NAMES[int(PIPELINE.predict(row)[0])]


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/clusters")
def clusters():
    """?top=1..10 or ?top=all  -> players per cluster (best overall first)."""
    top = request.args.get("top", "5")
    out = []
    for name in CLUSTER_ORDER:
        sub = df[df["cluster"] == name]
        total = len(sub)
        if top != "all":
            sub = sub.head(max(1, min(int(top), 10)))
        out.append({"name": name, "total": total, "players": [card(r) for _, r in sub.iterrows()]})
    return jsonify({"clusters": out})


@app.route("/api/player/<int:pid>")
def player(pid):
    if pid not in df_by_id.index:
        return jsonify({"error": "player not found"}), 404
    r = df_by_id.loc[pid]
    return jsonify({
        "id": pid, "name": r["short_name"], "long_name": r["long_name"], "age": int(r["age"]),
        "nationality": r["nationality"], "club": r["club"], "overall": int(r["overall"]),
        "potential": int(r["potential"]), "position": r["player_positions"],
        "photo": photo_url(pid), "cluster": r["cluster"],
        "skills": {f: int(r[f]) if pd.notna(r[f]) else None for f in FEATURES},
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    skills = data.get("skills", {})
    clean = {}
    for f in FEATURES:
        v = skills.get(f)
        clean[f] = float(np.clip(float(v), 1, 99)) if v is not None else np.nan
    return jsonify({"cluster": predict_cluster(clean)})


@app.route("/api/meta")
def meta():
    return jsonify({"features": FEATURES, "clusters": CLUSTER_ORDER, "metadata": artifact.get("metadata", {})})


@app.route("/api/photo/<int:pid>")
def photo(pid):
    """Proxy player headshots through this server.

    cdn.sofifa.net blocks some direct browser/network requests, which made the
    page fall back to name initials. This endpoint downloads the headshot with
    browser-like headers, caches it on disk, and serves it to the page.
    If the CDN refuses, the endpoint returns 502 and the page falls back to
    the next source in its chain (direct CDN -> image proxy -> initials).
    """
    if pid not in df_by_id.index:
        abort(404)
    os.makedirs(PHOTO_CACHE, exist_ok=True)
    cached = os.path.join(PHOTO_CACHE, f"{pid}.png")
    if not os.path.exists(cached):
        url = photo_url(pid)
        req = urllib.request.Request(url, headers={
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            "Referer": "https://sofifa.com/",
            "Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8",
        })
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
        except Exception:
            abort(502)
        with open(cached, "wb") as f:
            f.write(data)
    return send_file(cached, mimetype="image/png", max_age=86400)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
