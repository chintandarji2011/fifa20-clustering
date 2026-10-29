# FIFA 20 Player Clustering – Flask App

Interactive web app for the K-Means player-segmentation model built in
`FIFA20_Players_Clustering_v2.ipynb`.

## Features
- One block per cluster (the saved model has **5 clusters**): Goalkeepers, Core Defenders,
  Two-Way Midfielders & Full-Backs, Established Attackers & Creators, Developing Attackers.
- Each block shows player photo, name, position, club and overall – scrollable.
- Dropdown: **Top 1 … Top 10** or **All players** per cluster.
- Click any player → a **details panel** below the blocks shows all 34 skills as sliders.
- Move the sliders → the model re-predicts live. When the cluster changes, the player
  **jumps** (arc animation) from the old cluster block into the new one.
- "Reset skills" restores the original ratings (and jumps back if needed).

## Project structure
```
fifa20_cluster_app/
├── app.py                        # Flask server + API
├── fifa20_cluster_model.joblib   # imputer -> scaler -> KMeans pipeline + metadata
├── players_20.csv                # FIFA 20 dataset
├── requirements.txt
├── README.md
└── templates/
    └── index.html                # UI (HTML + CSS + JS, no build step)
```

## Run locally
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## API
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/clusters?top=5` | Players per cluster (`top` = 1–10 or `all`) |
| GET | `/api/player/<sofifa_id>` | Player profile + 34 skill ratings |
| POST | `/api/predict` | Body `{"skills": {feature: value}}` → `{"cluster": name}` |
| GET | `/api/meta` | Feature list, cluster names, model metadata |

## Deploy (Render / Railway / Heroku)
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`

## Notes
- `scikit-learn` is pinned to **1.8.0** — the version the model was trained with.
- Player photos load from the public sofifa CDN; if one is unavailable, an initials avatar is shown.
- The `.joblib` file was converted from `fifa20_player_clustering_pipeline.pkl`; predictions are identical.
