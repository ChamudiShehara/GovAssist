# backend/src/ml/python/model.py
# Complaint urgency scoring using Random Forest Regressor
# Trains from dataset.csv — real labeled data

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "complaint_model.pkl")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")

# ── Feature Maps ───────────────────────────────────────────────────────────────

PRIORITY_MAP = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
STATUS_MAP   = {"PENDING": 3, "IN_PROGRESS": 2, "RESOLVED": 1}

# ── Feature Extraction ─────────────────────────────────────────────────────────

def extract_features(complaints: list[dict]) -> pd.DataFrame:
    """
    Convert raw complaint dicts (from MongoDB via Node) into feature DataFrame.
    """
    rows = []
    for c in complaints:
        votes    = len(c.get("votes", []))
        priority = PRIORITY_MAP.get(c.get("priority", "MEDIUM"), 2)
        status   = STATUS_MAP.get(c.get("status", "PENDING"), 3)

        created_at = c.get("createdAt", "")
        try:
            from datetime import datetime, timezone
            dt  = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            recency_days = max(0, (now - dt).total_seconds() / 86400)
        except Exception:
            recency_days = 0.0

        rows.append({
            "votes_count":  votes,
            "priority_num": priority,
            "recency_days": recency_days,
            "status_num":   status,
        })

    return pd.DataFrame(rows)


def extract_features_from_csv(df: pd.DataFrame) -> tuple:
    """
    Convert CSV rows into feature matrix X and label vector y.
    CSV columns: title, votes, priority, recency_days, status, urgency_score
    """
    X = pd.DataFrame({
        "votes_count":  df["votes"].astype(float),
        "priority_num": df["priority"].map(PRIORITY_MAP).fillna(2).astype(float),
        "recency_days": df["recency_days"].astype(float),
        "status_num":   df["status"].map(STATUS_MAP).fillna(3).astype(float),
    })
    y = df["urgency_score"].astype(float).values
    return X.values, y


# ── Model Pipeline ─────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ── Load CSV Dataset ───────────────────────────────────────────────────────────

def load_csv_dataset() -> tuple:
    """
    Load dataset.csv and return (X, y).
    Raises FileNotFoundError if CSV not found.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"dataset.csv not found at {DATASET_PATH}\n"
            "Please add your dataset CSV file to backend/src/ml/python/dataset.csv"
        )

    df = pd.read_csv(DATASET_PATH)

    # Validate required columns
    required = {"votes", "priority", "recency_days", "status", "urgency_score"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset.csv is missing columns: {missing}")

    print(f"[ML] Loaded dataset.csv — {len(df)} rows")
    print(f"[ML] Priority distribution:\n{df['priority'].value_counts().to_string()}")
    print(f"[ML] Urgency score range: {df['urgency_score'].min():.2f} – {df['urgency_score'].max():.2f}")

    return extract_features_from_csv(df)


# ── Synthetic Fallback Labels ──────────────────────────────────────────────────

def compute_urgency_label(df: pd.DataFrame) -> np.ndarray:
    """
    Fallback: generate labels from domain-knowledge formula.
    Only used when dataset.csv is NOT available.
    """
    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - mn) / (mx - mn)

    priority_norm = normalize(df["priority_num"])
    votes_norm    = normalize(df["votes_count"])
    recency_norm  = 1 - normalize(df["recency_days"])
    status_norm   = normalize(df["status_num"])

    return (
        0.40 * priority_norm +
        0.30 * votes_norm    +
        0.20 * recency_norm  +
        0.10 * status_norm
    ).values


# ── Train ──────────────────────────────────────────────────────────────────────

def train(complaints: list[dict] = None) -> Pipeline:
    """
    Train the model.
    - If dataset.csv exists → train from CSV (real labeled data)
    - If dataset.csv missing → fall back to synthetic labels from complaints batch
    """
    pipeline = build_pipeline()

    try:
        # PRIMARY: Train from your real CSV dataset
        X, y = load_csv_dataset()
        print(f"[ML] Training from dataset.csv ({len(y)} samples)...")

        # Split for validation
        if len(y) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
            print(f"[ML] ✅ Trained on CSV — R² score: {score:.3f} ({len(X_train)} train / {len(X_test)} test samples)")
        else:
            pipeline.fit(X, y)
            print(f"[ML] ✅ Trained on CSV — {len(y)} samples (too few for train/test split)")

    except FileNotFoundError as e:
        # FALLBACK: dataset.csv not found — use synthetic labels from live complaints
        print(f"[ML] ⚠️  {e}")
        print("[ML] Falling back to synthetic label generation from live complaints...")

        if not complaints or len(complaints) < 2:
            print("[ML] Not enough complaints for fallback training — using default weights")
            # Return untrained pipeline with default weights baked in
            dummy_X = np.array([[0,1,0,1],[1,3,0,3],[0,2,5,2],[1,3,1,3]])
            dummy_y = np.array([0.1, 0.9, 0.4, 0.85])
            pipeline.fit(dummy_X, dummy_y)
            return pipeline

        df     = extract_features(complaints)
        labels = compute_urgency_label(df)
        X      = df[["votes_count", "priority_num", "recency_days", "status_num"]].values
        pipeline.fit(X, labels)
        print(f"[ML] ✅ Trained on {len(complaints)} synthetic samples (add dataset.csv for real training)")

    # Log feature importances
    importances = dict(zip(
        ["votes", "priority", "recency", "status"],
        pipeline["model"].feature_importances_.round(3).tolist()
    ))
    print(f"[ML] Feature importances: {importances}")

    # Save trained model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[ML] Model saved to {MODEL_PATH}")

    return pipeline


# ── Load or Train ──────────────────────────────────────────────────────────────

def load_or_train(complaints: list[dict] = None) -> Pipeline:
    """
    Load saved model if exists.
    Always retrains if dataset.csv is present and model is stale or missing.
    """
    csv_exists   = os.path.exists(DATASET_PATH)
    model_exists = os.path.exists(MODEL_PATH)

    if csv_exists:
        # Always retrain when CSV is available to pick up any new rows
        csv_mtime   = os.path.getmtime(DATASET_PATH)
        model_mtime = os.path.getmtime(MODEL_PATH) if model_exists else 0

        if not model_exists or csv_mtime > model_mtime:
            print("[ML] dataset.csv is newer than saved model — retraining...")
            return train(complaints)
        else:
            print(f"[ML] Loading saved model from {MODEL_PATH}")
            return joblib.load(MODEL_PATH)
    else:
        # No CSV — use saved model or train from live complaints
        if model_exists:
            print(f"[ML] Loading saved model (no dataset.csv found)")
            return joblib.load(MODEL_PATH)
        return train(complaints)


# ── Predict ────────────────────────────────────────────────────────────────────

def predict_scores(pipeline: Pipeline, complaints: list[dict]) -> list[float]:
    """
    Predict urgency scores (0.0–1.0) for a list of complaint dicts.
    """
    df     = extract_features(complaints)
    X      = df[["votes_count", "priority_num", "recency_days", "status_num"]].values
    scores = pipeline.predict(X)
    return np.clip(scores, 0, 1).tolist()