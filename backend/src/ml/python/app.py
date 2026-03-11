# backend/src/ml/python/app.py
# Flask microservice — exposes ML scoring to Node backend via HTTP

from flask import Flask, request, jsonify
from flask_cors import CORS
from model import load_or_train, predict_scores, train
import traceback

app = Flask(__name__)
CORS(app)

# ── Health Check ───────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "complaint-ml-scorer"})


# ── Score & Sort Endpoint ──────────────────────────────────────────────────────

@app.route("/score", methods=["POST"])
def score():
    """
    POST /score
    Body: { "complaints": [ { _id, title, votes, priority, status, createdAt, ... } ] }

    Response: { "scored": [ { ...complaint, _urgencyScore, _rank } ] }
    Complaints are returned sorted by urgency score descending.
    """
    try:
        data       = request.get_json()
        complaints = data.get("complaints", [])

        if not complaints:
            return jsonify({"error": "No complaints provided"}), 400

        if len(complaints) == 1:
            # Single complaint — score it directly without training
            single = complaints[0]
            scored = [{**single, "_urgencyScore": 0.5, "_rank": 1}]
            return jsonify({"scored": scored, "model_info": {"note": "single complaint, default score"}})

        # Train (or load) model on this batch
        pipeline = load_or_train(complaints)

        # Predict urgency scores
        scores = predict_scores(pipeline, complaints)

        # Attach scores and sort
        scored = []
        for complaint, score in zip(complaints, scores):
            scored.append({**complaint, "_urgencyScore": round(float(score), 4)})

        scored.sort(key=lambda c: c["_urgencyScore"], reverse=True)

        # Add rank after sorting
        for i, c in enumerate(scored):
            c["_rank"] = i + 1

        # Feature importances for logging
        importances = dict(zip(
            ["votes", "priority", "recency", "status"],
            pipeline["model"].feature_importances_.round(3).tolist()
        ))

        print(f"[ML] Scored {len(scored)} complaints. Feature importances: {importances}")

        return jsonify({
            "scored": scored,
            "model_info": {
                "n_complaints": len(scored),
                "feature_importances": importances,
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Manual Retrain Endpoint ────────────────────────────────────────────────────

@app.route("/train", methods=["POST"])
def retrain():
    """
    POST /train
    Body: { "complaints": [...] }
    Forces a fresh model retrain on the provided data.
    """
    try:
        data       = request.get_json()
        complaints = data.get("complaints", [])

        if len(complaints) < 2:
            return jsonify({"error": "Need at least 2 complaints to train"}), 400

        pipeline = train(complaints)
        importances = dict(zip(
            ["votes", "priority", "recency", "status"],
            pipeline["model"].feature_importances_.round(3).tolist()
        ))

        return jsonify({
            "message": f"Model retrained on {len(complaints)} complaints",
            "feature_importances": importances,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[ML Service] Starting on http://localhost:5001")
    app.run(port=5001, debug=False)