// backend/src/ml/mlService.js
// Node.js client for the Python ML microservice
// Falls back to JS scorer if Python service is unavailable

import axios from "axios";
import { scoreAndSortComplaints } from "./complaintScorer.js"; // JS fallback

const ML_SERVICE_URL = "http://localhost:5001";
const TIMEOUT_MS = 10000; // 10s timeout

/**
 * Score and sort complaints using Python ML microservice.
 * Falls back to JS linear regression if Python service is down.
 */
export const mlScoreAndSort = async (complaints) => {
  if (!complaints || complaints.length === 0) return [];
  if (complaints.length === 1) return complaints;

  try {
    // Serialize complaints to plain objects for JSON transport
    const plain = complaints.map((c) =>
      c.toObject ? c.toObject() : c
    );

    const response = await axios.post(
      `${ML_SERVICE_URL}/score`,
      { complaints: plain },
      { timeout: TIMEOUT_MS }
    );

    const { scored, model_info } = response.data;

    if (model_info?.feature_importances) {
      console.log("[ML] Python model feature importances:", model_info.feature_importances);
    }

    return scored;

  } catch (err) {
    // Python service unavailable — fall back to JS scorer
    console.warn("[ML] Python service unavailable, using JS fallback:", err.message);
    return scoreAndSortComplaints(complaints);
  }
};

/**
 * Check if the Python ML service is running
 */
export const checkMlService = async () => {
  try {
    const res = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 3000 });
    return res.data?.status === "ok";
  } catch {
    return false;
  }
};