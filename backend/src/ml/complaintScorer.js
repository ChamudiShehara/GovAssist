// backend/src/ml/complaintScorer.js
// JS fallback scorer — used when Python ML service is unavailable

const PRIORITY_SCORE = { HIGH: 3, MEDIUM: 2, LOW: 1 };

const extractFeatures = (complaint, maxVotes = 1, maxAgeDays = 30) => {
  const votes = complaint.votes?.length || 0;
  const priority = PRIORITY_SCORE[complaint.priority] || 2;
  const ageMs = Date.now() - new Date(complaint.createdAt).getTime();
  const ageDays = ageMs / (1000 * 60 * 60 * 24);

  const votesNorm    = Math.min(votes / Math.max(maxVotes, 1), 1);
  const priorityNorm = (priority - 1) / 2;
  const recencyNorm  = Math.max(0, 1 - ageDays / Math.max(maxAgeDays, 1));

  return [votesNorm, priorityNorm, recencyNorm];
};

const generateLabels = (featureMatrix) =>
  featureMatrix.map(([votes, priority, recency]) =>
    0.50 * priority + 0.35 * votes + 0.15 * recency
  );

const trainModel = (featureMatrix, labels, options = {}) => {
  const { lr = 0.1, epochs = 500 } = options;
  const n = featureMatrix.length;
  if (n === 0) return { weights: [0.35, 0.50, 0.15], bias: 0 };

  let weights = [0.33, 0.33, 0.33];
  let bias = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    let dw = [0, 0, 0], db = 0;
    for (let i = 0; i < n; i++) {
      const features = featureMatrix[i];
      const pred = features.reduce((s, f, j) => s + f * weights[j], bias);
      const err  = pred - labels[i];
      dw = dw.map((d, j) => d + (err * features[j]) / n);
      db += err / n;
    }
    weights = weights.map((w, j) => w - lr * dw[j]);
    bias -= lr * db;
  }
  return { weights, bias };
};

export const scoreAndSortComplaints = (complaints) => {
  if (!complaints || complaints.length === 0) return [];
  if (complaints.length === 1) return complaints;

  const maxVotes = Math.max(...complaints.map((c) => c.votes?.length || 0), 1);
  const maxAgeDays = Math.max(
    ...complaints.map((c) => {
      const ageMs = Date.now() - new Date(c.createdAt).getTime();
      return ageMs / (1000 * 60 * 60 * 24);
    }), 1
  );

  const featureMatrix = complaints.map((c) => extractFeatures(c, maxVotes, maxAgeDays));
  const labels        = generateLabels(featureMatrix);
  const { weights, bias } = trainModel(featureMatrix, labels);

  console.log(`[ML-JS Fallback] weights — votes: ${weights[0].toFixed(3)}, priority: ${weights[1].toFixed(3)}, recency: ${weights[2].toFixed(3)}`);

  const scored = complaints.map((complaint, i) => {
    const features = featureMatrix[i];
    const score = Math.max(0, features.reduce((s, f, j) => s + f * weights[j], bias));
    return {
      ...(complaint.toObject ? complaint.toObject() : complaint),
      _urgencyScore: score,
    };
  });

  return scored.sort((a, b) => b._urgencyScore - a._urgencyScore);
};