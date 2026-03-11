// rag/embedder.js
// Uses your already-installed nomic-embed-text model for embeddings
import axios from "axios";

const OLLAMA_URL = "http://localhost:11434/api/embeddings";
const EMBED_MODEL = "nomic-embed-text"; // You already have this installed

/**
 * Get embedding vector for a single text string
 */
export const getEmbedding = async (text) => {
  const response = await axios.post(OLLAMA_URL, {
    model: EMBED_MODEL,
    prompt: text,
  });
  return response.data.embedding; // Returns float[]
};

/**
 * Cosine similarity between two vectors
 */
export const cosineSimilarity = (vecA, vecB) => {
  const dot = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  if (magA === 0 || magB === 0) return 0;
  return dot / (magA * magB);
};