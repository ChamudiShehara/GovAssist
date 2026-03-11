// backend/src/rag/buildIndex.js
// Run from the BACKEND root: node src/rag/buildIndex.js
// Re-run whenever you update knowledge.json

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import axios from "axios";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const knowledgePath = path.join(__dirname, "knowledge.json");
const indexPath = path.join(__dirname, "index.json");

const OLLAMA_URL = "http://localhost:11434/api/embeddings";
const EMBED_MODEL = "nomic-embed-text";

// Inline embedding so buildIndex doesn't depend on embedder.js module resolution
const getEmbedding = async (text) => {
  const response = await axios.post(OLLAMA_URL, {
    model: EMBED_MODEL,
    prompt: text,
  });
  return response.data.embedding;
};

const buildIndex = async () => {
  console.log("📚 Loading knowledge base...");
  const knowledge = JSON.parse(fs.readFileSync(knowledgePath, "utf-8"));
  const index = [];

  for (const entry of knowledge) {
    console.log(`\n🔄 Embedding department: ${entry.department}`);

    const documents = [
      `Department: ${entry.department}. ${entry.description}`,
      ...entry.complaintExamples.map((ex) => `${entry.department} complaint: ${ex}`),
      `${entry.department} handles issues related to: ${entry.keywords.join(", ")}`,
      ...(entry.synonyms ? [`${entry.department} also known as: ${entry.synonyms.join(", ")}`] : []),
    ];

    const chunks = [];
    for (const doc of documents) {
      try {
        const embedding = await getEmbedding(doc);
        chunks.push({ text: doc, embedding });
        process.stdout.write(".");
      } catch (err) {
        console.error(`\n⚠️  Failed to embed: "${doc.slice(0, 60)}..." — ${err.message}`);
      }
    }

    index.push({
      department: entry.department,
      description: entry.description,
      keywords: entry.keywords,
      synonyms: entry.synonyms || [],
      complaintExamples: entry.complaintExamples,
      chunks,
    });

    console.log(`\n✅ ${entry.department}: ${chunks.length} chunks embedded`);
  }

  fs.writeFileSync(indexPath, JSON.stringify(index, null, 2));
  console.log(`\n🎉 Index saved to ${indexPath}`);
  console.log(`📊 Total departments: ${index.length}`);
  console.log(`📊 Total chunks: ${index.reduce((s, e) => s + e.chunks.length, 0)}`);
};

buildIndex().catch((err) => {
  console.error("❌ Build failed:", err.message);
  process.exit(1);
});