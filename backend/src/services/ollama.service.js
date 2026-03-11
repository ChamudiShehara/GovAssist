// backend/src/services/ollama.service.js
import axios from "axios";
import { buildRagContext } from "../rag/retriever.js"; // ../ goes up from services/ to src/, then into rag/

const OLLAMA_URL = "http://localhost:11434/api/generate";
const OLLAMA_MODEL = "llama3";

export const getDepartmentSuggestion = async (complaintTitle) => {
  try {
    console.log(`[RAG] Retrieving context for: "${complaintTitle}"`);
    const { context, topDepts } = await buildRagContext(complaintTitle);

    // High confidence — skip LLM entirely
    if (topDepts[0]?.hybridScore > 0.85) {
      console.log(`[RAG] High confidence: ${topDepts[0].department} (${topDepts[0].hybridScore.toFixed(3)})`);
      return { department: topDepts[0].department };
    }

    // Low confidence — guide LLM with RAG context
    const prompt = `You are a government complaint routing assistant for Sri Lanka.

Based on the retrieved knowledge below, determine the single best department to handle this complaint.

RETRIEVED KNOWLEDGE:
${context}

COMPLAINT TITLE: "${complaintTitle}"

Rules:
- You MUST choose from ONLY these options: ${topDepts.map((d) => `"${d.department}"`).join(", ")}
- Respond with ONLY the department name — no explanation, no punctuation, nothing else

Department name:`;

    const response = await axios.post(OLLAMA_URL, {
      model: OLLAMA_MODEL,
      prompt,
      stream: false,
    });

    const rawText = response.data?.response?.trim();
    if (!rawText) return { department: topDepts[0]?.department || null };

    const llmSuggestion = rawText.replace(/['"\.]/g, "").trim();
    console.log(`[Ollama] LLM suggested: "${llmSuggestion}"`);

    // Validate LLM response against RAG results (prevent hallucination)
    const llmLower = llmSuggestion.toLowerCase();
    const validatedMatch = topDepts.find(
      (d) =>
        d.department.toLowerCase() === llmLower ||
        d.department.toLowerCase().includes(llmLower) ||
        llmLower.includes(d.department.toLowerCase())
    );

    if (validatedMatch) {
      console.log(`[Ollama] Validated: "${validatedMatch.department}"`);
      return { department: validatedMatch.department };
    }

    console.log(`[Ollama] Falling back to RAG top result: "${topDepts[0].department}"`);
    return { department: topDepts[0].department };

  } catch (error) {
    console.error("[getDepartmentSuggestion] Error:", error?.response?.data || error.message);
    throw error;
  }
};