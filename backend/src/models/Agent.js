// models/Agent.js
import mongoose from "mongoose";

const agentSchema = new mongoose.Schema(
  {
    name:       { type: String, required: true },
    email:      { type: String, required: true, unique: true },
    password:   { type: String, required: true },
    department: { type: mongoose.Schema.Types.ObjectId, ref: "Department", required: true },
    district:   { type: String, required: true },        // 👈 new
    agentType:  { type: String, required: true },        // 👈 new (e.g. "Road Service")
    complaints: [{ type: mongoose.Schema.Types.ObjectId, ref: "Complaint" }],
  },
  { timestamps: true }
);

export default mongoose.model("Agent", agentSchema);