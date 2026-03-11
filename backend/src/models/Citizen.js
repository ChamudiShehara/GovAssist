// models/Citizen.js
import mongoose from "mongoose";

const citizenSchema = new mongoose.Schema(
  {
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    complaints: [{ type: mongoose.Schema.Types.ObjectId, ref: "Complaint" }],
  },
  { timestamps: true }
);

export default mongoose.model("Citizen", citizenSchema);