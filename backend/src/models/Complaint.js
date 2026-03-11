// models/Complaint.js
import mongoose from "mongoose";

const complaintSchema = new mongoose.Schema(
  {
    citizen:     { type: mongoose.Schema.Types.ObjectId, ref: "Citizen", required: true },
    title:       { type: String, required: true, trim: true },
    name:        { type: String, required: true, trim: true },
    email:       { type: String, required: true, trim: true },
    description: { type: String, required: true, trim: true },
    phoneNumber: { type: String, required: true },
    address:     { type: String, required: true },
    district:    { type: String, required: true },
    department:  { type: mongoose.Schema.Types.ObjectId, ref: "Department", default: null },
    assignedAgent: { type: mongoose.Schema.Types.ObjectId, ref: "Agent", default: null }, // 👈 new
    status: {
      type: String,
      enum: ["PENDING", "IN_PROGRESS", "RESOLVED", "REJECTED"], // 👈 added REJECTED
      default: "PENDING",
    },
    priority: {
      type: String,
      enum: ["HIGH", "MEDIUM", "LOW"],
      default: "MEDIUM",
    },
    votes: [{ type: mongoose.Schema.Types.ObjectId, ref: "Citizen" }],
  },
  { timestamps: true }
);

export default mongoose.model("Complaint", complaintSchema);