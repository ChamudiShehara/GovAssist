// backend/src/controllers/complaint.controller.js
import Complaint from "../models/Complaint.js";
import Citizen from "../models/Citizen.js";
import Department from "../models/Department.js";
import { getDepartmentSuggestion } from "../services/ollama.service.js";
import { mlScoreAndSort } from "../ml/mlService.js"; // Python ML service client

const matchDepartment = (departments, suggestedName) => {
  if (!suggestedName) return null;
  const normalize = (str) =>
    str.toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ").trim();
  const stripSpaces = (str) => normalize(str).replace(/\s/g, "");
  const suggested = normalize(suggestedName);
  const suggestedNoSpace = stripSpaces(suggestedName);

  let match = departments.find((d) => normalize(d.name) === suggested);
  if (match) return match;
  match = departments.find((d) => stripSpaces(d.name) === suggestedNoSpace);
  if (match) return match;
  match = departments.find((d) => suggested.includes(normalize(d.name)));
  if (match) return match;
  match = departments.find((d) => normalize(d.name).includes(suggested));
  if (match) return match;

  const suggestedWords = suggested.split(" ").filter((w) => w.length > 2);
  let bestMatch = null, bestScore = 0;
  for (const dept of departments) {
    const deptWords = normalize(dept.name).split(" ").filter((w) => w.length > 2);
    const shared = suggestedWords.filter((w) =>
      deptWords.some((dw) => dw.includes(w) || w.includes(dw))
    ).length;
    if (shared > bestScore) { bestScore = shared; bestMatch = dept; }
  }
  if (bestMatch && bestScore > 0) return bestMatch;
  return departments[0];
};

export const addComplaint = async (req, res) => {
  const { title, description, phoneNumber, address, district, department, priority } = req.body;
  try {
    if (!req.user) return res.status(401).json({ error: "Citizen not authenticated" });
    const citizen = req.user;
    let departmentId = department || null;
    if (!departmentId) {
      try {
        const allDepartments = await Department.find({});
        const suggestion = await getDepartmentSuggestion(title);
        const matched = matchDepartment(allDepartments, suggestion?.department);
        if (matched) departmentId = matched._id;
      } catch (err) {
        console.warn("RAG fallback failed during addComplaint:", err.message);
      }
    }
    const newComplaint = new Complaint({
      citizen: citizen._id,
      name: citizen.name,
      email: citizen.email,
      title, description, phoneNumber, address, district,
      department: departmentId,
      priority,
    });
    await newComplaint.save();
    res.status(201).json({ message: "Complaint submitted successfully", complaint: newComplaint });
  } catch (error) {
    console.error("Error adding complaint:", error);
    res.status(500).json({ error: "Server error" });
  }
};

export const deleteComplaint = async (req, res) => {
  try {
    const complaint = await Complaint.findByIdAndDelete(req.params.id);
    if (!complaint) return res.status(404).json({ error: "Complaint not found" });
    res.json({ message: "Complaint deleted successfully" });
  } catch (error) {
    res.status(500).json({ error: "Server error" });
  }
};

export const updateComplaintStatus = async (req, res) => {
  try {
    const { status } = req.body;
    const complaint = await Complaint.findByIdAndUpdate(req.params.id, { status }, { new: true });
    if (!complaint) return res.status(404).json({ error: "Complaint not found" });
    res.json({ message: "Complaint status updated", complaint });
  } catch (error) {
    res.status(500).json({ error: "Server error" });
  }
};

export const getComplaintsByCitizen = async (req, res) => {
  try {
    const citizenId = req.user.id;
    const complaints = await Complaint.find({ citizen: citizenId })
      .populate("department", "name").exec();
    if (!complaints || complaints.length === 0)
      return res.status(404).json({ error: "No complaints found" });
    res.json(complaints);
  } catch (error) {
    console.error("Error fetching complaints:", error);
    res.status(500).json({ error: "Server error" });
  }
};

// Fetch complaints for a department — scored & sorted by Python ML model
export const getComplaintsByDepartment = async (req, res) => {
  const { departmentId } = req.params;
  try {
    const complaints = await Complaint.find({ department: departmentId })
      .populate("department", "name")
      .exec();

    if (!complaints || complaints.length === 0)
      return res.status(404).json({ error: "No complaints found for this department" });

    // Send to Python ML microservice (localhost:5001) for scoring + sorting
    // Falls back to JS scorer automatically if Python service is down
    const sorted = await mlScoreAndSort(complaints);
    console.log(`[ML] Returned ${sorted.length} sorted complaints for department ${departmentId}`);

    res.json(sorted);
  } catch (error) {
    console.error("Error fetching complaints by department:", error);
    res.status(500).json({ error: "Server error" });
  }
};

export const voteComplaint = async (req, res) => {
  const { complaintId, voteType } = req.body;
  try {
    const citizen = req.user;
    if (!citizen) return res.status(401).json({ error: "Citizen not authenticated" });
    const complaint = await Complaint.findById(complaintId);
    if (!complaint) return res.status(404).json({ error: "Complaint not found" });
    if (complaint.votes.includes(citizen._id))
      return res.status(400).json({ error: "You have already voted on this complaint" });
    complaint.votes.push(citizen._id);
    await complaint.save();
    res.json({ message: `Complaint ${voteType}d successfully`, complaint });
  } catch (error) {
    console.error("Error voting on complaint:", error);
    res.status(500).json({ error: "Server error" });
  }
};

export const suggestDepartment = async (req, res) => {
  const { title } = req.body;
  if (!title || !title.trim()) return res.status(400).json({ error: "Title is required" });
  try {
    const allDepartments = await Department.find({});
    if (!allDepartments.length)
      return res.status(404).json({ error: "No departments available in database" });

    const suggestion = await getDepartmentSuggestion(title);
    const suggestedName = suggestion?.department;

    console.log(`[Controller] RAG suggested: "${suggestedName}"`);
    console.log(`[Controller] DB departments: ${allDepartments.map(d => d.name).join(", ")}`);

    const matched = matchDepartment(allDepartments, suggestedName);
    console.log(`[Controller] Matched to DB department: "${matched.name}" (${matched._id})`);

    return res.json({
      suggestedDepartment: matched.name,
      suggestedDepartmentId: matched._id,
    });
  } catch (err) {
    console.error("[suggestDepartment] ERROR:", err.message);
    res.status(500).json({ error: "Error suggesting department", detail: err.message });
  }
};