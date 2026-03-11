// controllers/minister.controller.js
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import Minister from "../models/Minister.js";
import Complaint from "../models/Complaint.js";
import Agent from "../models/Agent.js";

export const registerMinister = async (req, res) => {
  const { name, email, password, department } = req.body;
  try {
    const ministerExists = await Minister.findOne({ email });
    if (ministerExists) return res.status(400).json({ error: "Minister already exists" });

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    const newMinister = new Minister({ name, email, password: hashedPassword, department });
    await newMinister.save();

    const token = jwt.sign({ id: newMinister._id }, process.env.JWT_SECRET);
    res.status(201).json({ message: "Minister registered successfully", token });
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};

export const loginMinister = async (req, res) => {
  const { email, password } = req.body;
  try {
    const minister = await Minister.findOne({ email });
    if (!minister) return res.status(400).json({ error: "Minister not found" });

    const isMatch = await bcrypt.compare(password, minister.password);
    if (!isMatch) return res.status(400).json({ error: "Invalid credentials" });

    const token = jwt.sign({ id: minister._id }, process.env.JWT_SECRET);
    res.json({ token });
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};

export const getMinisterProfile = async (req, res) => {
  try {
    const minister = await Minister.findById(req.user._id).populate("department");
    if (!minister) return res.status(404).json({ error: "Minister not found" });
    res.json(minister);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch minister profile" });
  }
};

// Get complaints for minister's department
export const getMinisterComplaints = async (req, res) => {
  try {
    const minister = await Minister.findById(req.user._id);
    if (!minister?.department) {
      return res.status(400).json({ error: "Minister has no department assigned" });
    }

    const { status } = req.query;
    const filter = { department: minister.department };
    if (status) filter.status = status.toUpperCase();

    const complaints = await Complaint.find(filter)
      .populate("citizen", "name email")
      .populate("department", "name")
      .populate("assignedAgent", "name email district agentType")
      .sort({ createdAt: -1 });

    res.json(complaints);
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};

// Get agents filtered by department and district
export const getAvailableAgents = async (req, res) => {
  const { department, district } = req.query;
  try {
    const filter = {};
    if (department) filter.department = department;
    if (district)   filter.district   = district;

    const agents = await Agent.find(filter).populate("department", "name");
    res.json(agents);
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};

// Assign agent to complaint
export const assignAgentToComplaint = async (req, res) => {
  const { complaintId, agentId } = req.body;
  try {
    const minister = await Minister.findById(req.user._id);

    const complaint = await Complaint.findOne({
      _id: complaintId,
      department: minister.department, // minister can only assign within their dept
    });

    if (!complaint) {
      return res.status(404).json({ error: "Complaint not found in your department" });
    }

    complaint.assignedAgent = agentId;
    complaint.status = "IN_PROGRESS";
    await complaint.save();

    const updated = await Complaint.findById(complaintId)
      .populate("assignedAgent", "name email district agentType")
      .populate("department", "name")
      .populate("citizen", "name email");

    res.json({ message: "Agent assigned successfully", complaint: updated });
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};