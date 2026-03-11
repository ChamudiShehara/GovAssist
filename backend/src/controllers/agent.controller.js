// controllers/agent.controller.js
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import Agent from "../models/Agent.js";
import Complaint from "../models/Complaint.js";

// controllers/agent.controller.js - registerAgent
export const registerAgent = async (req, res) => {
  const { name, email, password, department, district, agentType } = req.body;

  // 👇 Validate required fields explicitly
  if (!district || !agentType) {
    return res.status(400).json({ 
      error: "district and agentType are required fields" 
    });
  }

  try {
    const agentExists = await Agent.findOne({ email });
    if (agentExists) return res.status(400).json({ error: "Agent already exists" });

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    const newAgent = new Agent({
      name, email,
      password: hashedPassword,
      department, district, agentType,
    });

    await newAgent.save();
    const token = jwt.sign({ id: newAgent._id }, process.env.JWT_SECRET);
    res.status(201).json({ message: "Agent registered successfully", token });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Server Error" });
  }
};

export const loginAgent = async (req, res) => {
  const { email, password } = req.body;
  try {
    const agent = await Agent.findOne({ email });
    if (!agent) return res.status(400).json({ error: "Agent not found" });

    const isMatch = await bcrypt.compare(password, agent.password);
    if (!isMatch) return res.status(400).json({ error: "Invalid credentials" });

    const token = jwt.sign({ id: agent._id }, process.env.JWT_SECRET);
    res.json({ token });
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};

export const getAgentProfile = async (req, res) => {
  try {
    const agent = await Agent.findById(req.user._id).populate("department");
    if (!agent) return res.status(404).json({ error: "Agent not found" });
    res.json(agent);
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};

// 👇 Get complaints assigned to this agent
export const getAgentComplaints = async (req, res) => {
  try {
    const { status } = req.query; // optional filter: PENDING, IN_PROGRESS, RESOLVED, REJECTED
    const filter = { assignedAgent: req.user._id };
    if (status) filter.status = status.toUpperCase();

    const complaints = await Complaint.find(filter)
      .populate("department", "name")
      .populate("citizen", "name email")
      .sort({ createdAt: -1 });

    res.json(complaints);
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};

// 👇 Agent updates complaint status
export const updateAgentComplaintStatus = async (req, res) => {
  const { status } = req.body;
  const validStatuses = ["IN_PROGRESS", "RESOLVED", "REJECTED"];

  if (!validStatuses.includes(status)) {
    return res.status(400).json({ error: "Invalid status" });
  }

  try {
    const complaint = await Complaint.findOne({
      _id: req.params.id,
      assignedAgent: req.user._id, // agent can only update their own assigned complaints
    });

    if (!complaint) return res.status(404).json({ error: "Complaint not found or not assigned to you" });

    complaint.status = status;
    await complaint.save();
    res.json({ message: "Status updated", complaint });
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};