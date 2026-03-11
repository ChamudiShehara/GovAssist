// controllers/citizen.controller.js
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import Citizen from "../models/Citizen.js";

// Register new citizen
export const registerCitizen = async (req, res) => {
  const { name, email, password } = req.body;

  try {
    // Check if the citizen already exists
    const existingCitizen = await Citizen.findOne({ email });
    if (existingCitizen) {
      return res.status(400).json({ error: "Citizen already exists" });
    }

    // Hash the password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // Create and save new citizen
    const newCitizen = new Citizen({
      name,
      email,
      password: hashedPassword,
    });

    await newCitizen.save();
    res.status(201).json({ message: "Citizen registered successfully" });
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};

// Login citizen
export const loginCitizen = async (req, res) => {
  const { email, password } = req.body;

  try {
    const citizen = await Citizen.findOne({ email });
    if (!citizen) return res.status(400).json({ error: "Citizen not found" });

    // Compare the password
    const isMatch = await bcrypt.compare(password, citizen.password);
    if (!isMatch) return res.status(400).json({ error: "Invalid credentials" });

    // Create permanent JWT token (no expiration)
    const token = jwt.sign({ id: citizen._id }, process.env.JWT_SECRET);

    res.json({ token });
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};

// Get the authenticated citizen's details
export const getCitizenDetails = async (req, res) => {
  try {
    const citizen = await Citizen.findById(req.user.id); // Assuming the citizen's info is stored in req.user
    if (!citizen) {
      return res.status(404).json({ error: "Citizen not found" });
    }
    res.json({
      name: citizen.name,
      email: citizen.email,
    }); // Send the citizen details (name, email, etc.)
  } catch (error) {
    res.status(500).json({ error: "Server error" });
  }
};