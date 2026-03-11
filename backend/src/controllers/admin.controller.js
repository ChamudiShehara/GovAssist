// controllers/admin.controller.js
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import Admin from "../models/Admin.js";

// Register Admin
export const registerAdmin = async (req, res) => {
  const { name, email, password } = req.body;

  try {
    // Check if the admin already exists
    const adminExists = await Admin.findOne({ email });
    if (adminExists) return res.status(400).json({ error: "Admin already exists" });

    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // Create a new admin
    const newAdmin = new Admin({
      name,
      email,
      password: hashedPassword,
    });

    // Save the admin to the database
    await newAdmin.save();
    res.status(201).json({ message: "Admin registered successfully" });
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};

// Login Admin
export const loginAdmin = async (req, res) => {
  const { email, password } = req.body;

  try {
    // Find admin by email
    const admin = await Admin.findOne({ email });
    if (!admin) return res.status(400).json({ error: "Admin not found" });

    // Compare passwords
    const isMatch = await bcrypt.compare(password, admin.password);
    if (!isMatch) return res.status(400).json({ error: "Invalid credentials" });

    // Generate permanent JWT token (no expiration)
    const token = jwt.sign({ id: admin._id }, process.env.JWT_SECRET);

    // Send token in response
    res.json({ token });
  } catch (error) {
    res.status(500).json({ error: "Server Error" });
  }
};