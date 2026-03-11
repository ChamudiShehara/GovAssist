import jwt from "jsonwebtoken";
import Citizen from "../models/Citizen.js";
import Admin from "../models/Admin.js";
import Minister from "../models/Minister.js";
import Agent from "../models/Agent.js"; // 👈 add this

const authMiddleware = async (req, res, next) => {
  const token = req.header("Authorization")?.replace("Bearer ", "");

  if (!token) {
    return res.status(401).json({ error: "No token, authorization denied" });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);

    let user = await Admin.findById(decoded.id);
    if (user) { req.user = user; req.userRole = "admin"; return next(); }

    user = await Citizen.findById(decoded.id);
    if (user) { req.user = user; req.userRole = "citizen"; return next(); }

    user = await Minister.findById(decoded.id);
    if (user) { req.user = user; req.userRole = "minister"; return next(); }

    user = await Agent.findById(decoded.id); // 👈 add this
    if (user) { req.user = user; req.userRole = "agent"; return next(); }

    return res.status(404).json({ error: "User not found" });
  } catch (error) {
    console.error("JWT Error:", error.message);
    res.status(400).json({ error: "Token is not valid" });
  }
};

export default authMiddleware;