// routes/citizen.routes.js
import express from "express";
import { registerCitizen, loginCitizen, getCitizenDetails } from "../controllers/citizen.controller.js"; // Import the getCitizenDetails controller
import { validateCitizenRegistration, validate } from "../middlewares/citizenValidation.middleware.js";
import complaintRoutes from "./complaint.routes.js"; // Import complaint routes for citizens
import authMiddleware from "../middlewares/auth.middleware.js"; // Import auth middleware to protect routes

const router = express.Router();

// Route to register a new citizen with validation
router.post("/register", validateCitizenRegistration, validate, registerCitizen);

// Route to login a citizen
router.post("/login", loginCitizen);

// Route to get authenticated citizen's details (protected)
router.get("/me", authMiddleware, getCitizenDetails);  // Protected route for citizen's own details

// Use the complaint routes for the citizen
router.use("/complaints", complaintRoutes);

export default router;