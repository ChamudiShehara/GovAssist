// src/routes/complaint.routes.js
import express from "express";
import {
  addComplaint,
  deleteComplaint,
  updateComplaintStatus,
  getComplaintsByCitizen,
  getComplaintsByDepartment,
  voteComplaint,  // Import voteComplaint function
  suggestDepartment,  // Import suggestDepartment function
} from "../controllers/complaint.controller.js"; // Import all controllers
import authMiddleware from "../middlewares/auth.middleware.js";

const router = express.Router();

// Route for adding a complaint
router.post("/add", authMiddleware, addComplaint);

// Route for fetching complaints of the authenticated citizen
router.get("/", authMiddleware, getComplaintsByCitizen);

// Route for deleting a complaint
router.delete("/:id", authMiddleware, deleteComplaint);

// Route for updating a complaint status
router.put("/:id", authMiddleware, updateComplaintStatus);

// Route for fetching complaints for a specific department
router.get("/department/:departmentId", getComplaintsByDepartment); // Fetch complaints for a specific department

// Route for voting for a complaint
router.post("/vote", authMiddleware, voteComplaint); // Route for voting on complaints

// Route for suggesting a department based on the complaint title
router.post("/suggest-department", authMiddleware, suggestDepartment); // New route for suggesting department

export default router;