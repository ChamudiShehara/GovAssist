// routes/agent.routes.js
import express from "express";
import {
  registerAgent,
  loginAgent,
  getAgentProfile,
  getAgentComplaints,
  updateAgentComplaintStatus,
} from "../controllers/agent.controller.js";
import authMiddleware from "../middlewares/auth.middleware.js";

const router = express.Router();

router.post("/login-agent", loginAgent);
router.get("/profile",    authMiddleware, getAgentProfile);
router.get("/complaints", authMiddleware, getAgentComplaints);          // 👈 with ?status=PENDING etc.
router.put("/complaints/:id/status", authMiddleware, updateAgentComplaintStatus); // 👈 update status

export default router;