// routes/minister.routes.js
import express from "express";
import {
  registerMinister,
  loginMinister,
  getMinisterProfile,
  getMinisterComplaints,
  getAvailableAgents,
  assignAgentToComplaint,
} from "../controllers/minister.controller.js";
import authMiddleware from "../middlewares/auth.middleware.js";

const router = express.Router();

router.post("/login-minister",   loginMinister);
router.get("/profile",           authMiddleware, getMinisterProfile);
router.get("/complaints",        authMiddleware, getMinisterComplaints);      // ?status=PENDING
router.get("/agents",            authMiddleware, getAvailableAgents);         // ?department=id&district=name
router.post("/assign-agent",     authMiddleware, assignAgentToComplaint);

export default router;