import express from "express";
import { registerAdmin, loginAdmin } from "../controllers/admin.controller.js";
import { registerMinister, loginMinister } from "../controllers/minister.controller.js";
import { getDepartments, createDepartment } from "../controllers/department.controller.js";
import authMiddleware from "../middlewares/auth.middleware.js";
import { registerAdminValidation, validate } from "../middlewares/validate.middleware.js";
import { registerAgent, loginAgent } from "../controllers/agent.controller.js";

const router = express.Router();

router.post("/register", registerAdminValidation, validate, registerAdmin);
router.post("/login", loginAdmin);
router.post("/register-minister", authMiddleware, registerMinister);
router.get("/departments", authMiddleware, getDepartments);
router.post("/create-department", authMiddleware, createDepartment);
router.post("/register-agent", authMiddleware, registerAgent);
router.post("/login-agent", loginAgent);

// ✅ Single correct profile route
router.get("/profile", authMiddleware, (req, res) => {
  res.json(req.user);
});

export default router;