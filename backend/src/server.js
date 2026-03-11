// backend/src/server.js
import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import connectDB from "./config/db.js";
import adminRoutes from "./routes/admin.routes.js";
import citizenRoutes from "./routes/citizen.routes.js";
import ministerRoutes from "./routes/minister.routes.js";
import agentRoutes from "./routes/agent.routes.js";
import errorHandler from "./middlewares/error.middleware.js";
// NOTE: complaintRoutes are NOT imported here — they are already nested
// inside citizenRoutes via: router.use("/complaints", complaintRoutes)
// Mounting them again here caused duplicate route handling — removed.

dotenv.config();

const app = express();

app.use(cors({
  origin: "http://localhost:5173",
  methods: ["GET", "POST", "PUT", "DELETE"],
  credentials: true,
}));

app.use(express.json());

app.use("/api/admin", adminRoutes);
app.use("/api/citizen", citizenRoutes);   // /api/citizen/complaints/* handled inside here
app.use("/api/minister", ministerRoutes);
app.use("/api/agent", agentRoutes);

connectDB();

app.use(errorHandler);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server started on port ${PORT}`));