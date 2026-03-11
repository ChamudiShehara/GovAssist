// middlewares/validate.middleware.js
import { body, validationResult } from "express-validator";

// Admin registration validation rules
const registerAdminValidation = [
  body("name").not().isEmpty().withMessage("Name is required"),
  body("email").isEmail().withMessage("Invalid email"),
  body("password").isLength({ min: 6 }).withMessage("Password must be at least 6 characters long"),
];

// Middleware to handle validation errors
const validate = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  next(); // Proceed to the next middleware or route handler
};

export { registerAdminValidation, validate };