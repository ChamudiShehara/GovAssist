// middlewares/citizenValidation.middleware.js
import { body, validationResult } from "express-validator";

const validateCitizenRegistration = [
  body("name").not().isEmpty().withMessage("Name is required"),
  body("email").isEmail().withMessage("Invalid email"),
  body("password").isLength({ min: 6 }).withMessage("Password must be at least 6 characters long"),
];

const validate = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  next();
};

export { validateCitizenRegistration, validate };