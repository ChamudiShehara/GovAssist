// middlewares/error.middleware.js

const errorHandler = (err, req, res, next) => {
  console.error(err.stack); // Log error details

  res.status(500).json({
    message: "Something went wrong! Please try again later.",
    error: err.message || "Internal Server Error",
  });
};

export default errorHandler;