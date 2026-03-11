import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const CreateComplaint = () => {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [address, setAddress] = useState("");
  const [district, setDistrict] = useState("");
  const [department, setDepartment] = useState("");
  const [departments, setDepartments] = useState([]);
  const [priority, setPriority] = useState("MEDIUM");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [citizenName, setCitizenName] = useState("");
  const [citizenEmail, setCitizenEmail] = useState("");
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [suggestionNote, setSuggestionNote] = useState("");
  const [titleTouched, setTitleTouched] = useState(false);
  const navigate = useNavigate();

  const districtsList = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya", "Galle",
    "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar", "Vavuniya", "Batticaloa",
    "Ampara", "Trincomalee", "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa",
    "Badulla", "Moneragala", "Ratnapura", "Kegalle", "Mullaitivu",
  ];

  useEffect(() => {
    const fetchData = async () => {
      const token = localStorage.getItem("token");
      if (!token) return navigate("/citizen/login");
      try {
        const [deptRes, citizenRes] = await Promise.all([
          axios.get("http://localhost:5000/api/admin/departments", {
            headers: { Authorization: `Bearer ${token}` },
          }),
          axios.get("http://localhost:5000/api/citizen/me", {
            headers: { Authorization: `Bearer ${token}` },
          }),
        ]);
        setDepartments(deptRes.data);
        setCitizenName(citizenRes.data.name);
        setCitizenEmail(citizenRes.data.email);
      } catch {
        setError("Failed to fetch data. Please try again.");
      }
    };
    fetchData();
  }, [navigate]);

  const handleSuggestDepartment = async () => {
    if (!title.trim()) {
      setSuggestionNote("Please enter a complaint title first.");
      return;
    }
    setIsSuggesting(true);
    setSuggestionNote("");
    try {
      const token = localStorage.getItem("token");
      const response = await axios.post(
        "http://localhost:5000/api/citizen/complaints/suggest-department",
        { title },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      const { suggestedDepartment, suggestedDepartmentId } = response.data;

      if (suggestedDepartmentId) {
        // Backend already matched to a real department — auto-select it
        setDepartment(suggestedDepartmentId);
        setSuggestionNote(`✓ AI suggested & matched: "${suggestedDepartment}"`);
      } else if (suggestedDepartment) {
        setSuggestionNote(
          `AI suggested "${suggestedDepartment}" — no exact match found. Please select manually.`
        );
      } else {
        setSuggestionNote("No suggestion available. Please select manually.");
      }
    } catch {
      setSuggestionNote("Could not get a suggestion. Please select manually.");
    } finally {
      setIsSuggesting(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const token = localStorage.getItem("token");
    if (!token) return setError("No token, authorization denied.");

    try {
      await axios.post(
        "http://localhost:5000/api/citizen/complaints/add",
        { title, description, phoneNumber, address, district, department, priority },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setSuccess("Complaint submitted successfully!");
      setTimeout(() => navigate("/citizen/dashboard"), 2000);
    } catch {
      setError("Failed to submit complaint. Please try again.");
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-start justify-center py-10 px-4">
      <div className="w-full max-w-2xl">

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-1 h-8 bg-blue-600 rounded-full" />
            <h1 className="text-2xl font-bold text-slate-800 tracking-tight">File a Complaint</h1>
          </div>
          <p className="text-slate-500 text-sm ml-4 pl-3">
            Submit your complaint to the relevant government department.
          </p>
        </div>

        {/* Citizen Info Card */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 mb-6 flex items-center gap-4 shadow-sm">
          <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-700 font-bold text-sm flex-shrink-0">
            {citizenName?.charAt(0)?.toUpperCase() || "C"}
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-700">{citizenName || "Loading..."}</p>
            <p className="text-xs text-slate-400">{citizenEmail || "..."}</p>
          </div>
          <div className="ml-auto">
            <span className="text-xs bg-blue-50 text-blue-600 border border-blue-100 px-2 py-1 rounded-full font-medium">
              Citizen
            </span>
          </div>
        </div>

        {/* Alerts */}
        {error && (
          <div className="mb-5 flex items-start gap-3 bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">
            <span className="mt-0.5">⚠️</span>
            <span>{error}</span>
          </div>
        )}
        {success && (
          <div className="mb-5 flex items-start gap-3 bg-green-50 border border-green-200 text-green-700 text-sm px-4 py-3 rounded-lg">
            <span className="mt-0.5">✅</span>
            <span>{success}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
          <div className="p-6 space-y-5">

            {/* Title + Suggest */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-1.5">
                Complaint Title <span className="text-red-400">*</span>
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  className="flex-1 border border-slate-200 bg-slate-50 focus:bg-white focus:border-blue-400 focus:ring-2 focus:ring-blue-100 rounded-lg px-3 py-2.5 text-sm text-slate-800 outline-none transition-all"
                  placeholder="e.g. Road damage near school, Water supply issue..."
                  value={title}
                  onChange={(e) => {
                    setTitle(e.target.value);
                    setTitleTouched(true);
                    setSuggestionNote("");
                  }}
                  required
                />
                <button
                  type="button"
                  onClick={handleSuggestDepartment}
                  disabled={isSuggesting || !title.trim()}
                  className="flex-shrink-0 flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white text-sm font-medium px-4 py-2.5 rounded-lg transition-colors"
                >
                  {isSuggesting ? (
                    <>
                      <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                      </svg>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                      Suggest
                    </>
                  )}
                </button>
              </div>
              <p className="mt-1 text-xs text-slate-400">
                Enter your complaint title, then click <strong>Suggest</strong> to auto-detect the right department.
              </p>
            </div>

            {/* Description */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-1.5">
                Description <span className="text-red-400">*</span>
              </label>
              <textarea
                rows={4}
                className="w-full border border-slate-200 bg-slate-50 focus:bg-white focus:border-blue-400 focus:ring-2 focus:ring-blue-100 rounded-lg px-3 py-2.5 text-sm text-slate-800 outline-none transition-all resize-none"
                placeholder="Describe your complaint in detail..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                required
              />
            </div>

            {/* Phone + Address row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-1.5">
                  Phone Number <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  className="w-full border border-slate-200 bg-slate-50 focus:bg-white focus:border-blue-400 focus:ring-2 focus:ring-blue-100 rounded-lg px-3 py-2.5 text-sm text-slate-800 outline-none transition-all"
                  placeholder="07X XXX XXXX"
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-1.5">
                  District <span className="text-red-400">*</span>
                </label>
                <select
                  className="w-full border border-slate-200 bg-slate-50 focus:bg-white focus:border-blue-400 focus:ring-2 focus:ring-blue-100 rounded-lg px-3 py-2.5 text-sm text-slate-800 outline-none transition-all"
                  value={district}
                  onChange={(e) => setDistrict(e.target.value)}
                  required
                >
                  <option value="">Select District</option>
                  {districtsList.map((d) => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Address */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-1.5">
                Address <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                className="w-full border border-slate-200 bg-slate-50 focus:bg-white focus:border-blue-400 focus:ring-2 focus:ring-blue-100 rounded-lg px-3 py-2.5 text-sm text-slate-800 outline-none transition-all"
                placeholder="Your full address"
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                required
              />
            </div>

            {/* Department */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-1.5">
                Department <span className="text-red-400">*</span>
              </label>
              <select
                className={`w-full border rounded-lg px-3 py-2.5 text-sm outline-none transition-all ${
                  department
                    ? "border-green-300 bg-green-50 text-slate-800 focus:border-green-400 focus:ring-2 focus:ring-green-100"
                    : "border-slate-200 bg-slate-50 text-slate-800 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 focus:bg-white"
                }`}
                value={department}
                onChange={(e) => {
                  setDepartment(e.target.value);
                  setSuggestionNote("");
                }}
                required
              >
                <option value="">Select Department</option>
                {departments.map((dept) => (
                  <option key={dept._id} value={dept._id}>
                    {dept.name}
                  </option>
                ))}
              </select>

              {/* Suggestion feedback */}
              {suggestionNote && (
                <div
                  className={`mt-2 text-xs flex items-center gap-1.5 px-3 py-2 rounded-lg ${
                    suggestionNote.startsWith("✓")
                      ? "bg-green-50 text-green-700 border border-green-200"
                      : suggestionNote.includes("no exact match") || suggestionNote.includes("Could not")
                      ? "bg-amber-50 text-amber-700 border border-amber-200"
                      : "bg-slate-50 text-slate-600 border border-slate-200"
                  }`}
                >
                  {suggestionNote}
                </div>
              )}
            </div>

            {/* Priority */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2">
                Priority Level <span className="text-red-400">*</span>
              </label>
              <div className="flex gap-3">
                {["LOW", "MEDIUM", "HIGH"].map((level) => (
                  <label
                    key={level}
                    className={`flex-1 flex items-center justify-center gap-2 cursor-pointer border rounded-lg py-2.5 text-sm font-medium transition-all ${
                      priority === level
                        ? level === "HIGH"
                          ? "bg-red-50 border-red-300 text-red-700"
                          : level === "MEDIUM"
                          ? "bg-amber-50 border-amber-300 text-amber-700"
                          : "bg-slate-100 border-slate-300 text-slate-700"
                        : "bg-slate-50 border-slate-200 text-slate-400 hover:border-slate-300"
                    }`}
                  >
                    <input
                      type="radio"
                      name="priority"
                      value={level}
                      checked={priority === level}
                      onChange={() => setPriority(level)}
                      className="hidden"
                    />
                    <span
                      className={`w-2 h-2 rounded-full ${
                        level === "HIGH"
                          ? "bg-red-400"
                          : level === "MEDIUM"
                          ? "bg-amber-400"
                          : "bg-slate-400"
                      }`}
                    />
                    {level.charAt(0) + level.slice(1).toLowerCase()}
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Submit footer */}
          <div className="px-6 py-4 bg-slate-50 border-t border-slate-100 flex items-center justify-between">
            <button
              type="button"
              onClick={() => navigate("/citizen/dashboard")}
              className="text-sm text-slate-500 hover:text-slate-700 font-medium transition-colors"
            >
              ← Cancel
            </button>
            <button
              type="submit"
              className="bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white text-sm font-semibold px-6 py-2.5 rounded-lg transition-colors shadow-sm"
            >
              Submit Complaint
            </button>
          </div>
        </form>

        <p className="text-center text-xs text-slate-400 mt-4">
          Your complaint will be reviewed by the assigned department.
        </p>
      </div>
    </div>
  );
};

export default CreateComplaint;