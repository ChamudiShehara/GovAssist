// pages/agent/AgentComplaintsPage.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const STATUS_TABS = ["ALL", "PENDING", "IN_PROGRESS", "RESOLVED", "REJECTED"];

const statusColor = {
  PENDING:     "bg-yellow-100 text-yellow-700",
  IN_PROGRESS: "bg-blue-100 text-blue-700",
  RESOLVED:    "bg-green-100 text-green-700",
  REJECTED:    "bg-red-100 text-red-700",
};

export default function AgentComplaintsPage() {
  const [complaints, setComplaints] = useState([]);
  const [activeTab, setActiveTab] = useState("ALL");
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  const fetchComplaints = async (status) => {
    const token = localStorage.getItem("token");
    setLoading(true);
    try {
      const url = status === "ALL"
        ? "http://localhost:5000/api/agent/complaints"
        : `http://localhost:5000/api/agent/complaints?status=${status}`;

      const res = await axios.get(url, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setComplaints(res.data);
    } catch (err) {
      setComplaints([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchComplaints(activeTab); }, [activeTab]);

  const handleStatusUpdate = async (complaintId, newStatus) => {
    const token = localStorage.getItem("token");
    try {
      await axios.put(
        `http://localhost:5000/api/agent/complaints/${complaintId}/status`,
        { status: newStatus },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      fetchComplaints(activeTab); // refresh
    } catch (err) {
      alert("Failed to update status");
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">My Assigned Complaints</h1>

        {/* Status Tabs */}
        <div className="flex space-x-2 mb-6 flex-wrap gap-2">
          {STATUS_TABS.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition
                ${activeTab === tab
                  ? "bg-blue-600 text-white"
                  : "bg-white text-gray-600 hover:bg-gray-200"}`}
            >
              {tab.replace("_", " ")}
            </button>
          ))}
        </div>

        {/* Complaints List */}
        {loading ? (
          <p className="text-gray-500">Loading...</p>
        ) : complaints.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-6 text-center text-gray-500">
            No complaints found for this status.
          </div>
        ) : (
          <div className="space-y-4">
            {complaints.map(complaint => (
              <div key={complaint._id} className="bg-white rounded-lg shadow p-5">
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h3 className="font-semibold text-gray-800 text-lg">{complaint.title}</h3>
                    <p className="text-sm text-gray-500">
                      {complaint.citizen?.name} • {complaint.district}
                    </p>
                  </div>
                  <span className={`text-xs font-medium px-2 py-1 rounded-full ${statusColor[complaint.status]}`}>
                    {complaint.status.replace("_", " ")}
                  </span>
                </div>

                <p className="text-gray-600 text-sm mb-3">{complaint.description}</p>

                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">
                    Dept: {complaint.department?.name || "N/A"} •
                    Priority: {complaint.priority} •
                    {new Date(complaint.createdAt).toLocaleDateString()}
                  </span>

                  {/* Status update buttons */}
                  {complaint.status !== "RESOLVED" && complaint.status !== "REJECTED" && (
                    <div className="flex gap-2">
                      {complaint.status === "PENDING" && (
                        <button
                          onClick={() => handleStatusUpdate(complaint._id, "IN_PROGRESS")}
                          className="bg-blue-500 text-white text-xs px-3 py-1 rounded hover:bg-blue-600"
                        >
                          Start Progress
                        </button>
                      )}
                      {complaint.status === "IN_PROGRESS" && (
                        <button
                          onClick={() => handleStatusUpdate(complaint._id, "RESOLVED")}
                          className="bg-green-500 text-white text-xs px-3 py-1 rounded hover:bg-green-600"
                        >
                          Mark Resolved
                        </button>
                      )}
                      <button
                        onClick={() => handleStatusUpdate(complaint._id, "REJECTED")}
                        className="bg-red-500 text-white text-xs px-3 py-1 rounded hover:bg-red-600"
                      >
                        Reject
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}