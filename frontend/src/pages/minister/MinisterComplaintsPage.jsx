// pages/minister/MinisterComplaintsPage.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const STATUS_TABS = ["ALL", "PENDING", "IN_PROGRESS", "RESOLVED", "REJECTED"];

const statusColor = {
  PENDING:     "bg-yellow-100 text-yellow-700",
  IN_PROGRESS: "bg-blue-100 text-blue-700",
  RESOLVED:    "bg-green-100 text-green-700",
  REJECTED:    "bg-red-100 text-red-700",
};

export default function MinisterComplaintsPage() {
  const [complaints, setComplaints]     = useState([]);
  const [activeTab, setActiveTab]       = useState("ALL");
  const [loading, setLoading]           = useState(true);
  const [selectedComplaint, setSelectedComplaint] = useState(null); // for modal
  const [agents, setAgents]             = useState([]);
  const [selectedAgent, setSelectedAgent] = useState("");
  const [assigning, setAssigning]       = useState(false);
  const [modalOpen, setModalOpen]       = useState(false);

  const token = localStorage.getItem("token");

  const fetchComplaints = async (status) => {
    setLoading(true);
    try {
      const url = status === "ALL"
        ? "http://localhost:5000/api/minister/complaints"
        : `http://localhost:5000/api/minister/complaints?status=${status}`;

      const res = await axios.get(url, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setComplaints(res.data);
    } catch {
      setComplaints([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchComplaints(activeTab); }, [activeTab]);

  const openAssignModal = async (complaint) => {
    setSelectedComplaint(complaint);
    setSelectedAgent("");
    setModalOpen(true);

    // Fetch agents for this complaint's department and district
    try {
      const res = await axios.get(
        `http://localhost:5000/api/minister/agents?department=${complaint.department?._id}&district=${complaint.district}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setAgents(res.data);
    } catch {
      setAgents([]);
    }
  };

  const handleAssign = async () => {
    if (!selectedAgent) return alert("Please select an agent");
    setAssigning(true);
    try {
      await axios.post(
        "http://localhost:5000/api/minister/assign-agent",
        { complaintId: selectedComplaint._id, agentId: selectedAgent },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setModalOpen(false);
      fetchComplaints(activeTab);
    } catch {
      alert("Failed to assign agent");
    } finally {
      setAssigning(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">Department Complaints</h1>

        {/* Status Tabs */}
        <div className="flex space-x-2 mb-6 flex-wrap gap-2">
          {STATUS_TABS.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition
                ${activeTab === tab
                  ? "bg-purple-600 text-white"
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
            No complaints found.
          </div>
        ) : (
          <div className="space-y-4">
            {complaints.map(complaint => (
              <div key={complaint._id} className="bg-white rounded-lg shadow p-5">
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h3 className="font-semibold text-gray-800 text-lg">{complaint.title}</h3>
                    <p className="text-sm text-gray-500">
                      {complaint.citizen?.name} • {complaint.district} • {complaint.department?.name}
                    </p>
                  </div>
                  <span className={`text-xs font-medium px-2 py-1 rounded-full ${statusColor[complaint.status]}`}>
                    {complaint.status.replace("_", " ")}
                  </span>
                </div>

                <p className="text-gray-600 text-sm mb-3">{complaint.description}</p>

                {/* Assigned Agent */}
                {complaint.assignedAgent ? (
                  <div className="text-sm text-green-600 mb-3">
                    ✅ Assigned to: <strong>{complaint.assignedAgent.name}</strong>
                    {" "}({complaint.assignedAgent.district} • {complaint.assignedAgent.agentType})
                  </div>
                ) : (
                  <div className="text-sm text-yellow-600 mb-3">⚠️ No agent assigned</div>
                )}

                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">
                    Priority: {complaint.priority} • {new Date(complaint.createdAt).toLocaleDateString()}
                  </span>
                  {/* Assign button — only for PENDING complaints */}
                  {complaint.status === "PENDING" && (
                    <button
                      onClick={() => openAssignModal(complaint)}
                      className="bg-purple-500 text-white text-sm px-4 py-1.5 rounded hover:bg-purple-600"
                    >
                      {complaint.assignedAgent ? "Reassign Agent" : "Assign Agent"}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Assign Agent Modal */}
      {modalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-md">
            <h2 className="text-lg font-bold text-gray-800 mb-2">Assign Agent</h2>
            <p className="text-sm text-gray-500 mb-4">
              Complaint: <strong>{selectedComplaint?.title}</strong><br/>
              District: <strong>{selectedComplaint?.district}</strong>
            </p>

            {agents.length === 0 ? (
              <p className="text-red-500 text-sm mb-4">
                No agents available for this district and department.
              </p>
            ) : (
              <select
                className="w-full border border-gray-300 rounded p-2 mb-4 text-sm"
                value={selectedAgent}
                onChange={(e) => setSelectedAgent(e.target.value)}
              >
                <option value="">-- Select an Agent --</option>
                {agents.map(agent => (
                  <option key={agent._id} value={agent._id}>
                    {agent.name} — {agent.agentType} ({agent.district})
                  </option>
                ))}
              </select>
            )}

            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setModalOpen(false)}
                className="px-4 py-2 text-sm text-gray-600 bg-gray-100 rounded hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={handleAssign}
                disabled={assigning || !selectedAgent}
                className="px-4 py-2 text-sm text-white bg-purple-500 rounded hover:bg-purple-600 disabled:opacity-50"
              >
                {assigning ? "Assigning..." : "Assign"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}