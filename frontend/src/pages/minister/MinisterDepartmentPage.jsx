// pages/minister/MinisterDepartmentPage.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

export default function MinisterDepartmentPage() {
  const [minister, setMinister] = useState(null);
  const [agents, setAgents]     = useState([]);
  const [loading, setLoading]   = useState(true);
  const navigate = useNavigate();
  const token = localStorage.getItem("token");

  useEffect(() => {
    const fetchData = async () => {
      try {
        const profileRes = await axios.get(
          "http://localhost:5000/api/minister/profile",
          { headers: { Authorization: `Bearer ${token}` } }
        );
        setMinister(profileRes.data);

        // Fetch agents in this department
        const agentsRes = await axios.get(
          `http://localhost:5000/api/minister/agents?department=${profileRes.data.department?._id}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        setAgents(agentsRes.data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-4xl mx-auto">

        {/* Department Info */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h1 className="text-2xl font-bold text-gray-800 mb-1">My Department</h1>
          <p className="text-xl text-purple-600 font-semibold">
            {minister?.department?.name || "No Department Assigned"}
          </p>
          {minister?.department?.description && (
            <p className="text-gray-500 mt-2">{minister.department.description}</p>
          )}
        </div>

        {/* Agents in Department */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-700 mb-4 border-b pb-2">
            Agents in this Department ({agents.length})
          </h2>

          {agents.length === 0 ? (
            <p className="text-gray-500">No agents assigned to this department yet.</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {agents.map(agent => (
                <div key={agent._id} className="border border-gray-200 rounded-lg p-4">
                  <p className="font-semibold text-gray-800">{agent.name}</p>
                  <p className="text-sm text-gray-500">{agent.email}</p>
                  <div className="flex gap-2 mt-2">
                    <span className="text-xs bg-green-100 text-green-600 px-2 py-0.5 rounded-full">
                      {agent.agentType}
                    </span>
                    <span className="text-xs bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full">
                      {agent.district}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}