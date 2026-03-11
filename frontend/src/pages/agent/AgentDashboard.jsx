// pages/agent/AgentDashboard.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

export default function AgentDashboard() {
  const [agent, setAgent] = useState(null);
  const [stats, setStats] = useState({ total: 0, pending: 0, inProgress: 0, resolved: 0, rejected: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      const token = localStorage.getItem('token');
      if (!token) { navigate('/agent/login'); return; }

      try {
        const [profileRes, complaintsRes] = await Promise.all([
          axios.get('http://localhost:5000/api/agent/profile', {
            headers: { Authorization: `Bearer ${token}` },
          }),
          axios.get('http://localhost:5000/api/agent/complaints', {
            headers: { Authorization: `Bearer ${token}` },
          }),
        ]);

        setAgent(profileRes.data);

        const all = complaintsRes.data;
        setStats({
          total:      all.length,
          pending:    all.filter(c => c.status === "PENDING").length,
          inProgress: all.filter(c => c.status === "IN_PROGRESS").length,
          resolved:   all.filter(c => c.status === "RESOLVED").length,
          rejected:   all.filter(c => c.status === "REJECTED").length,
        });
      } catch (err) {
        setError('Failed to load dashboard');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [navigate]);

  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  if (error)   return <div className="min-h-screen flex items-center justify-center text-red-500">{error}</div>;

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-4xl mx-auto">

        {/* Header */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h1 className="text-2xl font-bold text-gray-800 mb-1">Agent Dashboard</h1>
          <p className="text-gray-500">Welcome back, {agent.name}!</p>
        </div>

        {/* Complaint Stats */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
          {[
            { label: "Total",       value: stats.total,      color: "bg-gray-100 text-gray-700" },
            { label: "Pending",     value: stats.pending,    color: "bg-yellow-100 text-yellow-700" },
            { label: "In Progress", value: stats.inProgress, color: "bg-blue-100 text-blue-700" },
            { label: "Resolved",    value: stats.resolved,   color: "bg-green-100 text-green-700" },
            { label: "Rejected",    value: stats.rejected,   color: "bg-red-100 text-red-700" },
          ].map(stat => (
            <div key={stat.label} className={`rounded-lg p-4 text-center ${stat.color}`}>
              <p className="text-2xl font-bold">{stat.value}</p>
              <p className="text-sm">{stat.label}</p>
            </div>
          ))}
        </div>

        {/* Profile Details */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-700 mb-4 border-b pb-2">Profile Details</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-500">Full Name</p>
              <p className="font-medium text-gray-800">{agent.name}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Email</p>
              <p className="font-medium text-gray-800">{agent.email}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Department</p>
              <p className="font-medium text-gray-800">{agent.department?.name || "Not assigned"}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Agent Type</p>
              <p className="font-medium text-gray-800">{agent.agentType || "N/A"}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">District</p>
              <p className="font-medium text-gray-800">{agent.district || "N/A"}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Member Since</p>
              <p className="font-medium text-gray-800">
                {new Date(agent.createdAt).toLocaleDateString('en-US', {
                  year: 'numeric', month: 'long', day: 'numeric'
                })}
              </p>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-700 mb-4 border-b pb-2">Quick Actions</h2>
          <div className="flex gap-4 flex-wrap">
            <button
              className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
              onClick={() => navigate('/agent/complaints')}
            >
              View All Complaints
            </button>
            <button
              className="bg-yellow-500 text-white px-4 py-2 rounded hover:bg-yellow-600"
              onClick={() => navigate('/agent/complaints?status=PENDING')}
            >
              Pending Complaints
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}