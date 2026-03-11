// pages/minister/MinisterDashboard.jsx
import React, { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';

const MinisterDashboard = () => {
  const [ministerData, setMinisterData] = useState(null);
  const [stats, setStats] = useState({ total: 0, pending: 0, inProgress: 0, resolved: 0, rejected: 0 });
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) { navigate('/minister/login'); return; }

    const fetchData = async () => {
      try {
        const [profileRes, complaintsRes] = await Promise.all([
          axios.get('http://localhost:5000/api/minister/profile', {
            headers: { Authorization: `Bearer ${token}` },
          }),
          axios.get('http://localhost:5000/api/minister/complaints', {
            headers: { Authorization: `Bearer ${token}` },
          }),
        ]);

        setMinisterData(profileRes.data);

        const all = complaintsRes.data;
        setStats({
          total:      all.length,
          pending:    all.filter(c => c.status === "PENDING").length,
          inProgress: all.filter(c => c.status === "IN_PROGRESS").length,
          resolved:   all.filter(c => c.status === "RESOLVED").length,
          rejected:   all.filter(c => c.status === "REJECTED").length,
        });
      } catch (error) {
        console.error('Error fetching minister data', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [navigate]);

  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  if (!ministerData) return <div className="min-h-screen flex items-center justify-center text-red-500">Failed to load</div>;

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-4xl mx-auto">

        {/* Header */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h1 className="text-2xl font-bold text-gray-800 mb-1">Minister Dashboard</h1>
          <p className="text-gray-500">Welcome, {ministerData.name}!</p>
        </div>

        {/* Stats */}
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

        {/* Profile */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-700 mb-4 border-b pb-2">Profile Details</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-500">Full Name</p>
              <p className="font-medium text-gray-800">{ministerData.name}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Email</p>
              <p className="font-medium text-gray-800">{ministerData.email}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Department</p>
              <p className="font-medium text-gray-800">
                {ministerData.department?.name || "No Department Assigned"}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Role</p>
              <span className="inline-block bg-purple-100 text-purple-600 text-xs font-medium px-2 py-1 rounded-full">
                Minister
              </span>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-700 mb-4 border-b pb-2">Quick Actions</h2>
          <div className="flex gap-4 flex-wrap">
            <button
              className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600"
              onClick={() => navigate('/minister/complaints')}
            >
              View Complaints
            </button>
            <button
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
              onClick={() => navigate('/minister/departments')}
            >
              View Department
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};

export default MinisterDashboard;