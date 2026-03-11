import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const SRI_LANKA_DISTRICTS = [
  "Ampara", "Anuradhapura", "Badulla", "Batticaloa", "Colombo",
  "Galle", "Gampaha", "Hambantota", "Jaffna", "Kalutara",
  "Kandy", "Kegalle", "Kilinochchi", "Kurunegala", "Mannar",
  "Matale", "Matara", "Monaragala", "Mullaitivu", "Nuwara Eliya",
  "Polonnaruwa", "Puttalam", "Ratnapura", "Trincomalee", "Vavuniya"
];

const AGENT_TYPES = [
  "Road Service", "Public Transport", "Water Supply", "Electricity",
  "Waste Management", "Health Services", "Education", "Housing",
  "Agriculture", "Police Services", "Fire & Rescue", "Social Services"
];

const CreateAgentPage = () => {
  const [name, setName]             = useState('');
  const [email, setEmail]           = useState('');
  const [password, setPassword]     = useState('');
  const [department, setDepartment] = useState('');
  const [district, setDistrict]     = useState('');   // 👈 new
  const [agentType, setAgentType]   = useState('');   // 👈 new
  const [departments, setDepartments] = useState([]);
  const [error, setError]           = useState('');
  const [success, setSuccess]       = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchDepartments = async () => {
      const token = localStorage.getItem('token');
      if (!token) return navigate('/admin/login');

      try {
        const response = await axios.get('http://localhost:5000/api/admin/departments', {
          headers: { Authorization: `Bearer ${token}` },
        });
        setDepartments(response.data);
      } catch (err) {
        setError('Failed to fetch departments');
      }
    };
    fetchDepartments();
  }, [navigate]);

  const handleRegisterAgent = async (e) => {
    e.preventDefault();
    const token = localStorage.getItem('token');
    if (!token) return navigate('/admin/login');

    setError('');
    setSuccess('');

    try {
      await axios.post(
        'http://localhost:5000/api/admin/register-agent',
        { name, email, password, department, district, agentType }, // 👈 new fields
        { headers: { Authorization: `Bearer ${token}` } }
      );

      setSuccess('Agent registered successfully');
      setName('');
      setEmail('');
      setPassword('');
      setDepartment('');
      setDistrict('');
      setAgentType('');
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to register agent');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4">Register New Agent</h2>

        {error   && <p className="text-red-500 mb-4">{error}</p>}
        {success && <p className="text-green-500 mb-4">{success}</p>}

        <form onSubmit={handleRegisterAgent}>

          <div className="mb-4">
            <label className="block text-gray-700">Name</label>
            <input
              type="text"
              className="w-full p-2 border border-gray-300 rounded mt-1"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>

          <div className="mb-4">
            <label className="block text-gray-700">Email</label>
            <input
              type="email"
              className="w-full p-2 border border-gray-300 rounded mt-1"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="mb-4">
            <label className="block text-gray-700">Password</label>
            <input
              type="password"
              className="w-full p-2 border border-gray-300 rounded mt-1"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {/* Department Dropdown */}
          <div className="mb-4">
            <label className="block text-gray-700">Department</label>
            <select
              className="w-full p-2 border border-gray-300 rounded mt-1"
              value={department}
              onChange={(e) => setDepartment(e.target.value)}
              required
            >
              <option value="">Select Department</option>
              {departments.map((dept) => (
                <option key={dept._id} value={dept._id}>{dept.name}</option>
              ))}
            </select>
          </div>

          {/* District Dropdown 👇 */}
          <div className="mb-4">
            <label className="block text-gray-700">District</label>
            <select
              className="w-full p-2 border border-gray-300 rounded mt-1"
              value={district}
              onChange={(e) => setDistrict(e.target.value)}
              required
            >
              <option value="">Select District</option>
              {SRI_LANKA_DISTRICTS.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          {/* Agent Type Dropdown 👇 */}
          <div className="mb-6">
            <label className="block text-gray-700">Agent Type</label>
            <select
              className="w-full p-2 border border-gray-300 rounded mt-1"
              value={agentType}
              onChange={(e) => setAgentType(e.target.value)}
              required
            >
              <option value="">Select Agent Type</option>
              {AGENT_TYPES.map((type) => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
          >
            Register Agent
          </button>
        </form>

        <div className="mt-4">
          <button
            onClick={() => navigate('/admin/dashboard')}
            className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
};

export default CreateAgentPage;