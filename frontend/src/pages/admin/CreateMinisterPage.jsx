import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const CreateMinisterPage = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [department, setDepartment] = useState('');
  const [departments, setDepartments] = useState([]); // For storing available departments
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const navigate = useNavigate();

  // Fetch the available departments when the page loads
  useEffect(() => {
    const fetchDepartments = async () => {
      const token = localStorage.getItem('token'); // Get admin JWT token from localStorage

      if (!token) {
        return navigate('/admin/login'); // Redirect to login if no token
      }

      try {
        const response = await axios.get('http://localhost:5000/api/admin/departments', {
          headers: {
            Authorization: `Bearer ${token}`, // Send the admin token in the Authorization header
          },
        });
        setDepartments(response.data); // Set the fetched departments to the state
      } catch (err) {
        setError('Failed to fetch departments');
      }
    };

    fetchDepartments();
  }, [navigate]);

  // Handle minister registration
  const handleRegisterMinister = async (e) => {
    e.preventDefault();
    const token = localStorage.getItem('token'); // Get admin JWT token from localStorage

    if (!token) {
      return navigate('/admin/login'); // Redirect to login if no token
    }

    try {
      const response = await axios.post(
        'http://localhost:5000/api/admin/register-minister',
        {
          name,
          email,
          password,
          department,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`, // Send the admin token in the Authorization header
          },
        }
      );

      setSuccess('Minister registered successfully');
      setName('');
      setEmail('');
      setPassword('');
      setDepartment('');
    } catch (err) {
      setError('Failed to register minister');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4">Register Minister</h2>
        
        {/* Display error or success message */}
        {error && <p className="text-red-500">{error}</p>}
        {success && <p className="text-green-500">{success}</p>}

        <form onSubmit={handleRegisterMinister}>
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

          <div className="mb-6">
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
                <option key={dept._id} value={dept._id}>
                  {dept.name}
                </option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
          >
            Register Minister
          </button>
        </form>

        {/* Button to go back to admin dashboard */}
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

export default CreateMinisterPage;