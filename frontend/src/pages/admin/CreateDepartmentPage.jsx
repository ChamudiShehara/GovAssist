import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const CreateDepartmentPage = () => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const navigate = useNavigate();

  const handleCreateDepartment = async (e) => {
    e.preventDefault();
    const token = localStorage.getItem('token');
    
    if (!token) {
      return navigate('/admin/login');  // Redirect to login if no token
    }

    try {
      // Send request to the backend to create a department
      const response = await axios.post(
        'http://localhost:5000/api/admin/create-department',
        {
          name,
          description,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,  // Send the token in Authorization header
          },
        }
      );

      setSuccess('Department created successfully');
      setName('');
      setDescription('');
    } catch (err) {
      setError('Failed to create department');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4">Create Department</h2>

        {/* Display error or success message */}
        {error && <p className="text-red-500">{error}</p>}
        {success && <p className="text-green-500">{success}</p>}

        <form onSubmit={handleCreateDepartment}>
          <div className="mb-4">
            <label className="block text-gray-700">Department Name</label>
            <input
              type="text"
              className="w-full p-2 border border-gray-300 rounded mt-1"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>

          <div className="mb-4">
            <label className="block text-gray-700">Description</label>
            <textarea
              className="w-full p-2 border border-gray-300 rounded mt-1"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              required
            ></textarea>
          </div>

          <button
            type="submit"
            className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
          >
            Create Department
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

export default CreateDepartmentPage;