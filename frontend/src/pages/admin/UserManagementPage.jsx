import React from 'react';
import { Link } from 'react-router-dom';

const UserManagementPage = () => {
  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4">User Management</h2>
        <p className="text-lg mb-4">Manage ministers, agents, and other users in the system</p>

        {/* Button to navigate to Create Minister page */}
        <Link
          to="/admin/create-minister"  // Navigate to Create Minister page
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 mb-4 inline-block"
        >
          Register New Minister
        </Link>

        {/* Button to navigate to Create Agent page */}
        <Link
          to="/admin/create-agent"  // Navigate to Create Agent page
          className="bg-yellow-500 text-white px-4 py-2 rounded hover:bg-yellow-600 inline-block"
        >
          Register New Agent
        </Link>

        {/* Add more user management actions here (like view list of ministers) */}
      </div>
    </div>
  );
};

export default UserManagementPage;