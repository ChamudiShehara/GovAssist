import React from 'react';
import { useNavigate } from 'react-router-dom';

const AdminHomePage = () => {
  const navigate = useNavigate();

  const handleNavigate = (role) => {
    if (role === "admin") {
      navigate("/admin/login");
    } else if (role === "minister") {
      navigate("/minister/login");
    } else if (role === "agent") {
      navigate("/agent/login");
    } else if (role === "police") {
      navigate("/police/login");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 p-6">
      <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4 text-center">Welcome to the System</h2>
        
        {/* Tabs with buttons for each role */}
        <div className="grid grid-cols-2 gap-6">
          <div className="flex flex-col items-center">
            <div className="mb-4">
              <button 
                onClick={() => handleNavigate("minister")}
                className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
              >
                Minister Login
              </button>
            </div>
            <p className="text-center">Minister</p>
          </div>

          <div className="flex flex-col items-center">
            <div className="mb-4">
              <button 
                onClick={() => handleNavigate("agent")}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
              >
                Agent Login
              </button>
            </div>
            <p className="text-center">Agent</p>
          </div>

          <div className="flex flex-col items-center">
            <div className="mb-4">
              <button 
                onClick={() => handleNavigate("police")}
                className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
              >
                Police Login
              </button>
            </div>
            <p className="text-center">Police</p>
          </div>

          <div className="flex flex-col items-center">
            <div className="mb-4">
              <button 
                onClick={() => handleNavigate("admin")}
                className="bg-yellow-500 text-white px-4 py-2 rounded hover:bg-yellow-600"
              >
                Admin Login
              </button>
            </div>
            <p className="text-center">Admin</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminHomePage;