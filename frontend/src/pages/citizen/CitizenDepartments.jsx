import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const CitizenDepartments = () => {
  const [departments, setDepartments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  // Fetch departments when the component mounts
  useEffect(() => {
    const fetchDepartments = async () => {
      try {
        const token = localStorage.getItem("token");

        if (!token) {
          return navigate("/citizen/login"); // Redirect to login if no token
        }

        // Fetch departments from the backend
        const response = await axios.get("http://localhost:5000/api/admin/departments", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        setDepartments(response.data); // Set the fetched departments to the state
      } catch (err) {
        setError("Failed to fetch departments");
      } finally {
        setLoading(false);
      }
    };

    fetchDepartments();
  }, [navigate]);

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-6xl mx-auto bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6">Departments</h2>

        {/* Error message */}
        {error && <div className="text-red-500 mb-4">{error}</div>}

        {/* Departments Table */}
        {loading ? (
          <p className="text-center text-gray-500">Loading...</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full table-auto border-collapse">
              <thead className="bg-gray-200">
                <tr>
                  <th className="border-b p-2 text-left">Department Name</th>
                  <th className="border-b p-2 text-left">Description</th>
                  <th className="border-b p-2 text-left">Actions</th>
                </tr>
              </thead>
              <tbody>
                {departments.length === 0 ? (
                  <tr>
                    <td colSpan="3" className="text-center p-4">No departments available</td>
                  </tr>
                ) : (
                  departments.map((department) => (
                    <tr key={department._id}>
                      <td className="border-b p-2">{department.name}</td>
                      <td className="border-b p-2">{department.description}</td>
                      <td className="border-b p-2">
                        <button
                          className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
                          onClick={() => navigate(`/citizen/departments/${department._id}`)} // Navigate to department details page
                        >
                          Page
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default CitizenDepartments;