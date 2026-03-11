// CitizenDashboard.jsx
import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const CitizenDashboard = () => {
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  // Fetch complaints for the logged-in citizen
  useEffect(() => {
    const fetchComplaints = async () => {
      try {
        const token = localStorage.getItem("token");
        const response = await axios.get("http://localhost:5000/api/citizen/complaints", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setComplaints(response.data); // Assuming the response contains an array of complaints
      } catch (err) {
        setError("Failed to fetch complaints");
      } finally {
        setLoading(false);
      }
    };

    fetchComplaints();
  }, []);

  // Redirect to create a new complaint page
  const handleNewComplaint = () => {
    navigate("/citizen/create-complaint");
  };

  // Delete a complaint
  const handleDeleteComplaint = async (id) => {
    const token = localStorage.getItem("token");

    try {
      await axios.delete(`http://localhost:5000/api/citizen/complaints/${id}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      setComplaints(complaints.filter((complaint) => complaint._id !== id));
    } catch (err) {
      setError("Failed to delete complaint");
    }
  };

  // Update complaint status (e.g., change to "IN_PROGRESS" or "RESOLVED")
  const handleUpdateStatus = async (id, status) => {
    const token = localStorage.getItem("token");

    try {
      await axios.put(
        `http://localhost:5000/api/citizen/complaints/${id}`,
        { status },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      setComplaints(
        complaints.map((complaint) =>
          complaint._id === id ? { ...complaint, status } : complaint
        )
      );
    } catch (err) {
      setError("Failed to update status");
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-6xl mx-auto bg-white rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold">Citizen Dashboard</h2>
          <button
            onClick={handleNewComplaint}
            className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
          >
            New Complaint
          </button>
        </div>

        {/* Error message */}
        {error && <div className="text-red-500 mb-4">{error}</div>}

        {/* Complaints Table */}
        {loading ? (
          <p className="text-center text-gray-500">Loading...</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full table-auto border-collapse">
              <thead className="bg-gray-200">
                <tr>
                  <th className="border-b p-2 text-left">Complaint Title</th>
                  <th className="border-b p-2 text-left">Department</th>
                  <th className="border-b p-2 text-left">Status</th>
                  <th className="border-b p-2 text-left">Actions</th>
                </tr>
              </thead>
              <tbody>
                {complaints.length === 0 ? (
                  <tr>
                    <td colSpan="4" className="text-center p-4">No complaints found</td>
                  </tr>
                ) : (
                  complaints.map((complaint) => (
                    <tr key={complaint._id}>
                      <td className="border-b p-2">{complaint.title}</td> {/* Complaint Title */}
                      <td className="border-b p-2">{complaint.department.name}</td> {/* Department Name */}
                      <td className="border-b p-2">
                        <span
                          className={`${
                            complaint.status === "PENDING"
                              ? "text-yellow-500"
                              : complaint.status === "IN_PROGRESS"
                              ? "text-blue-500"
                              : "text-green-500"
                          } font-semibold`}
                        >
                          {complaint.status}
                        </span>
                      </td>
                      <td className="border-b p-2">
                        <div className="flex space-x-2">
                          <button
                            onClick={() => navigate(`/citizen/complaint/${complaint._id}`)}
                            className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
                          >
                            View Details
                          </button>

                          <button
                            onClick={() => handleDeleteComplaint(complaint._id)}
                            className="bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-600"
                          >
                            Delete
                          </button>
                        </div>
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

export default CitizenDashboard;