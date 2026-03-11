import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';

const Navbar = () => {
  const [userName, setUserName] = useState('');
  const [userEmail, setUserEmail] = useState('');
  const [userRole, setUserRole] = useState('');
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const navigate = useNavigate();

  const fetchUserDetails = async () => {
    const token = localStorage.getItem("token");
    const role = localStorage.getItem("role");

    if (!token || !role) {
      setUserName('');
      setUserEmail('');
      setUserRole('');
      return;
    }

    const urlMap = {
      admin:    "http://localhost:5000/api/admin/profile",
      minister: "http://localhost:5000/api/minister/profile",
      agent:    "http://localhost:5000/api/agent/profile",
      citizen:  "http://localhost:5000/api/citizen/me",
    };

    const url = urlMap[role];
    if (!url) return;

    try {
      const response = await axios.get(url, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setUserName(response.data.name);
      setUserEmail(response.data.email);
      setUserRole(role);
    } catch (err) {
      console.error("Error fetching user details:", err);
    }
  };

  useEffect(() => {
    fetchUserDetails();
    window.addEventListener("authChange", fetchUserDetails);
    return () => window.removeEventListener("authChange", fetchUserDetails);
  }, []);

  const handleLogout = () => {
    const role = localStorage.getItem("role");

    localStorage.removeItem("token");
    localStorage.removeItem("role");
    window.dispatchEvent(new Event("authChange"));

    if (role === "admin")         navigate("/admin/login");
    else if (role === "minister") navigate("/minister/login");
    else if (role === "agent")    navigate("/agent/login");
    else                          navigate("/citizen/login");
  };

  const homeLink =
    userRole === "admin"    ? "/admin/dashboard" :
    userRole === "minister" ? "/minister/dashboard" :
    userRole === "agent"    ? "/agent/dashboard" :
    "/";

  const roleBadgeColor = {
    admin:    "bg-red-100 text-red-600",
    minister: "bg-purple-100 text-purple-600",
    agent:    "bg-green-100 text-green-600",
    citizen:  "bg-blue-100 text-blue-600",
  };

  return (
    <header className="bg-blue-600 text-white p-4 sticky top-0 z-50 shadow-lg">
      <div className="flex justify-between items-center">

        {/* Logo */}
        <h1 className="text-xl font-bold">GovConnect</h1>

        {/* Role-based nav links */}
        <nav className="space-x-4">

          <Link to={homeLink} className="hover:text-gray-200">Home</Link>

          {userRole === "admin" && (
            <>
              <Link to="/admin/dashboard" className="hover:text-gray-200">Dashboard</Link>
              <Link to="/admin/manage-users" className="hover:text-gray-200">Manage Users</Link>
            </>
          )}

          {userRole === "minister" && (
            <>
              <Link to="/minister/dashboard"   className="hover:text-gray-200">Dashboard</Link>
              <Link to="/minister/complaints"  className="hover:text-gray-200">Complaints</Link>
              <Link to="/minister/departments" className="hover:text-gray-200">Departments</Link>
            </>
          )}

          {userRole === "agent" && (
            <>
              <Link to="/agent/dashboard" className="hover:text-gray-200">Dashboard</Link>
              <Link to="/agent/complaints" className="hover:text-gray-200">Complaints</Link>
              <Link to="/agent/departments" className="hover:text-gray-200">Departments</Link>  {/* 👈 new */}
            </>
          )}

          {userRole === "citizen" && (
            <>
              <Link to="/citizen/dashboard" className="hover:text-gray-200">Dashboard</Link>
              <Link to="/citizen/create-complaint" className="hover:text-gray-200">Complaints</Link>
              <Link to="/citizen/departments" className="hover:text-gray-200">Departments</Link>
              <Link to="/citizen/entries" className="hover:text-gray-200">Entries</Link>
            </>
          )}

          {!userRole && (
            <>
              <Link to="/citizen/login" className="hover:text-gray-200">Login</Link>
              <Link to="/citizen/register" className="hover:text-gray-200">Register</Link>
            </>
          )}

        </nav>

        {/* Profile Dropdown */}
        <div className="relative">
          <button
            className="flex items-center space-x-2"
            onClick={() => setDropdownOpen(!dropdownOpen)}
          >
            <span className="font-semibold">{userName || "Guest"}</span>
            <span>🔽</span>
          </button>

          {dropdownOpen && (
            <div className="absolute right-0 mt-2 w-52 bg-white text-gray-800 shadow-lg rounded-lg p-2 z-50">
              {userName ? (
                <>
                  {/* User info */}
                  <div className="px-2 pb-2 border-b border-gray-100 mb-2">
                    <p className="font-semibold truncate">{userName}</p>
                    <p className="text-sm text-gray-500 truncate">{userEmail}</p>
                    <span className={`text-xs font-medium px-2 py-0.5 rounded-full mt-1 inline-block ${roleBadgeColor[userRole]}`}>
                      {userRole}
                    </span>
                  </div>

                  {/* Role-based dropdown actions */}
                  {userRole === "admin" && (
                    <button
                      className="block w-full text-left px-2 py-2 text-red-500 hover:bg-gray-100 rounded"
                      onClick={() => navigate("/admin/dashboard")}
                    >
                      My Dashboard
                    </button>
                  )}

                  {userRole === "minister" && (
                    <button
                      className="block w-full text-left px-2 py-2 text-purple-500 hover:bg-gray-100 rounded"
                      onClick={() => navigate("/minister/dashboard")}
                    >
                      My Dashboard
                    </button>
                  )}

                  {userRole === "agent" && (
                    <button
                      className="block w-full text-left px-2 py-2 text-green-500 hover:bg-gray-100 rounded"
                      onClick={() => navigate("/agent/dashboard")}
                    >
                      My Dashboard
                    </button>
                  )}

                  {userRole === "citizen" && (
                    <button
                      className="block w-full text-left px-2 py-2 text-blue-500 hover:bg-gray-100 rounded"
                      onClick={() => navigate("/citizen/update-profile")}
                    >
                      Update Profile
                    </button>
                  )}

                  <button
                    className="block w-full text-left px-2 py-2 text-red-500 hover:bg-gray-100 rounded"
                    onClick={handleLogout}
                  >
                    Logout
                  </button>
                </>
              ) : (
                <>
                  <button
                    className="block w-full text-left px-2 py-2 text-blue-500 hover:bg-gray-100 rounded"
                    onClick={() => navigate("/citizen/login")}
                  >
                    Login
                  </button>
                  <button
                    className="block w-full text-left px-2 py-2 text-green-500 hover:bg-gray-100 rounded"
                    onClick={() => navigate("/citizen/register")}
                  >
                    Register
                  </button>
                </>
              )}
            </div>
          )}
        </div>

      </div>
    </header>
  );
};

export default Navbar;