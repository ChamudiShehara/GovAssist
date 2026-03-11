import React from 'react';
import { Route, Routes } from 'react-router-dom';
import AdminLoginPage from './pages/admin/AdminLoginPage';
import AdminRegisterPage from './pages/admin/AdminRegisterPage';
import AdminDashboard from './pages/admin/AdminDashboard'; 
import CreateDepartmentPage from './pages/admin/CreateDepartmentPage'; 
import UserManagementPage from './pages/admin/UserManagementPage'; // Import User Management page
import CreateMinisterPage from './pages/admin/CreateMinisterPage'; // Import Create Minister page
import CreateAgentPage from './pages/admin/CreateAgentPage'; // Import Create Agent page
import HomePage from './pages/citizen/HomePage';  
import CitizenLoginPage from './pages/citizen/CitizenLoginPage';  
import CitizenRegisterPage from './pages/citizen/CitizenRegisterPage';  
import CitizenDashboard from './pages/citizen/CitizenDashboard';  
import MinisterLoginPage from './pages/minister/MinisterLoginPage';  // Import Minister Login page
import MinisterDashboard from './pages/minister/MinisterDashboard';  // Import Minister Dashboard page
import AgentLoginPage from './pages/agent/AgentLoginPage';  // Import Agent Login page
import AgentDashboard from './pages/agent/AgentDashboard';  // Import Agent Dashboard page
import Navbar from './components/Navbar'; 
import AdminHomePage from './pages/admin/AdminHomePage';  // Import Admin Home page
import CreateComplaint from './pages/citizen/CreateComplaint'; // Import Create Complaint page
import CitizenDepartments from './pages/citizen/CitizenDepartments'; // Import the new CitizenDepartments page
import DepartmentContent from './pages/citizen/DepartmentContent'; // Import the DepartmentContent page
import CitizenEntries from './pages/citizen/CitizenEntries';  // Import the CitizenEntries page
import AgentComplaintsPage from './pages/agent/AgentComplaintsPage';
import MinisterComplaintsPage   from './pages/minister/MinisterComplaintsPage';
import MinisterDepartmentPage   from './pages/minister/MinisterDepartmentPage';

function App() {
  return (
    <div className="App">
      <Navbar /> {/* Navbar component */}

      <Routes>
        {/* Home page route */}
        <Route path="/" element={<HomePage />} />

        {/* Admin Routes */}
        <Route path="/admin/login" element={<AdminLoginPage />} />
        <Route path="/admin/register" element={<AdminRegisterPage />} />
        <Route path="/admin/dashboard" element={<AdminDashboard />} />
        <Route path="/admin/create-department" element={<CreateDepartmentPage />} />
        <Route path="/admin/user-management" element={<UserManagementPage />} /> {/* User Management page */}
        <Route path="/admin/create-minister" element={<CreateMinisterPage />} /> {/* Create Minister page */}
        <Route path="/admin/create-agent" element={<CreateAgentPage />} /> {/* Create Agent page */}
        
        {/* Admin Home Page */}
        <Route path="/admin/home" element={<AdminHomePage />} />
        

        {/* Citizen Routes */}
        <Route path="/citizen/login" element={<CitizenLoginPage />} />
        <Route path="/citizen/register" element={<CitizenRegisterPage />} />
        <Route path="/citizen/dashboard" element={<CitizenDashboard />} />
        <Route path="/citizen/create-complaint" element={<CreateComplaint />} /> {/* Add this route */}
        <Route path="/citizen/departments" element={<CitizenDepartments />} /> {/* Route for CitizenDepartments page */}
        <Route path="/citizen/departments/:departmentId" element={<DepartmentContent />} /> {/* Route for Department Content */}
        <Route path="/citizen/entries" element={<CitizenEntries />} /> {/* Route for Citizen Entries Page */}

        {/* Minister Routes */}
        <Route path="/minister/login" element={<MinisterLoginPage />} />  {/* Minister login page */}
        <Route path="/minister/dashboard" element={<MinisterDashboard />} />  {/* Minister dashboard page */}
        <Route path="/minister/dashboard"   element={<MinisterDashboard />} />
        <Route path="/minister/complaints"  element={<MinisterComplaintsPage />} />
        <Route path="/minister/departments" element={<MinisterDepartmentPage />} />

        {/* Agent Routes */}
        <Route path="/agent/login" element={<AgentLoginPage />} />  {/* Agent login page */}
        <Route path="/agent/dashboard" element={<AgentDashboard />} />  {/* Agent dashboard page */}
        <Route path="/agent/complaints" element={<AgentComplaintsPage />} />
        
      </Routes>
    </div>
  );
}

export default App;