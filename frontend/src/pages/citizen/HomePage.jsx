import React from 'react';
import { Link } from 'react-router-dom'; // Import Link from react-router-dom

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gray-100 p-6 flex flex-col items-center justify-center">
      <div className="w-full max-w-lg bg-white p-8 rounded-lg shadow-lg text-center">
        <h1 className="text-3xl font-bold mb-4">Welcome to GovConnect</h1>
        <p className="text-lg text-gray-700 mb-6">
          GovConnect is a platform that allows citizens to report complaints directly to government departments. 
          Stay informed and help improve services in your community. Easily register, login, and submit your concerns.
        </p>

        {/* Buttons to navigate to login and register pages */}
        <div className="space-x-4">
          <Link
            to="/citizen/login"
            className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
          >
            Citizen Login
          </Link>
          <Link
            to="/citizen/register"
            className="bg-green-500 text-white py-2 px-4 rounded hover:bg-green-600"
          >
            Citizen Register
          </Link>
        </div>
      </div>
    </div>
  );
}