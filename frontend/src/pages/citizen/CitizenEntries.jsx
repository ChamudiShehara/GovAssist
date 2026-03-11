import React from "react";

const CitizenEntries = () => {
  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-6xl mx-auto bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6 text-center">Citizen Entries</h2>

        <div className="flex justify-center items-center h-64">
          <div className="bg-yellow-100 text-yellow-800 p-4 rounded-lg shadow-lg w-full text-center">
            <h3 className="text-xl font-semibold">Under Development</h3>
            <p className="mt-2 text-lg">This feature is currently under development. Please check back later.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CitizenEntries;