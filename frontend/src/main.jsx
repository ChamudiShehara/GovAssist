import React from 'react';
import ReactDOM from 'react-dom/client'; // Make sure to import from 'react-dom/client'
import './index.css'; // TailwindCSS styles
import { BrowserRouter as Router } from 'react-router-dom';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root')); // Create the root
root.render(
  <React.StrictMode>
    <Router>
      <App />
    </Router>
  </React.StrictMode>
);