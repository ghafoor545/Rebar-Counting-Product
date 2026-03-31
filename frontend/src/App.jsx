// src/App.jsx
import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  NavLink,
} from "react-router-dom";

import "./App.css";

import CameraPage from "./pages/CameraPage";
import UploadPage from "./pages/UploadPage";
import ResultPage from "./pages/ResultPage";

// Import logo sitting at frontend/NUTECH_logo.png
import logo from "../NUTECH_logo.png";

const DEFAULT_USER = { id: 1, name: "Guest" };

function AppShell() {
  return (
    <Router>
      <div className="app-root">
        <div className="dashboard-wrap">
          <div className="navbar">
            <NavLink
              to="/camera"
              className={({ isActive }) =>
                isActive ? "btn nav-btn active" : "btn nav-btn"
              }
            >
              Camera
            </NavLink>
            <NavLink
              to="/upload"
              className={({ isActive }) =>
                isActive ? "btn nav-btn active" : "btn nav-btn"
              }
            >
              Upload
            </NavLink>
            <NavLink
              to="/results"
              className={({ isActive }) =>
                isActive ? "btn nav-btn active" : "btn nav-btn"
              }
            >
              Results
            </NavLink>
          </div>
        </div>

        <div className="dashboard-wrap">
          <div className="card hero-card">
            <div className="hero-logo">
              <img src={logo} alt="NUTECH logo" className="hero-logo-img" />
              <h1 className="gradient-heading hero-title">Rebar-Counting</h1>
            </div>
            <p>Experience seamless live monitoring with Bundle Detection.</p>
          </div>
        </div>

        <Routes>
          <Route path="/" element={<CameraPage user={DEFAULT_USER} />} />
          <Route path="/camera" element={<CameraPage user={DEFAULT_USER} />} />
          <Route path="/upload" element={<UploadPage user={DEFAULT_USER} />} />
          <Route path="/results" element={<ResultPage user={DEFAULT_USER} />} />
        </Routes>
      </div>
    </Router>
  );
}

export default AppShell;
