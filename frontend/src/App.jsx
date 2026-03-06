// src/App.jsx
import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  NavLink,
  Navigate,
} from "react-router-dom";

import "./App.css";

import { AuthProvider, useAuth } from "./context/AuthContext";

import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import CameraPage from "./pages/CameraPage";
import UploadPage from "./pages/UploadPage";
import ResultPage from "./pages/ResultPage";

// Import logo sitting at frontend/NUTECH_logo.png
import logo from "../NUTECH_logo.png";

function PrivateRoute({ children }) {
  const { user } = useAuth();
  return user ? children : <Navigate to="/" replace />;
}

function AppShell() {
  const { user, logout } = useAuth();

  return (
    <Router>
      <div className="app-root">
        {/* Navbar + hero only when logged in */}
        {user && (
          <>
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
                <button
                  type="button"
                  className="btn nav-btn"
                  onClick={logout}
                >
                  Logout
                </button>
              </div>
            </div>

            {/* Hero with logo + app name */}
            <div className="dashboard-wrap">
              <div className="card hero-card">
                <div className="hero-logo">
                  <img
                    src={logo}
                    alt="NUTECH logo"
                    className="hero-logo-img"
                  />
                  <h1 className="gradient-heading hero-title">
                    Rebar-Counting
                  </h1>
                </div>
                <p>Experience seamless live monitoring with Bundle Detection.</p>
              </div>
            </div>
          </>
        )}

        <Routes>
          {/* Root route: login or redirect to /camera if already logged in */}
          <Route
            path="/"
            element={
              user ? <Navigate to="/camera" replace /> : <LoginPage />
            }
          />

          {/* Public signup */}
          <Route path="/register" element={<RegisterPage />} />

          {/* Protected routes */}
          <Route
            path="/camera"
            element={
              <PrivateRoute>
                <CameraPage user={user} />
              </PrivateRoute>
            }
          />
          <Route
            path="/upload"
            element={
              <PrivateRoute>
                <UploadPage user={user} />
              </PrivateRoute>
            }
          />
          <Route
            path="/results"
            element={
              <PrivateRoute>
                <ResultPage user={user} />
              </PrivateRoute>
            }
          />
        </Routes>
      </div>
    </Router>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  );
}