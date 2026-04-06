import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  NavLink,
  Navigate,
} from "react-router-dom";

import "./App.css";

import CameraPage from "./pages/CameraPage";
import UploadPage from "./pages/UploadPage";
import ResultPage from "./pages/ResultPage";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";

import { useAuth } from "./context/AuthContext";

// Import logo sitting at frontend/NUTECH_logo.png
import logo from "../NUTECH_logo.png";

function PrivateRoute({ children }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

function AppShell() {
  const { user, logout } = useAuth();

  return (
    <Router>
      <div className="app-root">
        {/* NAVBAR */}
        <div className="dashboard-wrap">
          <div className="navbar">
            {user ? (
              <>
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

                <button className="btn nav-btn" type="button" onClick={logout}>
                  Logout
                </button>
              </>
            ) : (
              <>
                <NavLink
                  to="/login"
                  className={({ isActive }) =>
                    isActive ? "btn nav-btn active" : "btn nav-btn"
                  }
                >
                  Login
                </NavLink>
                <NavLink
                  to="/register"
                  className={({ isActive }) =>
                    isActive ? "btn nav-btn active" : "btn nav-btn"
                  }
                >
                  Register
                </NavLink>
              </>
            )}
          </div>
        </div>

        {/* HERO */}
        <div className="dashboard-wrap">
          <div className="card hero-card">
            <div className="hero-logo">
              <img src={logo} alt="NUTECH logo" className="hero-logo-img" />
              <h1 className="gradient-heading hero-title">Rebar-Counting</h1>
            </div>
            <p>Experience seamless live monitoring with Bundle Detection.</p>
            {user && (
              <p className="muted" style={{ marginTop: 6 }}>
                Signed in as: <strong>{user.username || user.email || `User ${user.id}`}</strong>
              </p>
            )}
          </div>
        </div>

        {/* ROUTES */}
        <Routes>
          {/* Default route */}
          <Route
            path="/"
            element={user ? <Navigate to="/camera" replace /> : <Navigate to="/login" replace />}
          />

          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

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

          {/* fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default AppShell;