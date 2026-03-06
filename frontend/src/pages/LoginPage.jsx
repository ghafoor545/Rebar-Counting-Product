// src/pages/LoginPage.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

// adjust if your backend URL is different
const API = "http://localhost:8000";

// Import logo (same file used in App.jsx)
// App.jsx is in src/, using "../NUTECH_logo.png"
// LoginPage.jsx is in src/pages/, so go one level higher: ../../
import logo from "../../NUTECH_logo.png";

export default function LoginPage() {
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState(null);

  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    try {
      const res = await fetch(`${API}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identifier, password }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Login failed");
      }

      const data = await res.json();
      login(data.user, remember);
      navigate("/camera");
    } catch (err) {
      setError(err.message || "Login failed");
    }
  };

  return (
    <div className="fullpage-center">
      <div className="auth-card">
        <div className="auth-card-header">
          {/* ✅ Logo above app name */}
          <div className="auth-logo-wrap">
            <img
              src={logo}
              alt="NUTECH logo"
              className="auth-logo-img"
            />
          </div>

          <h2 className="auth-title gradient-heading">
            SMART REBAR COUNTING SYSTEM
          </h2>
          <p className="auth-subtitle">
            Secure login to access your dashboard.
          </p>
        </div>

        {error && <div className="alert error">{error}</div>}

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="auth-label">Username / Email</label>
          <input
            className="auth-input"
            type="text"
            placeholder="you@example.com"
            value={identifier}
            onChange={(e) => setIdentifier(e.target.value)}
            required
          />

          <label className="auth-label">Password</label>
          <div className="auth-input-row">
            <input
              className="auth-input"
              type={showPassword ? "text" : "password"}
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <button
              type="button"
              className="auth-eye-btn"
              onClick={() => setShowPassword((v) => !v)}
            >
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>

          <div className="auth-options-row">
            <label className="remember-me">
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
              />
              <span>Remember Me</span>
            </label>
            <button
              type="button"
              className="link-button"
              onClick={() =>
                alert("Forgot password flow is not implemented yet.")
              }
            >
              Forgot Password?
            </button>
          </div>

          <div className="auth-actions-row">
            <button type="submit" className="btn auth-primary-btn">
              <span className="auth-btn-icon">🔐</span>
              <span>Sign In</span>
            </button>
            <button
              type="button"
              className="btn auth-secondary-btn"
              onClick={() => navigate("/register")}
            >
              Sign Up
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}