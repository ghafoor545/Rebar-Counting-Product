import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const API = "http://localhost:8000";

export default function RegisterPage() {
  const [form, setForm] = useState({
    username: "",
    email: "",
    password: "",
  });
  const [error, setError] = useState(null);

  const navigate = useNavigate();

  const handleChange = (e) =>
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    try {
      const res = await fetch(`${API}/auth/users`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Registration failed");
      }

      // On success, go back to login
      navigate("/");
    } catch (err) {
      setError(err.message || "Registration failed");
    }
  };

  return (
    <div className="fullpage-center">
      <div className="auth-card">
        <div className="auth-card-header">
          <div className="auth-icon">📝</div>
          <h2 className="auth-title gradient-heading">Create Account</h2>
          <p className="auth-subtitle">
            Sign up to start using the Rebar Counting dashboard.
          </p>
        </div>

        {error && <div className="alert error">{error}</div>}

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="auth-label">Username</label>
          <input
            className="auth-input"
            name="username"
            placeholder="Your username"
            value={form.username}
            onChange={handleChange}
            required
          />

          <label className="auth-label">Email</label>
          <input
            className="auth-input"
            name="email"
            type="email"
            placeholder="you@example.com"
            value={form.email}
            onChange={handleChange}
            required
          />

          <label className="auth-label">Password</label>
          <input
            className="auth-input"
            name="password"
            type="password"
            placeholder="Create a password"
            value={form.password}
            onChange={handleChange}
            required
          />

          <div className="auth-actions-row">
            <button type="submit" className="btn auth-primary-btn">
              <span className="auth-btn-icon">📝</span>
              <span>Sign Up</span>
            </button>
            <button
              type="button"
              className="btn auth-secondary-btn"
              onClick={() => navigate("/")}
            >
              Back to Sign In
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}