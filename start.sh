#!/bin/bash

# Start both backend and frontend
echo "🚀 Starting Rebar Counting Application..."

# Function to kill processes on exit
cleanup() {
    echo "\n🛑 Stopping all services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# Start backend
echo "📦 Starting backend server..."
cd ~/Desktop/Rebar-Counting-Product
source venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start frontend
echo "🎨 Starting frontend server..."
cd ~/Desktop/Rebar-Counting-Product/frontend
npm run dev &
FRONTEND_PID=$!

echo "✅ Both servers are running!"
echo "📱 Frontend: http://localhost:5173"
echo "🔧 Backend API: http://localhost:8000/docs"
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
wait
