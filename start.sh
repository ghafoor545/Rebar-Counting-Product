#!/bin/bash

# 🚀 Start Rebar Counting Application

echo "🚀 Starting Rebar Counting Application..."

# --- Function to kill processes on exit ---
cleanup() {
    echo -e "\n🛑 Stopping all services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "✅ All services stopped."
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# --- Free port 8000 if in use ---
PORT=8000
OCCUPIED=$(lsof -ti :$PORT)
if [ ! -z "$OCCUPIED" ]; then
    echo "⚠️  Port $PORT is in use. Killing existing processes..."
    kill -9 $OCCUPIED
fi

# --- Free port 5173 for frontend ---
FRONTEND_PORT=5173
OCCUPIED_FRONT=$(lsof -ti :$FRONTEND_PORT)
if [ ! -z "$OCCUPIED_FRONT" ]; then
    echo "⚠️  Frontend port $FRONTEND_PORT is in use. Killing existing processes..."
    kill -9 $OCCUPIED_FRONT
fi

# --- Start backend ---
echo "📦 Starting backend server..."
cd ~/Desktop/Rebar-Counting-Product
source venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port $PORT &
BACKEND_PID=$!

# --- Build frontend production (optional, first time) ---
cd ~/Desktop/Rebar-Counting-Product/frontend
if [ ! -d "dist" ]; then
    echo "⚡ Building frontend production..."
    npm install
    npm run build
fi

# --- Serve frontend production build ---
echo "🎨 Serving frontend production build..."
npx serve -s dist -l $FRONTEND_PORT &
FRONTEND_PID=$!

echo "✅ Both servers are running!"
echo "📱 Frontend: http://localhost:$FRONTEND_PORT"
echo "🔧 Backend API: http://localhost:$PORT/docs"
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
wait