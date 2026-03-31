#!/bin/bash

echo "🚀 Starting Rebar Counting Application..."

# Function to kill processes on exit
cleanup() {
    echo -e "\n🛑 Stopping all services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "✅ All services stopped."
    exit 0
}

trap cleanup SIGINT

# Free port 8000 if in use
PORT=8000
OCCUPIED=$(lsof -ti :$PORT)
if [ ! -z "$OCCUPIED" ]; then
    echo "⚠️ Port $PORT in use, killing..."
    kill -9 $OCCUPIED
fi

# Free port 5173 if in use
FRONTEND_PORT=5173
OCCUPIED_FRONT=$(lsof -ti :$FRONTEND_PORT)
if [ ! -z "$OCCUPIED_FRONT" ]; then
    echo "⚠️ Port $FRONTEND_PORT in use, killing..."
    kill -9 $OCCUPIED_FRONT
fi

# Start backend
cd ~/Desktop/Rebar-Counting-Product
source venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port $PORT &
BACKEND_PID=$!

# Build frontend production if not exists
cd ~/Desktop/Rebar-Counting-Product/frontend
if [ ! -d "dist" ]; then
    echo "⚡ Building frontend production..."
    npm install
    npm run build
fi

# Serve frontend
npx serve -s dist -l $FRONTEND_PORT &
FRONTEND_PID=$!

# Chromium auto-launch in kiosk mode after 10 sec
(sleep 10 && chromium-browser --kiosk --app=http://localhost:$FRONTEND_PORT) &

echo "✅ Both servers running! Press Ctrl+C to stop."
wait
