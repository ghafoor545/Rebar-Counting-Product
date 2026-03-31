# Rebar Counting Product

**Automatic Rebar Detection & Counting System**

A full-stack application to detect and count reinforcement bars (rebars) from images and live OAK camera feed.

## Features

- Upload image or live camera input (OAK-D / DepthAI)
- Bundle grouping with DBSCAN
- Distance-based smart count logic (nearest-only vs all bundles)
- Per-bundle and per-rebar annotation with ID labels
- Ignore uncounted bundles and show status
- Save detection results to PostgreSQL
- Frontend dashboard (React + Vite) with gallery

## Tech Stack

- Python 3.11+
- FastAPI + Uvicorn
- ONNX Runtime
- OpenCV, NumPy, scikit-learn
- PostgreSQL (or local SQLite fallback)
- React + Vite

## Project Structure

- `backend/`
  - `main.py`: FastAPI app and route setup
  - `api/`: routes (`detection_routes.py`, `auth_routes.py`)
  - `services/`: detection logic (`detector.py`, `oak_utils.py`)
  - `db.py`: DB connection
- `frontend/`: React app
  - `src/pages/CameraPage.jsx`: live capture + bundle details
  - `src/components/Upload.jsx`: image upload + summary
  - `src/pages/UploadPage.jsx`: history gallery
- `data/`: generated outputs (images and detection logs)
- `requirements.txt`: Python dependencies

## Setup

1. Clone repo
```bash
git clone <repo-url>
cd Rebar-Counting-Product
```
2. Create `.env` (example):
```ini
DB_DRIVER=postgres
PGHOST=localhost
PGPORT=5432
PGDATABASE=rebar_db
PGUSER=rebar_user
PGPASSWORD=secret
```
3. PostgreSQL setup:
```sql
CREATE ROLE rebar_user WITH LOGIN PASSWORD 'secret';
CREATE DATABASE rebar_db OWNER rebar_user;
GRANT ALL PRIVILEGES ON DATABASE rebar_db TO rebar_user;
```
4. Python env + packages:
```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# or cmd
.\venv\Scripts\activate.bat

pip install -r requirements.txt
```
5. Frontend setup:
```bash
cd frontend
npm install
npm run build # or npm run dev
```

## Run

- Backend:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
- Frontend:
```bash
cd frontend
npm run dev
```

Open:
- API docs: `http://localhost:8000/docs`
- Web app: `http://localhost:5173`

## Testing & Validation

- `python -m py_compile backend/main.py`
- `python -m py_compile backend/services/detector.py`
- `npm run build`

## Behavior Notes

- If no depth data: capture mode falls back to nearest-only counting (no distance needed)
- Counted bundles are color-annotated with per-rebar labels
- Ignored bundles are shown in gray with explicit `IGNORED` status
- Upload results annotate counted rebars clearly and avoid noise

## Troubleshooting

- If OAK stream fails, ensure device connected and DepthAI installed
- If distances show `unknown`, it means missing depth data
- Watch console for max distance difference and mode decisions

## Contributing

1. Fork -> clone
2. Create branch
3. Implement feature/bugfix
4. `git add`, `git commit`, `git push`
5. Open PR with details

## License

MIT / internal project use
