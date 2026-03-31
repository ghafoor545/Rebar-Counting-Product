# Rebar Counting Product

**Automatic Rebar Detection & Counting System**  
A full-stack application for counting steel reinforcement bars (rebars) in construction using computer vision. Supports both uploaded images and live capture from OAK cameras (DepthAI), with post-processing via clustering, storage in PostgreSQL, and Excel report export.

## Features
- Image upload or live feed from OAK camera (DepthAI)  
- Rebar detection using ONNX-exported model  
- Bundle grouping & accurate counting with DBSCAN clustering  
- Results saved to PostgreSQL database  
- Export counts and details to Excel (.xlsx)  
- FastAPI backend + modern frontend (React + Vite)  
- Environment configuration via `.env`  

## Tech Stack
**Backend**  
- Python 3.11+  
- FastAPI + Uvicorn  
- ONNX Runtime (model inference)  
- DepthAI (OAK camera support)  
- OpenCV, Pillow, NumPy  
- scikit-learn (DBSCAN)  
- psycopg2-binary (PostgreSQL)  
- openpyxl (Excel export)  

**Frontend**  
- React  
- Vite  

**Database**  
- PostgreSQL  

## Project Structure
Rebar Counting Product/
├── backend/                  # FastAPI application
│   ├── main.py               # App entry + routes + startup
│   ├── services/             # Business logic
│   │   ├── detector.py
│   │   ├── oak_utils.py
│   │   └── ...
│   ├── api/                  # Route modules
│   │   ├── auth_routes.py
│   │   └── detection_routes.py
│   └── db.py                 # Database connection & init
├── frontend/                 # React + Vite frontend
│   ├── src/
│   ├── public/
│   └── vite.config.js
├── venv/                     # Python virtual environment (gitignored)
├── .env                      # Environment variables (gitignored)
├── requirement.txt           # Dependencies (consider renaming → requirements.txt)
└── README.md





## Setup (Windows)

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm
- PostgreSQL 15+ (running locally)
- (Optional) OAK camera (OAK-D / OAK-1 etc.) + DepthAI drivers

### Environment configuration

Create a `.env` file in project root. Two database modes are supported:

- PostgreSQL (default):
  - `DB_DRIVER=postgres`
  - `PGHOST=localhost`
  - `PGPORT=5432`
  - `PGDATABASE=rebar_db`
  - `PGUSER=rebar_user`
  - `PGPASSWORD=<your_password>`

- SQLite (local development, no PostgreSQL required):
  - `DB_DRIVER=sqlite`
  - `DB_PATH=data/app.db`

If using PostgreSQL, ensure the DB+user exist:

```sql
CREATE ROLE rebar_user WITH LOGIN PASSWORD '<your_password>';
CREATE DATABASE rebar_db OWNER rebar_user;
GRANT ALL PRIVILEGES ON DATABASE rebar_db TO rebar_user;
```

### Virtual Environment Setup

**Windows**  
Create: `python -m venv venv`  
Activate (PowerShell): `.\venv\Scripts\Activate.ps1`  
Activate (cmd): `venv\Scripts\activate.bat`

**Linux / macOS**  
Create: `python3 -m venv venv`  
Activate: `source venv/bin/activate`

Deactivate (both): `deactivate`

After activation: `pip install -r requirements.txt`


3. Frontend Setup

cd frontend
npm install


4. Run the Application
Terminal 1 – Backend

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000


Terminal 2 – Frontend

cd frontend
npm run dev

API docs: http://localhost:8000/docs
Web app: http://localhost:5173 (default Vite port)


Usage

Open the frontend in browser
Upload rebar image or connect live OAK camera
View detection results, count, and export to Excel



