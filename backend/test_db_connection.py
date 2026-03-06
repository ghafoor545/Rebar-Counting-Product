# test_db_connection.py
from backend.db import get_conn

conn = get_conn()
cur = conn.cursor()
cur.execute("SELECT version();")
print(cur.fetchone())
conn.close()