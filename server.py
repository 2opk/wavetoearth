from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import duckdb
import logging
import uvicorn
import os
import wavetoearth_core

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wavetoearth.server")

class WaveformDatabase:
    def __init__(self):
        self.conn = duckdb.connect(":memory:") # Use in-memory DuckDB, or persistent file
        self.loaded_file: Optional[str] = None
        self.parquet_path: Optional[str] = None
        self.rust_parser = wavetoearth_core.WaveParser()

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # Check/Create Parquet Cache
        parquet_path = path + ".parquet"

        if not os.path.exists(parquet_path):
             logger.info(f"Converting {path} to Parquet (High-Performance Rust)...")
             # chunk_size 1,000,000 is reasonable
             if path.endswith(".fst"):
                 self.rust_parser.convert_fst_to_parquet(path, parquet_path, 1_000_000)
             else:
                 self.rust_parser.convert_to_parquet(path, parquet_path, 1_000_000)
             logger.info("Conversion complete.")
        else:
             logger.info(f"Using cached Parquet: {parquet_path}")

        self.parquet_path = parquet_path

        # Load into DuckDB
        # We replace the view 'wave'
        self.conn.execute("DROP VIEW IF EXISTS wave")
        self.conn.execute(f"CREATE VIEW wave AS SELECT * FROM read_parquet('{parquet_path}')")

        self.loaded_file = path
        logger.info(f"Loaded {path} into DuckDB successfully.")

    def get_signal_names(self, pattern: Optional[str] = None) -> List[str]:
        if not self.loaded_file:
            return []

        # Query distinct signal names
        query = "SELECT DISTINCT signal_name FROM wave"
        if pattern:
             # DuckDB 'REGEXP_MATCHES'
             query += f" WHERE regexp_matches(signal_name, '{pattern}')"

        res = self.conn.execute(query).fetchall()
        return [r[0] for r in res]

    def query(self, signal: str, start: int, end: int) -> Dict[str, Any]:
        """
        Query signal values within time range [start, end].
        Returns list of (time, value).
        """
        if not self.loaded_file:
            raise HTTPException(status_code=400, detail="No file loaded")

        # DuckDB Query
        # We want changes in range [start, end]
        # AND strictly speaking, the initial value at 'start'.

        # 1. Get initial value (Last change BEFORE start)
        initial_res = self.conn.execute(
            "SELECT value FROM wave WHERE signal_name = ? AND timestamp < ? ORDER BY timestamp DESC LIMIT 1",
            [signal, start]
        ).fetchone()

        initial_value = None
        if initial_res:
             initial_value = initial_res[0]

        # 2. Get changes in range
        rows = self.conn.execute(
            "SELECT timestamp, value FROM wave WHERE signal_name = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
             [signal, start, end]
        ).fetchall()

        changes = []
        for r in rows:
            changes.append((r[0], r[1]))

        return {
            "signal": signal,
            "start": start,
            "end": end,
            "initial_value": initial_value,
            "changes": changes
        }

db = WaveformDatabase()

app = FastAPI(title="WaveToEarth Server")

class LoadRequest(BaseModel):
    file_path: str

class QueryRequest(BaseModel):
    signal: str
    start: int
    end: int

@app.get("/health")
def health():
    return {"status": "ok", "loaded_file": db.loaded_file}

@app.post("/load")
def load_file(req: LoadRequest, background_tasks: BackgroundTasks):
    try:
        db.load(req.file_path)
        # Get signal count
        count = db.conn.execute("SELECT COUNT(DISTINCT signal_name) FROM wave").fetchone()[0]
        return {"status": "loaded", "file": req.file_path, "signal_count": count}
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals")
def list_signals(pattern: Optional[str] = None):
    return {"signals": db.get_signal_names(pattern)}

@app.post("/query")
def query_signal(req: QueryRequest):
    return db.query(req.signal, req.start, req.end)

from core.semantic_analyzer import SemanticAnalyzer

@app.post("/analyze")
def analyze_signal(req: QueryRequest):
    """
    Semantic analysis of a signal.
    """
    if not db.loaded_file:
        raise HTTPException(status_code=400, detail="No file loaded")

    # Simply reuse query output
    raw = db.query(req.signal, req.start, req.end)
    # The analyze logic expects (times, values) lists.
    # Reformat
    times = [c[0] for c in raw['changes']]
    values = [c[1] for c in raw['changes']]

    summary = SemanticAnalyzer.summarize_activity(req.signal, times, values, req.start, req.end)
    return summary


def start_server(port=8000):
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    start_server()
