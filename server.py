from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import duckdb
import logging
import uvicorn
import os
import glob
import re
import wavetoearth_core
from core.semantic_analyzer import SemanticAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wavetoearth.server")

class WaveformDatabase:
    def __init__(self):
        self.conn = duckdb.connect(":memory:") # Use in-memory DuckDB, or persistent file
        self.loaded_file: Optional[str] = None
        self.parquet_path: Optional[str] = None
        self.meta_path: Optional[str] = None
        self.global_path: Optional[str] = None
        self.cycle_tables: Dict[str, str] = {}
        self.signal_cache: Dict[str, int] = {}
        self.signal_cache_name: Dict[str, str] = {}
        self.rust_parser = wavetoearth_core.WaveParser()

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        for table in self.cycle_tables.values():
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.cycle_tables = {}
        self.signal_cache = {}
        use_shards = os.environ.get("WAVETOEART_USE_SHARDS", "1") != "0"
        shards_env = os.environ.get("WAVETOEART_SHARDS", "auto").strip()
        cache_tag = os.environ.get("WAVETOEART_CACHE_TAG", "v2").strip()
        max_shards_env = os.environ.get("WAVETOEART_MAX_SHARDS", "").strip()
        min_shards_env = os.environ.get("WAVETOEART_MIN_SHARDS", "").strip()

        file_size = os.path.getsize(path)
        cpu_count = os.cpu_count() or 1

        def auto_shards() -> int:
            gb = 1024 * 1024 * 1024
            if file_size < 256 * 1024 * 1024:
                base = min(cpu_count, 8)
            elif file_size < 1 * gb:
                base = min(cpu_count, 16)
            elif file_size < 5 * gb:
                base = min(cpu_count, 32)
            elif file_size < 10 * gb:
                base = min(cpu_count, 48)
            else:
                base = min(cpu_count, 64)
            if path.endswith(".fst"):
                base = max(4, base // 2)
            return max(1, base)

        if shards_env.lower() in ("auto", "0", ""):
            shards = auto_shards()
        else:
            try:
                shards = int(shards_env)
            except ValueError:
                shards = auto_shards()

        try:
            max_shards = int(max_shards_env) if max_shards_env else None
        except ValueError:
            max_shards = None
        try:
            min_shards = int(min_shards_env) if min_shards_env else None
        except ValueError:
            min_shards = None

        if min_shards:
            shards = max(shards, min_shards)
        if max_shards:
            shards = min(shards, max_shards)

        if cache_tag:
            parquet_file = f"{path}.parquet_{cache_tag}"
            parquet_dir = f"{path}.parquet_shards_{cache_tag}"
        else:
            parquet_file = path + ".parquet"
            parquet_dir = path + ".parquet_shards"
        parquet_glob = os.path.join(parquet_dir, "part-*.parquet")

        if use_shards:
            if not os.path.isdir(parquet_dir) or not glob.glob(parquet_glob):
                logger.info(f"Converting {path} to sharded Parquet (shards={shards})...")
                os.makedirs(parquet_dir, exist_ok=True)
                if path.endswith(".fst"):
                    self.rust_parser.convert_fst_to_parquet_sharded(path, parquet_dir, 1_000_000, shards)
                else:
                    self.rust_parser.convert_to_parquet_sharded(path, parquet_dir, 1_000_000, shards)
                logger.info("Conversion complete.")
            else:
                logger.info(f"Using cached Parquet shards: {parquet_dir}")
            self.parquet_path = parquet_glob
            self.meta_path = os.path.join(parquet_dir, "_meta.parquet")
            self.global_path = os.path.join(parquet_dir, "_global.parquet")
        else:
            if not os.path.exists(parquet_file):
                logger.info(f"Converting {path} to Parquet (High-Performance Rust)...")
                if path.endswith(".fst"):
                    self.rust_parser.convert_fst_to_parquet(path, parquet_file, 1_000_000)
                else:
                    self.rust_parser.convert_to_parquet(path, parquet_file, 1_000_000)
                logger.info("Conversion complete.")
            else:
                logger.info(f"Using cached Parquet: {parquet_file}")
            self.parquet_path = parquet_file
            self.meta_path = parquet_file + ".meta.parquet"
            self.global_path = parquet_file + ".global.parquet"

        # Load into DuckDB
        # We replace the view 'wave'
        self.conn.execute("DROP VIEW IF EXISTS wave")
        self.conn.execute(f"CREATE VIEW wave AS SELECT * FROM read_parquet('{self.parquet_path}')")
        self.conn.execute("DROP VIEW IF EXISTS wave_meta")
        self.conn.execute("DROP VIEW IF EXISTS wave_global")
        self.conn.execute("DROP VIEW IF EXISTS signals")
        if self.meta_path and os.path.exists(self.meta_path):
            self.conn.execute(f"CREATE VIEW wave_meta AS SELECT * FROM read_parquet('{self.meta_path}')")
            self.conn.execute("CREATE VIEW signals AS SELECT signal_id, signal_name FROM wave_meta")
        else:
            self.meta_path = None
        if self.global_path and os.path.exists(self.global_path):
            self.conn.execute(f"CREATE VIEW wave_global AS SELECT * FROM read_parquet('{self.global_path}')")
        else:
            self.global_path = None

        self.loaded_file = path
        logger.info(f"Loaded {path} into DuckDB successfully.")

    def get_signal_names(self, pattern: Optional[str] = None) -> List[str]:
        if not self.loaded_file:
            return []
        if not self.meta_path:
            return []

        query = "SELECT DISTINCT signal_name FROM signals"
        if pattern:
            query += f" WHERE regexp_matches(signal_name, '{pattern}')"

        res = self.conn.execute(query).fetchall()
        return [r[0] for r in res]

    def _resolve_signal(self, signal: str) -> Dict[str, Any]:
        cached = self.signal_cache.get(signal)
        if cached is not None:
            return {
                "signal_id": cached,
                "signal_name": self.signal_cache_name.get(signal, signal),
                "match": "cached",
            }
        if not self.meta_path:
            raise HTTPException(status_code=400, detail="Signal metadata not loaded")
        row = self.conn.execute(
            "SELECT signal_id, signal_name FROM signals WHERE signal_name = ? LIMIT 1",
            [signal],
        ).fetchone()
        if row:
            signal_id = int(row[0])
            resolved = row[1]
            self.signal_cache[signal] = signal_id
            self.signal_cache_name[signal] = resolved
            return {"signal_id": signal_id, "signal_name": resolved, "match": "exact"}

        suffix = signal.replace("*", "%").replace("?", "_")
        suffix_rows = self.conn.execute(
            "SELECT signal_id, signal_name FROM signals WHERE signal_name LIKE ? LIMIT 200",
            [f"%{suffix}"],
        ).fetchall()
        if len(suffix_rows) == 1:
            signal_id = int(suffix_rows[0][0])
            resolved = suffix_rows[0][1]
            self.signal_cache[signal] = signal_id
            self.signal_cache_name[signal] = resolved
            return {"signal_id": signal_id, "signal_name": resolved, "match": "suffix"}

        parts = [p for p in signal.replace("*", "%").replace("?", "_").split(".") if p]
        pattern = "%" + "%".join(parts) + "%" if parts else signal.replace("*", "%").replace("?", "_")
        candidates = self.conn.execute(
            "SELECT signal_id, signal_name FROM signals WHERE signal_name LIKE ? LIMIT 200",
            [pattern],
        ).fetchall()
        if not candidates:
            raise HTTPException(status_code=404, detail=f"Signal not found: {signal}")

        def score(name: str) -> int:
            s = 0
            if name == signal:
                s += 1000
            if name.endswith(signal):
                s += 400
            if parts:
                s += 10 * len(parts)
                idx = 0
                for p in parts:
                    pos = name.find(p, idx)
                    if pos < 0:
                        s -= 5
                        break
                    idx = pos + len(p)
                    s += 5
            s -= len(name) // 5
            return s

        candidates_sorted = sorted(candidates, key=lambda c: (-score(c[1]), len(c[1])))
        signal_id = int(candidates_sorted[0][0])
        resolved = candidates_sorted[0][1]
        self.signal_cache[signal] = signal_id
        self.signal_cache_name[signal] = resolved
        return {"signal_id": signal_id, "signal_name": resolved, "match": "fuzzy"}

    def _signal_id(self, signal: str) -> int:
        return self._resolve_signal(signal)["signal_id"]

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
        signal_id = self._signal_id(signal)
        initial_res = self.conn.execute(
            "SELECT value_raw FROM wave WHERE signal_id = ? AND timestamp < ? ORDER BY timestamp DESC LIMIT 1",
            [signal_id, start]
        ).fetchone()

        initial_value = None
        if initial_res:
             initial_value = initial_res[0]

        # 2. Get changes in range
        rows = self.conn.execute(
            "SELECT timestamp, value_raw FROM wave WHERE signal_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
             [signal_id, start, end]
        ).fetchall()

        changes = []
        for r in rows:
            changes.append((r[0], r[1]))

        return {
            "signal": signal,
            "signal_id": signal_id,
            "start": start,
            "end": end,
            "initial_value": initial_value,
            "changes": changes
        }

    def _cycle_range_to_time(self, clock_signal: str, start_cycle: int, end_cycle: int, edge: str) -> Dict[str, int]:
        if start_cycle < 0 or end_cycle < 0:
            raise HTTPException(status_code=400, detail="cycle must be >= 0")
        if end_cycle < start_cycle:
            raise HTTPException(status_code=400, detail="end_cycle must be >= start_cycle")

        table = self._ensure_cycle_table(clock_signal, edge)
        start_row = self.conn.execute(
            f"SELECT timestamp FROM {table} WHERE cycle = ?",
            [start_cycle],
        ).fetchone()
        if not start_row:
            raise HTTPException(status_code=404, detail="start_cycle not found")
        start_ts = start_row[0]

        end_edge_row = self.conn.execute(
            f"SELECT timestamp FROM {table} WHERE cycle = ?",
            [end_cycle + 1],
        ).fetchone()

        if end_edge_row:
            end_ts = max(start_ts, end_edge_row[0] - 1)
        else:
            end_ts = self.conn.execute("SELECT MAX(timestamp) FROM wave").fetchone()[0]

        return {"start_ts": start_ts, "end_ts": end_ts}

    def probe(
        self,
        signals: List[str],
        start: Optional[int],
        end: Optional[int],
        clock: Optional[str],
        start_cycle: Optional[int],
        end_cycle: Optional[int],
        edge: str,
        include_changes: bool,
        max_changes: int,
        max_unique: int,
    ) -> Dict[str, Any]:
        if not self.loaded_file:
            raise HTTPException(status_code=400, detail="No file loaded")
        if not signals:
            raise HTTPException(status_code=400, detail="signals must be non-empty")

        range_info: Dict[str, Any] = {}
        if clock or start_cycle is not None or end_cycle is not None:
            if clock is None or start_cycle is None or end_cycle is None:
                raise HTTPException(status_code=400, detail="clock, start_cycle, end_cycle required for cycle mode")
            edge = edge or "rising"
            time_range = self._cycle_range_to_time(clock, start_cycle, end_cycle, edge)
            start_ts = time_range["start_ts"]
            end_ts = time_range["end_ts"]
            range_info = {
                "mode": "cycle",
                "clock": clock,
                "edge": edge,
                "start_cycle": start_cycle,
                "end_cycle": end_cycle,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
            }
        else:
            if start is None or end is None:
                raise HTTPException(status_code=400, detail="start and end required for time mode")
            start_ts = start
            end_ts = end
            range_info = {
                "mode": "time",
                "start": start_ts,
                "end": end_ts,
            }

        summaries = []
        seen_ids: Dict[int, int] = {}
        for signal in signals:
            resolved = self._resolve_signal(signal)
            signal_id = resolved["signal_id"]
            existing_idx = seen_ids.get(signal_id)
            if existing_idx is not None:
                aliases = summaries[existing_idx].setdefault("aliases", [])
                if signal not in aliases:
                    aliases.append(signal)
                continue

            initial_row = self.conn.execute(
                "SELECT value_raw FROM wave WHERE signal_id = ? AND timestamp < ? ORDER BY timestamp DESC LIMIT 1",
                [signal_id, start_ts],
            ).fetchone()
            final_row = self.conn.execute(
                "SELECT value_raw FROM wave WHERE signal_id = ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
                [signal_id, end_ts],
            ).fetchone()
            count_row = self.conn.execute(
                "SELECT COUNT(*) FROM wave WHERE signal_id = ? AND timestamp >= ? AND timestamp <= ?",
                [signal_id, start_ts, end_ts],
            ).fetchone()
            count = int(count_row[0]) if count_row else 0
            first_row = self.conn.execute(
                "SELECT MIN(timestamp) FROM wave WHERE signal_id = ? AND timestamp >= ? AND timestamp <= ?",
                [signal_id, start_ts, end_ts],
            ).fetchone()
            last_row = self.conn.execute(
                "SELECT MAX(timestamp) FROM wave WHERE signal_id = ? AND timestamp >= ? AND timestamp <= ?",
                [signal_id, start_ts, end_ts],
            ).fetchone()
            unique_rows = self.conn.execute(
                "SELECT DISTINCT value_raw FROM wave WHERE signal_id = ? AND timestamp >= ? AND timestamp <= ? LIMIT ?",
                [signal_id, start_ts, end_ts, max_unique],
            ).fetchall()
            unique_values = [r[0] for r in unique_rows]

            changes = []
            truncated = False
            if include_changes and max_changes > 0:
                changes = self.conn.execute(
                    "SELECT timestamp, value_raw FROM wave WHERE signal_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC LIMIT ?",
                    [signal_id, start_ts, end_ts, max_changes],
                ).fetchall()
                truncated = count > max_changes

            summaries.append(
                {
                    "signal": signal,
                    "resolved_signal": resolved["signal_name"],
                    "match": resolved["match"],
                    "signal_id": signal_id,
                    "aliases": [signal],
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                    "initial_value": initial_row[0] if initial_row else None,
                    "final_value": final_row[0] if final_row else None,
                    "change_count": count,
                    "unique_values": unique_values,
                    "first_change": first_row[0] if first_row else None,
                    "last_change": last_row[0] if last_row else None,
                    "changes": changes if include_changes else None,
                    "truncated": truncated,
                }
            )
            seen_ids[signal_id] = len(summaries) - 1

        return {
            "range": range_info,
            "signals": summaries,
        }

    def inspect(
        self,
        signals: List[str],
        start: Optional[int],
        end: Optional[int],
        clock: Optional[str],
        start_cycle: Optional[int],
        end_cycle: Optional[int],
        edge: str,
        include_changes: bool,
        max_changes: int,
        max_unique: int,
    ) -> Dict[str, Any]:
        base = self.probe(
            signals,
            start,
            end,
            clock,
            start_cycle,
            end_cycle,
            edge,
            include_changes,
            max_changes,
            max_unique,
        )
        range_info = base["range"]
        if range_info.get("mode") == "cycle":
            start_ts = range_info.get("start_timestamp")
            end_ts = range_info.get("end_timestamp")
        else:
            start_ts = range_info.get("start")
            end_ts = range_info.get("end")
        out_signals = []
        for item in base["signals"]:
            changes = item["changes"] if include_changes else None
            truncated = item["truncated"] or not include_changes
            raw = {
                "start_timestamp": item["start_timestamp"],
                "end_timestamp": item["end_timestamp"],
                "initial_value": item["initial_value"],
                "final_value": item["final_value"],
                "change_count": item["change_count"],
                "unique_values": item["unique_values"],
                "first_change": item["first_change"],
                "last_change": item["last_change"],
                "changes": changes,
                "truncated": truncated,
            }
            analysis = SemanticAnalyzer.summarize_signal(
                item["signal"],
                item["start_timestamp"],
                item["end_timestamp"],
                raw["initial_value"],
                raw["final_value"],
                raw["change_count"],
                raw["unique_values"],
                raw["changes"],
                raw["truncated"],
            )
            out_signals.append(
                {
                    "signal": item["signal"],
                    "resolved_signal": item.get("resolved_signal", item["signal"]),
                    "match": item.get("match", "unknown"),
                    "aliases": item.get("aliases", [item["signal"]]),
                    "signal_id": item["signal_id"],
                    "raw": raw,
                    "analysis": analysis,
                }
            )
        relations = []
        if start_ts is not None and end_ts is not None:
            relations = SemanticAnalyzer.analyze_relations(out_signals, start_ts, end_ts)

        return {
            "range": base["range"],
            "signals": out_signals,
            "relations": relations,
        }

    def _ensure_cycle_table(self, clock_signal: str, edge: str) -> str:
        if not self.loaded_file:
            raise HTTPException(status_code=400, detail="No file loaded")

        edge = edge.lower()
        if edge not in ("rising", "falling"):
            raise HTTPException(status_code=400, detail="edge must be 'rising' or 'falling'")

        safe = re.sub(r"[^a-zA-Z0-9_]", "_", clock_signal)
        table = f"cycles_{edge}_{safe}"
        if table in self.cycle_tables:
            return table

        if edge == "rising":
            cond = "value_raw = '1' AND (prev IS NULL OR prev != '1')"
        else:
            cond = "value_raw = '0' AND (prev IS NULL OR prev != '0')"

        signal_id = self._signal_id(clock_signal)

        self.conn.execute(
            f"""
            CREATE TABLE {table} AS
            SELECT
                row_number() OVER (ORDER BY timestamp) - 1 AS cycle,
                timestamp
            FROM (
                SELECT
                    timestamp,
                    value_raw,
                    lag(value_raw) OVER (ORDER BY timestamp) AS prev
                FROM wave
                WHERE signal_id = ?
            ) t
            WHERE {cond}
            """,
            [signal_id],
        )
        self.cycle_tables[table] = table
        return table

    def query_cycles(self, signal: str, clock: str, start_cycle: int, end_cycle: int, edge: str) -> Dict[str, Any]:
        if start_cycle < 0 or end_cycle < 0:
            raise HTTPException(status_code=400, detail="cycle must be >= 0")
        if end_cycle < start_cycle:
            raise HTTPException(status_code=400, detail="end_cycle must be >= start_cycle")

        table = self._ensure_cycle_table(clock, edge)
        start_row = self.conn.execute(
            f"SELECT timestamp FROM {table} WHERE cycle = ?",
            [start_cycle],
        ).fetchone()
        if not start_row:
            raise HTTPException(status_code=404, detail="start_cycle not found")
        start_ts = start_row[0]

        end_edge_row = self.conn.execute(
            f"SELECT timestamp FROM {table} WHERE cycle = ?",
            [end_cycle + 1],
        ).fetchone()

        if end_edge_row:
            end_ts = max(start_ts, end_edge_row[0] - 1)
        else:
            end_ts = self.conn.execute("SELECT MAX(timestamp) FROM wave").fetchone()[0]

        result = self.query(signal, start_ts, end_ts)
        result["clock"] = clock
        result["edge"] = edge
        result["start_cycle"] = start_cycle
        result["end_cycle"] = end_cycle
        result["start_timestamp"] = start_ts
        result["end_timestamp"] = end_ts
        return result

db = WaveformDatabase()

app = FastAPI(title="WaveToEarth Server")

class LoadRequest(BaseModel):
    file_path: str

class QueryRequest(BaseModel):
    signal: str
    start: int
    end: int

class CycleQueryRequest(BaseModel):
    signal: str
    clock: str
    start_cycle: int
    end_cycle: int
    edge: Optional[str] = "rising"

class ProbeRequest(BaseModel):
    signals: List[str]
    start: Optional[int] = None
    end: Optional[int] = None
    clock: Optional[str] = None
    start_cycle: Optional[int] = None
    end_cycle: Optional[int] = None
    edge: Optional[str] = "rising"
    include_changes: bool = False
    max_changes: int = 200
    max_unique: int = 32

class InspectRequest(BaseModel):
    signals: List[str]
    start: Optional[int] = None
    end: Optional[int] = None
    clock: Optional[str] = None
    start_cycle: Optional[int] = None
    end_cycle: Optional[int] = None
    edge: Optional[str] = "rising"
    include_changes: bool = True
    max_changes: int = 200
    max_unique: int = 32

@app.get("/health")
def health():
    return {"status": "ok", "loaded_file": db.loaded_file}

@app.post("/load")
def load_file(req: LoadRequest, background_tasks: BackgroundTasks):
    try:
        db.load(req.file_path)
        # Get signal count
        if db.meta_path:
            count = db.conn.execute("SELECT COUNT(DISTINCT signal_id) FROM signals").fetchone()[0]
        else:
            count = 0
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

@app.post("/query_cycles")
def query_signal_cycles(req: CycleQueryRequest):
    return db.query_cycles(req.signal, req.clock, req.start_cycle, req.end_cycle, req.edge or "rising")

@app.post("/probe")
def probe(req: ProbeRequest):
    return db.probe(
        req.signals,
        req.start,
        req.end,
        req.clock,
        req.start_cycle,
        req.end_cycle,
        req.edge or "rising",
        req.include_changes,
        req.max_changes,
        req.max_unique,
    )

@app.post("/inspect")
def inspect(req: InspectRequest):
    return db.inspect(
        req.signals,
        req.start,
        req.end,
        req.clock,
        req.start_cycle,
        req.end_cycle,
        req.edge or "rising",
        req.include_changes,
        req.max_changes,
        req.max_unique,
    )

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
