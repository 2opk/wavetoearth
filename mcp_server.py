"""
MCP Server for WaveToEarth - RTL Waveform Analysis

Provides MCP tools for Claude Code to analyze VCD/FST waveforms efficiently.
Designed to minimize context pollution and time overhead.
"""

import asyncio
import json
import logging
from typing import Any, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

logger = logging.getLogger("wavetoearth.mcp")

# Import the database singleton from server.py
# This allows MCP to share state with FastAPI
from server import db, WaveformDatabase


def build_signal_tree(signals: list[str], max_depth: Optional[int] = None) -> dict:
    """
    Build a hierarchical tree from flat signal names.
    Returns a nested dict where leaves have "_signals" key with signal list.
    """
    tree: dict = {}
    for sig in signals:
        parts = sig.split(".")
        if max_depth is not None and len(parts) > max_depth:
            # Truncate to max_depth, rest goes into a bucket
            parts = parts[:max_depth] + ["..."]

        current = tree
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        # Leaf node
        leaf = parts[-1]
        if leaf not in current:
            current[leaf] = {"_signals": []}
        if "_signals" not in current[leaf]:
            current[leaf]["_signals"] = []
        current[leaf]["_signals"].append(sig)

    return tree


def format_tree_text(tree: dict, prefix: str = "", is_last: bool = True) -> str:
    """Format tree as text for display."""
    lines = []
    items = [(k, v) for k, v in tree.items() if k != "_signals"]
    signal_count = len(tree.get("_signals", []))

    for i, (key, subtree) in enumerate(items):
        is_last_item = (i == len(items) - 1)
        connector = "`-- " if is_last_item else "|-- "

        # Count signals in subtree
        def count_signals(t):
            c = len(t.get("_signals", []))
            for k, v in t.items():
                if k != "_signals" and isinstance(v, dict):
                    c += count_signals(v)
            return c

        subtree_count = count_signals(subtree)
        lines.append(f"{prefix}{connector}{key} ({subtree_count} signals)")

        extension = "    " if is_last_item else "|   "
        subtree_text = format_tree_text(subtree, prefix + extension, is_last_item)
        if subtree_text:
            lines.append(subtree_text)

    return "\n".join(lines)


def create_mcp_server() -> Server:
    """Create and configure the MCP server with all tools."""
    server = Server("wavetoearth")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="wave_load",
                description="Load a VCD/FST waveform file into memory for analysis. Must be called before other wave_* tools.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to VCD or FST file"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="wave_unload",
                description="Unload the current waveform file and free memory. Use when done analyzing.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="wave_status",
                description="Get current server status: loaded file, signal count, time range.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="wave_signals",
                description="List available signals. Use pattern for filtering. Use tree=true for hierarchical view.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to filter signals (e.g., '.*valid.*', '.*gemmini.*')"
                        },
                        "tree": {
                            "type": "boolean",
                            "description": "Return as hierarchical tree instead of flat list",
                            "default": False
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Max tree depth (only with tree=true)",
                            "default": 4
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max signals to return (flat mode only)",
                            "default": 100
                        }
                    }
                }
            ),
            Tool(
                name="wave_query",
                description="Query a signal's values in a timestamp range. Returns initial value and all changes.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "signal": {
                            "type": "string",
                            "description": "Signal name (supports fuzzy matching, e.g., 'io_valid' matches 'TOP.system.io_valid')"
                        },
                        "start": {
                            "type": "integer",
                            "description": "Start timestamp (simulation time units)"
                        },
                        "end": {
                            "type": "integer",
                            "description": "End timestamp (simulation time units)"
                        },
                        "max_changes": {
                            "type": "integer",
                            "description": "Max changes to return (prevents context overflow)",
                            "default": 50
                        }
                    },
                    "required": ["signal", "start", "end"]
                }
            ),
            Tool(
                name="wave_query_cycles",
                description="Query a signal by clock cycle range. Automatically converts cycles to timestamps.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "signal": {
                            "type": "string",
                            "description": "Signal name to query"
                        },
                        "clock": {
                            "type": "string",
                            "description": "Clock signal name (e.g., 'clock', 'clk')"
                        },
                        "start_cycle": {
                            "type": "integer",
                            "description": "Start cycle number (0-indexed)"
                        },
                        "end_cycle": {
                            "type": "integer",
                            "description": "End cycle number (inclusive)"
                        },
                        "edge": {
                            "type": "string",
                            "enum": ["rising", "falling"],
                            "description": "Clock edge for cycle counting",
                            "default": "rising"
                        },
                        "max_changes": {
                            "type": "integer",
                            "description": "Max changes to return",
                            "default": 50
                        }
                    },
                    "required": ["signal", "clock", "start_cycle", "end_cycle"]
                }
            ),
            Tool(
                name="wave_probe",
                description="Efficiently inspect multiple signals at once. Returns summary stats without full change lists.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "signals": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of signal names to probe"
                        },
                        "start": {
                            "type": "integer",
                            "description": "Start timestamp (or use clock/start_cycle)"
                        },
                        "end": {
                            "type": "integer",
                            "description": "End timestamp (or use clock/end_cycle)"
                        },
                        "clock": {
                            "type": "string",
                            "description": "Clock signal for cycle-based range"
                        },
                        "start_cycle": {
                            "type": "integer",
                            "description": "Start cycle (requires clock)"
                        },
                        "end_cycle": {
                            "type": "integer",
                            "description": "End cycle (requires clock)"
                        },
                        "include_changes": {
                            "type": "boolean",
                            "description": "Include change list (increases output size)",
                            "default": False
                        },
                        "max_changes": {
                            "type": "integer",
                            "default": 20
                        }
                    },
                    "required": ["signals"]
                }
            ),
            Tool(
                name="wave_inspect",
                description="Deep analysis of signals with semantic interpretation. Detects handshakes, stalls, patterns.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "signals": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Signals to analyze (max 8 for full relation analysis)"
                        },
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "clock": {"type": "string"},
                        "start_cycle": {"type": "integer"},
                        "end_cycle": {"type": "integer"},
                        "include_changes": {
                            "type": "boolean",
                            "default": False
                        },
                        "max_changes": {
                            "type": "integer",
                            "default": 20
                        }
                    },
                    "required": ["signals"]
                }
            ),
            Tool(
                name="wave_find_stall",
                description="Find where a signal stops changing (potential deadlock/stall). Useful for debugging hangs.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "signal": {
                            "type": "string",
                            "description": "Signal to monitor for stalls"
                        },
                        "min_duration": {
                            "type": "integer",
                            "description": "Minimum time units without change to consider a stall",
                            "default": 1000
                        },
                        "start": {
                            "type": "integer",
                            "description": "Start timestamp to search from"
                        },
                        "end": {
                            "type": "integer",
                            "description": "End timestamp to search to"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max stall periods to return",
                            "default": 10
                        }
                    },
                    "required": ["signal"]
                }
            ),
            Tool(
                name="wave_compare_points",
                description="Compare signal values at two different time points. Useful for understanding state changes.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "signals": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Signals to compare"
                        },
                        "time_a": {
                            "type": "integer",
                            "description": "First time point"
                        },
                        "time_b": {
                            "type": "integer",
                            "description": "Second time point"
                        }
                    },
                    "required": ["signals", "time_a", "time_b"]
                }
            ),
            Tool(
                name="wave_find_event",
                description="Find when a signal matches a specific value/condition. Useful for locating specific states.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "signal": {
                            "type": "string",
                            "description": "Signal to search"
                        },
                        "value": {
                            "type": "string",
                            "description": "Value to find (e.g., '1', '0', 'x', or hex like 'deadbeef')"
                        },
                        "start": {
                            "type": "integer",
                            "description": "Start timestamp"
                        },
                        "end": {
                            "type": "integer",
                            "description": "End timestamp"
                        },
                        "edge": {
                            "type": "string",
                            "enum": ["any", "rising", "falling"],
                            "description": "Type of transition to find",
                            "default": "any"
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 20
                        }
                    },
                    "required": ["signal", "value"]
                }
            ),
            Tool(
                name="wave_last_activity",
                description="Find the last change timestamp for signals. Critical for hang debugging - shows what stopped last.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "signals": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Signals to check (or use pattern)"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to match signals"
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["time_asc", "time_desc", "name"],
                            "default": "time_desc"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 20
                        }
                    }
                }
            ),
            Tool(
                name="wave_hierarchy",
                description="Show module hierarchy structure. Quick way to understand design organization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Filter pattern (e.g., 'vector_unit', 'gemmini')"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Max hierarchy depth to show",
                            "default": 3
                        }
                    }
                }
            ),
            Tool(
                name="wave_busy_tree",
                description="Show busy signal tree at a specific time. Traces which sub-modules are causing busy.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "root": {
                            "type": "string",
                            "description": "Root module to start from (e.g., 'vector_unit', 'gemmini')"
                        },
                        "time": {
                            "type": "integer",
                            "description": "Timestamp to check busy signals at"
                        },
                        "busy_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Patterns to match busy signals",
                            "default": ["busy", "stall", "block", "wait", "pending"]
                        }
                    },
                    "required": ["root", "time"]
                }
            ),
            Tool(
                name="wave_find_deadlock",
                description="Detect circular waits in ready/valid/busy signals. Finds potential deadlocks.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "signals": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Signal patterns to check (e.g., ['*.valid', '*.ready', '*.busy'])"
                        },
                        "time": {
                            "type": "integer",
                            "description": "Timestamp to analyze"
                        },
                        "module": {
                            "type": "string",
                            "description": "Module scope to limit search (e.g., 'gemmini')"
                        }
                    },
                    "required": ["time"]
                }
            ),
            Tool(
                name="wave_trace_dependency",
                description="Trace signal dependencies backward to find root cause of a stall.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "signal": {
                            "type": "string",
                            "description": "Starting signal to trace from"
                        },
                        "time": {
                            "type": "integer",
                            "description": "Timestamp to analyze"
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["backward", "forward"],
                            "description": "Trace direction",
                            "default": "backward"
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Max trace depth",
                            "default": 5
                        }
                    },
                    "required": ["signal", "time"]
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            result = await handle_tool_call(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception as e:
            logger.exception(f"Tool {name} failed")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


async def handle_tool_call(name: str, args: dict) -> dict[str, Any]:
    """Handle individual tool calls."""

    if name == "wave_load":
        file_path = args["file_path"]
        try:
            db.load(file_path)
            # Get stats
            signal_count = 0
            time_range = {"min": None, "max": None}
            if db.meta_path:
                signal_count = db.conn.execute("SELECT COUNT(DISTINCT signal_id) FROM signals").fetchone()[0]
            time_row = db.conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM wave").fetchone()
            if time_row:
                time_range = {"min": time_row[0], "max": time_row[1]}
            return {
                "status": "loaded",
                "file": file_path,
                "signal_count": signal_count,
                "time_range": time_range
            }
        except Exception as e:
            return {"error": str(e)}

    elif name == "wave_unload":
        if db.loaded_file:
            old_file = db.loaded_file
            # Clear all state
            db.conn.execute("DROP VIEW IF EXISTS wave")
            db.conn.execute("DROP VIEW IF EXISTS wave_meta")
            db.conn.execute("DROP VIEW IF EXISTS wave_global")
            db.conn.execute("DROP VIEW IF EXISTS signals")
            for table in list(db.cycle_tables.values()):
                db.conn.execute(f"DROP TABLE IF EXISTS {table}")
            db.cycle_tables.clear()
            db.signal_cache.clear()
            db.signal_cache_name.clear()
            db.loaded_file = None
            db.parquet_path = None
            db.meta_path = None
            db.global_path = None
            return {"status": "unloaded", "previous_file": old_file}
        return {"status": "no_file_loaded"}

    elif name == "wave_status":
        if not db.loaded_file:
            return {"status": "no_file_loaded"}

        signal_count = 0
        time_range = {"min": None, "max": None}
        if db.meta_path:
            signal_count = db.conn.execute("SELECT COUNT(DISTINCT signal_id) FROM signals").fetchone()[0]
        time_row = db.conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM wave").fetchone()
        if time_row:
            time_range = {"min": time_row[0], "max": time_row[1]}

        return {
            "status": "loaded",
            "file": db.loaded_file,
            "signal_count": signal_count,
            "time_range": time_range,
            "cached_cycle_tables": list(db.cycle_tables.keys())
        }

    elif name == "wave_signals":
        if not db.loaded_file:
            return {"error": "No file loaded. Use wave_load first."}

        pattern = args.get("pattern")
        as_tree = args.get("tree", False)
        max_depth = args.get("max_depth", 4)
        limit = args.get("limit", 100)

        signals = db.get_signal_names(pattern)
        total = len(signals)

        if as_tree:
            tree = build_signal_tree(signals, max_depth)
            tree_text = format_tree_text(tree)
            return {
                "total_signals": total,
                "tree": tree_text,
                "note": f"Showing hierarchy up to depth {max_depth}"
            }
        else:
            truncated = len(signals) > limit
            return {
                "total_signals": total,
                "signals": signals[:limit],
                "truncated": truncated,
                "note": f"Showing {min(limit, total)} of {total} signals" if truncated else None
            }

    elif name == "wave_query":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        signal = args["signal"]
        start = args["start"]
        end = args["end"]
        max_changes = args.get("max_changes", 50)

        result = db.query(signal, start, end)

        # Truncate changes to limit context size
        changes = result.get("changes", [])
        truncated = len(changes) > max_changes
        if truncated:
            result["changes"] = changes[:max_changes]
            result["truncated"] = True
            result["total_changes"] = len(changes)
            result["note"] = f"Showing {max_changes} of {len(changes)} changes"

        return result

    elif name == "wave_query_cycles":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        signal = args["signal"]
        clock = args["clock"]
        start_cycle = args["start_cycle"]
        end_cycle = args["end_cycle"]
        edge = args.get("edge", "rising")
        max_changes = args.get("max_changes", 50)

        result = db.query_cycles(signal, clock, start_cycle, end_cycle, edge)

        changes = result.get("changes", [])
        truncated = len(changes) > max_changes
        if truncated:
            result["changes"] = changes[:max_changes]
            result["truncated"] = True
            result["total_changes"] = len(changes)

        return result

    elif name == "wave_probe":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        signals = args["signals"]
        start = args.get("start")
        end = args.get("end")
        clock = args.get("clock")
        start_cycle = args.get("start_cycle")
        end_cycle = args.get("end_cycle")
        include_changes = args.get("include_changes", False)
        max_changes = args.get("max_changes", 20)

        result = db.probe(
            signals, start, end, clock, start_cycle, end_cycle,
            "rising", include_changes, max_changes, 16
        )
        return result

    elif name == "wave_inspect":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        signals = args["signals"]
        start = args.get("start")
        end = args.get("end")
        clock = args.get("clock")
        start_cycle = args.get("start_cycle")
        end_cycle = args.get("end_cycle")
        include_changes = args.get("include_changes", False)
        max_changes = args.get("max_changes", 20)

        result = db.inspect(
            signals, start, end, clock, start_cycle, end_cycle,
            "rising", include_changes, max_changes, 16
        )
        return result

    elif name == "wave_find_stall":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        signal = args["signal"]
        min_duration = args.get("min_duration", 1000)
        start = args.get("start")
        end = args.get("end")
        max_results = args.get("max_results", 10)

        signal_id = db._signal_id(signal)
        resolved = db._resolve_signal(signal)

        # Get time bounds
        if start is None or end is None:
            bounds = db.conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM wave").fetchone()
            if start is None:
                start = bounds[0]
            if end is None:
                end = bounds[1]

        # Get all changes for this signal
        rows = db.conn.execute(
            """SELECT timestamp, value_raw FROM wave
               WHERE signal_id = ? AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp ASC""",
            [signal_id, start, end]
        ).fetchall()

        stalls = []
        for i in range(len(rows) - 1):
            t1, v1 = rows[i]
            t2, v2 = rows[i + 1]
            duration = t2 - t1
            if duration >= min_duration:
                stalls.append({
                    "start_time": t1,
                    "end_time": t2,
                    "duration": duration,
                    "value_held": v1
                })

        # Check final value to end of trace
        if rows:
            last_t, last_v = rows[-1]
            final_duration = end - last_t
            if final_duration >= min_duration:
                stalls.append({
                    "start_time": last_t,
                    "end_time": end,
                    "duration": final_duration,
                    "value_held": last_v,
                    "note": "Extends to end of trace - POTENTIAL HANG"
                })

        # Sort by duration descending
        stalls.sort(key=lambda x: -x["duration"])

        return {
            "signal": resolved["signal_name"],
            "search_range": {"start": start, "end": end},
            "min_duration_threshold": min_duration,
            "stall_periods": stalls[:max_results],
            "total_found": len(stalls)
        }

    elif name == "wave_compare_points":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        signals = args["signals"]
        time_a = args["time_a"]
        time_b = args["time_b"]

        comparisons = []
        for signal in signals:
            try:
                resolved = db._resolve_signal(signal)
                signal_id = resolved["signal_id"]

                # Get value at time_a
                val_a_row = db.conn.execute(
                    """SELECT value_raw FROM wave
                       WHERE signal_id = ? AND timestamp <= ?
                       ORDER BY timestamp DESC LIMIT 1""",
                    [signal_id, time_a]
                ).fetchone()

                # Get value at time_b
                val_b_row = db.conn.execute(
                    """SELECT value_raw FROM wave
                       WHERE signal_id = ? AND timestamp <= ?
                       ORDER BY timestamp DESC LIMIT 1""",
                    [signal_id, time_b]
                ).fetchone()

                val_a = val_a_row[0] if val_a_row else None
                val_b = val_b_row[0] if val_b_row else None

                comparisons.append({
                    "signal": signal,
                    "resolved": resolved["signal_name"],
                    f"value_at_{time_a}": val_a,
                    f"value_at_{time_b}": val_b,
                    "changed": val_a != val_b
                })
            except Exception as e:
                comparisons.append({
                    "signal": signal,
                    "error": str(e)
                })

        return {
            "time_a": time_a,
            "time_b": time_b,
            "comparisons": comparisons,
            "changed_count": sum(1 for c in comparisons if c.get("changed", False))
        }

    elif name == "wave_find_event":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        signal = args["signal"]
        value = args["value"]
        start = args.get("start")
        end = args.get("end")
        edge = args.get("edge", "any")
        max_results = args.get("max_results", 20)

        resolved = db._resolve_signal(signal)
        signal_id = resolved["signal_id"]

        # Get time bounds
        if start is None or end is None:
            bounds = db.conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM wave").fetchone()
            if start is None:
                start = bounds[0]
            if end is None:
                end = bounds[1]

        if edge == "any":
            rows = db.conn.execute(
                """SELECT timestamp, value_raw FROM wave
                   WHERE signal_id = ? AND timestamp >= ? AND timestamp <= ? AND value_raw = ?
                   ORDER BY timestamp ASC LIMIT ?""",
                [signal_id, start, end, value, max_results]
            ).fetchall()
            events = [{"time": r[0], "value": r[1]} for r in rows]
        else:
            # Need to find transitions
            rows = db.conn.execute(
                """SELECT timestamp, value_raw, lag(value_raw) OVER (ORDER BY timestamp) as prev
                   FROM wave WHERE signal_id = ? AND timestamp >= ? AND timestamp <= ?
                   ORDER BY timestamp ASC""",
                [signal_id, start, end]
            ).fetchall()

            events = []
            for r in rows:
                ts, val, prev = r
                if val == value:
                    if edge == "rising" and (prev is None or prev != value):
                        events.append({"time": ts, "value": val, "prev": prev})
                    elif edge == "falling" and prev == value:
                        continue  # falling means leaving the value
                if edge == "falling" and prev == value and val != value:
                    events.append({"time": ts, "from_value": prev, "to_value": val})
                if len(events) >= max_results:
                    break

        return {
            "signal": resolved["signal_name"],
            "search_value": value,
            "edge": edge,
            "events": events,
            "total_found": len(events)
        }

    elif name == "wave_last_activity":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        signals = args.get("signals", [])
        pattern = args.get("pattern")
        sort_by = args.get("sort_by", "time_desc")
        limit = args.get("limit", 20)

        if pattern:
            signals = db.get_signal_names(pattern)

        if not signals:
            return {"error": "No signals specified or matched"}

        activities = []
        for signal in signals[:100]:  # Cap at 100 to prevent huge queries
            try:
                resolved = db._resolve_signal(signal)
                signal_id = resolved["signal_id"]

                last_row = db.conn.execute(
                    """SELECT timestamp, value_raw FROM wave
                       WHERE signal_id = ? ORDER BY timestamp DESC LIMIT 1""",
                    [signal_id]
                ).fetchone()

                if last_row:
                    activities.append({
                        "signal": resolved["signal_name"],
                        "last_change_time": last_row[0],
                        "final_value": last_row[1]
                    })
            except Exception:
                pass

        # Sort
        if sort_by == "time_desc":
            activities.sort(key=lambda x: -x["last_change_time"])
        elif sort_by == "time_asc":
            activities.sort(key=lambda x: x["last_change_time"])
        else:
            activities.sort(key=lambda x: x["signal"])

        return {
            "activities": activities[:limit],
            "total_checked": len(activities),
            "note": "Signals sorted by last activity time (most recent first)" if sort_by == "time_desc" else None
        }

    elif name == "wave_hierarchy":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        pattern = args.get("pattern")
        depth = args.get("depth", 3)

        # Get all signal names
        signals = db.get_signal_names(f".*{pattern}.*" if pattern else None)

        if not signals:
            return {"error": f"No signals match pattern: {pattern}"}

        # Find common prefix (the matched module path)
        if pattern:
            # Find where the pattern matches in the first signal
            first_sig = signals[0]
            parts = first_sig.split(".")
            common_prefix_parts = []
            for i, part in enumerate(parts):
                if pattern.lower() in part.lower():
                    common_prefix_parts = parts[:i+1]
                    break
            common_prefix = ".".join(common_prefix_parts) if common_prefix_parts else ""
            prefix_depth = len(common_prefix_parts)
        else:
            common_prefix = ""
            prefix_depth = 0

        # Build module tree from the matched point
        modules: dict = {}
        for sig in signals:
            parts = sig.split(".")
            # Start from after the common prefix
            for i in range(prefix_depth, min(len(parts) - 1, prefix_depth + depth)):
                path = ".".join(parts[:i+1])
                if path not in modules:
                    modules[path] = {"count": 0, "children": set()}
                modules[path]["count"] += 1
                if i > prefix_depth:
                    parent = ".".join(parts[:i])
                    if parent in modules:
                        modules[parent]["children"].add(parts[i])
                elif i == prefix_depth and common_prefix:
                    # Add as child of common_prefix
                    if common_prefix not in modules:
                        modules[common_prefix] = {"count": 0, "children": set()}
                    modules[common_prefix]["count"] += 1
                    if i < len(parts) - 1:
                        modules[common_prefix]["children"].add(parts[i])

        # Build tree text
        def build_tree(prefix: str, current_depth: int) -> list:
            lines = []
            children = sorted(modules.get(prefix, {}).get("children", set()))
            for i, child in enumerate(children):
                full_path = f"{prefix}.{child}" if prefix else child
                count = modules.get(full_path, {}).get("count", 0)
                is_last = i == len(children) - 1
                connector = "`-- " if is_last else "|-- "
                lines.append(f"{'    ' * current_depth}{connector}{child} ({count} signals)")
                if current_depth < depth - 1:
                    lines.extend(build_tree(full_path, current_depth + 1))
            return lines

        tree_lines = []
        if common_prefix:
            count = modules.get(common_prefix, {}).get("count", len(signals))
            tree_lines.append(f"{common_prefix} ({count} signals)")
            tree_lines.extend(build_tree(common_prefix, 0))
        else:
            # No pattern - show from root
            roots = set()
            for sig in signals[:1000]:
                parts = sig.split(".")
                if parts:
                    roots.add(parts[0])
            for root in sorted(roots):
                count = modules.get(root, {}).get("count", 0)
                tree_lines.append(f"{root} ({count} signals)")
                tree_lines.extend(build_tree(root, 0))

        return {
            "pattern": pattern,
            "depth": depth,
            "total_signals": len(signals),
            "hierarchy": "\n".join(tree_lines[:100]),
            "note": f"Showing top {min(100, len(tree_lines))} lines"
        }

    elif name == "wave_busy_tree":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        root = args["root"]
        time = args["time"]
        busy_patterns = args.get("busy_patterns", ["io_busy", "busy"])

        # Find all busy-like signals under root (prioritize io_busy pattern)
        busy_signals = []
        for pattern in busy_patterns:
            # Match signals ending with pattern (e.g., io_busy, not io_busy_counter)
            matches = db.get_signal_names(f".*{root}.*\\.{pattern}$")
            busy_signals.extend(matches)
            # Also match io_busy variants
            matches = db.get_signal_names(f".*{root}.*{pattern}[^a-zA-Z_].*")
            busy_signals.extend(matches)
        busy_signals = list(set(busy_signals))

        # Get values at specified time
        busy_tree = []
        for signal in busy_signals[:300]:  # Limit
            try:
                resolved = db._resolve_signal(signal)
                signal_id = resolved["signal_id"]

                # Get value at time
                row = db.conn.execute(
                    """SELECT value_raw FROM wave
                       WHERE signal_id = ? AND timestamp <= ?
                       ORDER BY timestamp DESC LIMIT 1""",
                    [signal_id, time]
                ).fetchone()

                if row:
                    value = row[0]
                    # Check if "busy" (value is '1' for single-bit signals)
                    is_busy = value == '1'
                    busy_tree.append({
                        "signal": resolved["signal_name"],
                        "value": value,
                        "is_busy": is_busy
                    })
            except Exception:
                pass

        # Sort by hierarchy depth and name
        busy_tree.sort(key=lambda x: (x["signal"].count("."), x["signal"]))

        # Build hierarchical tree for busy signals
        def get_module_path(sig: str) -> str:
            parts = sig.rsplit(".", 1)
            return parts[0] if len(parts) == 2 else ""

        def get_relative_path(sig: str, root_module: str) -> str:
            if root_module in sig:
                idx = sig.find(root_module)
                return sig[idx:]
            return sig

        # Build tree text showing hierarchy
        tree_lines = []
        seen_modules = set()

        for item in busy_tree:
            if item["is_busy"]:
                rel_path = get_relative_path(item["signal"], root)
                parts = rel_path.split(".")
                # Show with indentation based on depth
                indent = len(parts) - 2  # -2 because first part is root, last is signal
                indent = max(0, indent)
                sig_name = parts[-1]
                module = ".".join(parts[:-1])

                # Add module header if first time seeing this module
                if module not in seen_modules and indent > 0:
                    seen_modules.add(module)
                    module_name = parts[-2] if len(parts) >= 2 else module
                    tree_lines.append(f"{'  ' * (indent-1)}{module_name}/")

                tree_lines.append(f"{'  ' * indent}{sig_name}=1")

        return {
            "root": root,
            "time": time,
            "busy_count": len([x for x in busy_tree if x["is_busy"]]),
            "total_checked": len(busy_tree),
            "tree": "\n".join(tree_lines[:80]),
            "details": [x for x in busy_tree if x["is_busy"]][:30]
        }

    elif name == "wave_find_deadlock":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        time = args["time"]
        signal_patterns = args.get("signals", [".*valid.*", ".*ready.*", ".*busy.*"])
        module = args.get("module")

        # Find matching signals
        all_signals = []
        for pattern in signal_patterns:
            if module:
                pattern = f".*{module}.*{pattern}"
            matches = db.get_signal_names(pattern)
            all_signals.extend(matches)
        all_signals = list(set(all_signals))[:500]  # Limit

        # Get values at specified time
        signal_states = {}
        for signal in all_signals:
            try:
                resolved = db._resolve_signal(signal)
                signal_id = resolved["signal_id"]

                row = db.conn.execute(
                    """SELECT value_raw FROM wave
                       WHERE signal_id = ? AND timestamp <= ?
                       ORDER BY timestamp DESC LIMIT 1""",
                    [signal_id, time]
                ).fetchone()

                if row:
                    signal_states[resolved["signal_name"]] = row[0]
            except Exception:
                pass

        # Analyze ready/valid pairs for deadlock
        deadlock_candidates = []

        # Group by module
        modules_signals: dict = {}
        for sig, val in signal_states.items():
            parts = sig.rsplit(".", 1)
            if len(parts) == 2:
                mod, name = parts
            else:
                mod, name = "", parts[0]
            if mod not in modules_signals:
                modules_signals[mod] = {}
            modules_signals[mod][name] = val

        # Find potential deadlocks: valid=1 but ready=0
        for mod, sigs in modules_signals.items():
            issues = []
            for name, val in sigs.items():
                if "valid" in name.lower() and val == "1":
                    # Check if corresponding ready is 0
                    ready_name = name.replace("valid", "ready").replace("Valid", "Ready")
                    if ready_name in sigs and sigs[ready_name] == "0":
                        issues.append(f"{name}=1 but {ready_name}=0")
                if "busy" in name.lower() and val == "1":
                    issues.append(f"{name}=1 (busy)")

            if issues:
                deadlock_candidates.append({
                    "module": mod,
                    "issues": issues,
                    "signals": {k: v for k, v in sigs.items() if any(
                        x in k.lower() for x in ["valid", "ready", "busy", "stall"]
                    )}
                })

        # Sort by number of issues
        deadlock_candidates.sort(key=lambda x: -len(x["issues"]))

        return {
            "time": time,
            "total_signals_checked": len(signal_states),
            "potential_deadlocks": len(deadlock_candidates),
            "candidates": deadlock_candidates[:20],
            "note": "Modules with valid=1/ready=0 or busy=1 patterns"
        }

    elif name == "wave_trace_dependency":
        if not db.loaded_file:
            return {"error": "No file loaded"}

        signal = args["signal"]
        time = args["time"]
        direction = args.get("direction", "backward")
        max_depth = args.get("max_depth", 5)

        resolved = db._resolve_signal(signal)
        start_signal = resolved["signal_name"]

        # Get value at time
        signal_id = resolved["signal_id"]
        row = db.conn.execute(
            """SELECT value_raw FROM wave
               WHERE signal_id = ? AND timestamp <= ?
               ORDER BY timestamp DESC LIMIT 1""",
            [signal_id, time]
        ).fetchone()
        start_value = row[0] if row else "?"

        # Extract module path
        parts = start_signal.rsplit(".", 1)
        if len(parts) == 2:
            module_path, sig_name = parts
        else:
            module_path, sig_name = "", parts[0]

        # Find related signals in same/parent module
        trace = [{
            "depth": 0,
            "signal": start_signal,
            "value": start_value,
            "note": "Starting point"
        }]

        visited = {start_signal}

        def find_related(current_module: str, current_depth: int):
            if current_depth >= max_depth:
                return

            # Look for signals that might be dependencies
            dependency_patterns = [
                "ready", "valid", "busy", "stall", "wait",
                "pending", "full", "empty", "grant", "req"
            ]

            related = []
            for pattern in dependency_patterns:
                matches = db.get_signal_names(f".*{current_module}.*{pattern}.*")
                for m in matches[:20]:
                    if m not in visited:
                        related.append(m)
                        visited.add(m)

            # Get values for related signals
            for rel_signal in related[:10]:
                try:
                    rel_resolved = db._resolve_signal(rel_signal)
                    rel_id = rel_resolved["signal_id"]

                    rel_row = db.conn.execute(
                        """SELECT value_raw FROM wave
                           WHERE signal_id = ? AND timestamp <= ?
                           ORDER BY timestamp DESC LIMIT 1""",
                        [rel_id, time]
                    ).fetchone()

                    if rel_row:
                        val = rel_row[0]
                        # Check if this might be blocking
                        is_blocking = val in ("1", "0") and any(
                            x in rel_signal.lower() for x in ["busy", "stall", "wait", "pending", "full"]
                        ) and val == "1"
                        is_blocking = is_blocking or (
                            "ready" in rel_signal.lower() and val == "0"
                        )

                        if is_blocking or current_depth < 2:
                            trace.append({
                                "depth": current_depth + 1,
                                "signal": rel_resolved["signal_name"],
                                "value": val,
                                "note": "BLOCKING" if is_blocking else ""
                            })
                except Exception:
                    pass

            # Go up to parent module if backward tracing
            if direction == "backward" and "." in current_module:
                parent = current_module.rsplit(".", 1)[0]
                find_related(parent, current_depth + 1)

        find_related(module_path, 0)

        # Format trace
        trace_text = []
        for item in trace:
            indent = "  " * item["depth"]
            note = f" <- {item['note']}" if item["note"] else ""
            trace_text.append(f"{indent}{item['signal']}={item['value']}{note}")

        return {
            "start_signal": start_signal,
            "start_value": start_value,
            "time": time,
            "direction": direction,
            "trace": trace_text[:30],
            "details": trace[:30],
            "note": f"Traced {len(trace)} signals, max_depth={max_depth}"
        }

    else:
        return {"error": f"Unknown tool: {name}"}


async def run_mcp_server():
    """Run the MCP server on stdio."""
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point for MCP server."""
    asyncio.run(run_mcp_server())


if __name__ == "__main__":
    main()
