#!/usr/bin/env python3
import argparse
from typing import List, Tuple, Optional, Dict, Any

from server import WaveformDatabase


def normalize_bit(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if not isinstance(value, str):
        value = str(value)
    if not value:
        return None
    ch = value[0]
    if ch == "0":
        return 0
    if ch == "1":
        return 1
    return None


def get_changes(db: WaveformDatabase, signal: str) -> List[Tuple[int, str]]:
    resolved = db._resolve_signal(signal)
    signal_id = resolved["signal_id"]
    rows = db.conn.execute(
        "SELECT timestamp, value_raw FROM wave WHERE signal_id = ? ORDER BY timestamp ASC",
        [signal_id],
    ).fetchall()
    return [(int(t), v) for t, v in rows]


def get_state_before(changes: List[Tuple[int, str]], start: int) -> Optional[int]:
    state = None
    for t, v in changes:
        if t >= start:
            break
        state = normalize_bit(v)
    return state


def both_high_intervals(
    a_changes: List[Tuple[int, str]],
    b_changes: List[Tuple[int, str]],
    start: int,
    end: int,
) -> List[Tuple[int, int]]:
    if end < start:
        return []
    a_state = get_state_before(a_changes, start)
    b_state = get_state_before(b_changes, start)
    ai = 0
    bi = 0
    while ai < len(a_changes) and a_changes[ai][0] < start:
        ai += 1
    while bi < len(b_changes) and b_changes[bi][0] < start:
        bi += 1
    intervals: List[Tuple[int, int]] = []
    t = start
    end_excl = end + 1
    while t < end_excl:
        next_t = end_excl
        if ai < len(a_changes):
            next_t = min(next_t, a_changes[ai][0])
        if bi < len(b_changes):
            next_t = min(next_t, b_changes[bi][0])
        if a_state == 1 and b_state == 1 and next_t > t:
            intervals.append((t, next_t))
        if ai < len(a_changes) and a_changes[ai][0] == next_t:
            a_state = normalize_bit(a_changes[ai][1])
            ai += 1
        if bi < len(b_changes) and b_changes[bi][0] == next_t:
            b_state = normalize_bit(b_changes[bi][1])
            bi += 1
        t = next_t
    return intervals


def high_intervals(changes: List[Tuple[int, str]], start: int, end: int) -> List[Tuple[int, int]]:
    if end < start:
        return []
    state = get_state_before(changes, start)
    idx = 0
    while idx < len(changes) and changes[idx][0] < start:
        idx += 1
    t = start
    end_excl = end + 1
    intervals: List[Tuple[int, int]] = []
    while t < end_excl:
        next_t = end_excl
        if idx < len(changes):
            next_t = min(next_t, changes[idx][0])
        if state == 1 and next_t > t:
            intervals.append((t, next_t))
        if idx < len(changes) and changes[idx][0] == next_t:
            state = normalize_bit(changes[idx][1])
            idx += 1
        t = next_t
    return intervals


def summarize_intervals(intervals: List[Tuple[int, int]]) -> Dict[str, Any]:
    if not intervals:
        return {
            "count": 0,
            "total": 0,
            "first": None,
            "last": None,
            "max": 0,
            "avg": 0.0,
        }
    lengths = [end - start for start, end in intervals]
    total = sum(lengths)
    return {
        "count": len(intervals),
        "total": total,
        "first": intervals[0][0],
        "last": intervals[-1][1] - 1,
        "max": max(lengths),
        "avg": total / len(intervals),
    }


def compute_metrics(db: WaveformDatabase, label: str, busy_signal: str) -> Dict[str, Any]:
    min_ts = db.conn.execute("SELECT MIN(timestamp) FROM wave").fetchone()[0]
    max_ts = db.conn.execute("SELECT MAX(timestamp) FROM wave").fetchone()[0]

    busy_changes = get_changes(db, busy_signal)
    busy_intervals = high_intervals(busy_changes, min_ts, max_ts)
    busy_summary = summarize_intervals(busy_intervals)

    def handshake(valid_sig: str, ready_sig: str) -> Dict[str, Any]:
        v = get_changes(db, valid_sig)
        r = get_changes(db, ready_sig)
        intervals_all = both_high_intervals(v, r, min_ts, max_ts)
        summary_all = summarize_intervals(intervals_all)
        if busy_summary["count"] > 0:
            busy_start = busy_summary["first"]
            busy_end = busy_summary["last"]
            intervals_busy = both_high_intervals(v, r, busy_start, busy_end)
            summary_busy = summarize_intervals(intervals_busy)
        else:
            summary_busy = summarize_intervals([])
        return {"all": summary_all, "busy": summary_busy}

    metrics = {
        "label": label,
        "time_range": [min_ts, max_ts],
        "busy_signal": busy_signal,
        "busy": busy_summary,
        "rocc_cmd": handshake("core.io_rocc_cmd_valid", "core.io_rocc_cmd_ready"),
        "gemmini_cmd": handshake("gemmini.io_cmd_valid", "gemmini.io_cmd_ready"),
        "rocc_resp": handshake("core.io_rocc_resp_valid", "core.io_rocc_resp_ready"),
        "gemmini_resp": handshake("gemmini.io_resp_valid", "gemmini.io_resp_ready"),
    }

    cmd_first = metrics["rocc_cmd"]["all"]["first"]
    cmd_last = metrics["rocc_cmd"]["all"]["last"]
    resp_last = metrics["rocc_resp"]["all"]["last"]
    if cmd_first is not None and cmd_last is not None:
        cmd_busy_intervals = high_intervals(busy_changes, cmd_first, cmd_last)
        metrics["busy_in_cmd_window"] = summarize_intervals(cmd_busy_intervals)
        metrics["cmd_window"] = [cmd_first, cmd_last]
    else:
        metrics["busy_in_cmd_window"] = summarize_intervals([])
        metrics["cmd_window"] = None

    if busy_summary["count"] > 0:
        busy_start = busy_summary["first"]
        busy_end = busy_summary["last"]
        metrics["gaps"] = {
            "cmd_to_busy": (busy_start - cmd_first) if cmd_first is not None else None,
            "busy_to_resp": (resp_last - busy_end) if resp_last is not None else None,
            "busy_span": (busy_end - busy_start + 1),
        }
    else:
        metrics["gaps"] = {}

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare gemmini/rocc activity between two waveforms")
    parser.add_argument("--baremetal", required=True, help="Baremetal FST/VCD file")
    parser.add_argument("--iree", required=True, help="IREE FST/VCD file")
    parser.add_argument("--busy-signal", default="gemmini.ex_controller.io_busy")
    args = parser.parse_args()

    db = WaveformDatabase()
    db.load(args.baremetal)
    bare = compute_metrics(db, "baremetal", args.busy_signal)

    db = WaveformDatabase()
    db.load(args.iree)
    iree = compute_metrics(db, "iree", args.busy_signal)

    print("=== Gemmini/RoCC Metrics ===")
    for entry in (bare, iree):
        print(f"\n[{entry['label']}]")
        print("time_range:", entry["time_range"])
        print("busy_signal:", entry["busy_signal"])
        print("busy:", entry["busy"])
        print("gaps:", entry["gaps"])
        print("rocc_cmd(all):", entry["rocc_cmd"]["all"])
        print("rocc_cmd(busy):", entry["rocc_cmd"]["busy"])
        print("gemmini_cmd(all):", entry["gemmini_cmd"]["all"])
        print("gemmini_cmd(busy):", entry["gemmini_cmd"]["busy"])
        print("rocc_resp(all):", entry["rocc_resp"]["all"])
        print("rocc_resp(busy):", entry["rocc_resp"]["busy"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
