import numpy as np
import re
from typing import List, Tuple, Dict, Any, Union, Optional

class SemanticAnalyzer:
    @staticmethod
    def normalize_bit(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value == "0":
            return 0
        if value == "1":
            return 1
        return None

    @staticmethod
    def summarize_signal(
        signal: str,
        start: int,
        end: int,
        initial_value: Optional[str],
        final_value: Optional[str],
        change_count: int,
        unique_values: List[str],
        changes: Optional[List[Tuple[int, str]]],
        truncated: bool,
    ) -> Dict[str, Any]:
        duration = max(0, end - start + 1)
        change_rate = (change_count / duration) if duration > 0 else None

        rising = 0
        falling = 0
        hold_min = None
        hold_max = None
        hold_avg = None

        if changes is None:
            truncated = True

        if not truncated and changes is not None:
            prev = SemanticAnalyzer.normalize_bit(initial_value)
            prev_time = start
            hold_durations: List[int] = []
            for t, v in changes:
                cur = SemanticAnalyzer.normalize_bit(v)
                if prev is not None and cur is not None:
                    if prev == 0 and cur == 1:
                        rising += 1
                    elif prev == 1 and cur == 0:
                        falling += 1
                if t >= start:
                    hold_durations.append(max(0, t - prev_time))
                    prev_time = t
                prev = cur
            hold_durations.append(max(0, end + 1 - prev_time))
            if hold_durations:
                hold_min = min(hold_durations)
                hold_max = max(hold_durations)
                hold_avg = sum(hold_durations) / len(hold_durations)

        if change_count == 0:
            description = f"Signal {signal} remained constant at {initial_value}."
        elif change_count > 50:
            description = f"Signal {signal} toggled heavily ({change_count} changes)."
        else:
            description = f"Signal {signal} changed {change_count} times."

        if truncated:
            description += " Sample truncated."

        return {
            "toggle_count": change_count,
            "is_stable": change_count == 0,
            "initial_value": initial_value,
            "final_value": final_value,
            "unique_values": unique_values,
            "change_rate": change_rate,
            "edges": {"rising": rising, "falling": falling} if not truncated else None,
            "hold": {"min": hold_min, "max": hold_max, "avg": hold_avg} if hold_min is not None else None,
            "analysis_partial": truncated,
            "description": description,
        }

    @staticmethod
    def _handshake_parts(name: str) -> Optional[Tuple[str, str]]:
        lower = name.lower()
        for suffix in ("valid", "ready"):
            if lower.endswith("." + suffix) or lower.endswith("_" + suffix):
                prefix = name[: -(len(suffix) + 1)]
                return prefix, suffix
            if lower.endswith(suffix):
                prefix = name[: -len(suffix)]
                prefix = prefix.rstrip("._")
                if prefix:
                    return prefix, suffix
        return None

    @staticmethod
    def _coactivity_stats(
        start: int,
        end: int,
        a_initial: Optional[str],
        a_changes: List[Tuple[int, str]],
        b_initial: Optional[str],
        b_changes: List[Tuple[int, str]],
    ) -> Dict[str, Any]:
        total = max(0, end - start + 1)
        if total == 0:
            return {
                "total_time": 0,
                "both_high": 0,
                "both_low": 0,
                "a_only": 0,
                "b_only": 0,
                "unknown": 0,
            }

        ia = 0
        ib = 0
        va = SemanticAnalyzer.normalize_bit(a_initial)
        vb = SemanticAnalyzer.normalize_bit(b_initial)
        ta = a_changes[0][0] if ia < len(a_changes) else None
        tb = b_changes[0][0] if ib < len(b_changes) else None
        t = start

        both_high = 0
        both_low = 0
        a_only = 0
        b_only = 0
        unknown = 0

        while t <= end:
            next_t = end + 1
            if ta is not None:
                next_t = min(next_t, ta)
            if tb is not None:
                next_t = min(next_t, tb)
            duration = max(0, next_t - t)
            if va is None or vb is None:
                unknown += duration
            elif va == 1 and vb == 1:
                both_high += duration
            elif va == 0 and vb == 0:
                both_low += duration
            elif va == 1 and vb == 0:
                a_only += duration
            elif va == 0 and vb == 1:
                b_only += duration
            else:
                unknown += duration

            if ta is not None and next_t == ta:
                va = SemanticAnalyzer.normalize_bit(a_changes[ia][1])
                ia += 1
                ta = a_changes[ia][0] if ia < len(a_changes) else None
            if tb is not None and next_t == tb:
                vb = SemanticAnalyzer.normalize_bit(b_changes[ib][1])
                ib += 1
                tb = b_changes[ib][0] if ib < len(b_changes) else None
            t = next_t

        return {
            "total_time": total,
            "both_high": both_high,
            "both_low": both_low,
            "a_only": a_only,
            "b_only": b_only,
            "unknown": unknown,
        }

    @staticmethod
    def analyze_relations(signals: List[Dict[str, Any]], start: int, end: int) -> List[Dict[str, Any]]:
        relations: List[Dict[str, Any]] = []
        if len(signals) < 2:
            return relations

        groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for item in signals:
            parts = SemanticAnalyzer._handshake_parts(item["signal"])
            if not parts:
                continue
            prefix, kind = parts
            groups.setdefault(prefix, {})[kind] = item

        for prefix, pair in groups.items():
            if "valid" not in pair or "ready" not in pair:
                continue
            valid = pair["valid"]
            ready = pair["ready"]
            v_raw = valid["raw"]
            r_raw = ready["raw"]
            if v_raw.get("truncated") or r_raw.get("truncated"):
                relations.append(
                    {
                        "type": "handshake",
                        "valid": valid["signal"],
                        "ready": ready["signal"],
                        "prefix": prefix,
                        "skipped": True,
                        "reason": "truncated_changes",
                    }
                )
                continue
            stats = SemanticAnalyzer._coactivity_stats(
                start,
                end,
                v_raw.get("initial_value"),
                v_raw.get("changes") or [],
                r_raw.get("initial_value"),
                r_raw.get("changes") or [],
            )
            total = stats["total_time"] or 1
            relations.append(
                {
                    "type": "handshake",
                    "valid": valid["signal"],
                    "ready": ready["signal"],
                    "prefix": prefix,
                    "both_high": stats["both_high"],
                    "valid_only": stats["a_only"],
                    "ready_only": stats["b_only"],
                    "both_low": stats["both_low"],
                    "unknown": stats["unknown"],
                    "throughput_ratio": stats["both_high"] / total,
                    "stall_ratio": stats["a_only"] / total,
                    "ready_wait_ratio": stats["b_only"] / total,
                }
            )

        if len(signals) <= 8:
            for i in range(len(signals)):
                for j in range(i + 1, len(signals)):
                    a = signals[i]
                    b = signals[j]
                    a_raw = a["raw"]
                    b_raw = b["raw"]
                    if a_raw.get("truncated") or b_raw.get("truncated"):
                        continue
                    stats = SemanticAnalyzer._coactivity_stats(
                        start,
                        end,
                        a_raw.get("initial_value"),
                        a_raw.get("changes") or [],
                        b_raw.get("initial_value"),
                        b_raw.get("changes") or [],
                    )
                    total = stats["total_time"] or 1
                    relations.append(
                        {
                            "type": "coactivity",
                            "a": a["signal"],
                            "b": b["signal"],
                            "both_high": stats["both_high"],
                            "both_low": stats["both_low"],
                            "a_only": stats["a_only"],
                            "b_only": stats["b_only"],
                            "unknown": stats["unknown"],
                            "same_state_ratio": (stats["both_high"] + stats["both_low"]) / total,
                        }
                    )

        return relations
    @staticmethod
    def classify_signal_type(values: np.ndarray) -> str:
        """
        Guess if signal is 'clock', 'reset', 'control' (1-bit), or 'data' (multi-bit).
        """
        # Heuristic: check bit width or unique values
        if len(values) == 0:
            return "unknown"

        first_val = values[0]
        if isinstance(first_val, str):
            # If string length > 1, likely data bus
            if len(first_val) > 1:
                return "bus"
            # If 'x' or 'z' present, logic
            return "logic"

        # If integer (shouldn't happen with current parser returning strings/objects, but in future)
        return "unknown"

    @staticmethod
    def get_activity_ratio(times: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
        """
        Calculate % of time the signal is 'active' (1).
        Only for 1-bit logic.
        """
        # TODO: Implement accurate time integration
        return 0.0

    @staticmethod
    def summarize_activity(signal_name: str, times: np.ndarray, values: np.ndarray, start: int, end: int) -> Dict[str, Any]:
        """
        Generate a high-level summary of the signal activity in the window.
        """
        # Filter range
        idx_start = np.searchsorted(times, start, side='left')
        idx_end = np.searchsorted(times, end, side='right')

        slice_times = times[idx_start:idx_end]
        slice_values = values[idx_start:idx_end]

        num_toggles = len(slice_values)

        # Initial value: value at idx_start - 1
        initial_val = "unknown"
        if idx_start > 0:
            initial_val = values[idx_start - 1]

        summary = {
            "signal": signal_name,
            "window": (start, end),
            "initial_value": initial_val,
            "toggle_count": num_toggles,
            "is_stable": num_toggles == 0,
        }

        if num_toggles == 0:
            summary["description"] = f"Signal {signal_name} remained constant at {initial_val}."
        elif num_toggles > 50:
            summary["description"] = f"Signal {signal_name} toggled heavily ({num_toggles} times). Likely a clock or noisy data."
        else:
            # List changes succinctly
            changes_desc = []
            for t, v in zip(slice_times, slice_values):
                changes_desc.append(f"{t}: {v}")
            summary["description"] = f"Signal {signal_name} changed {num_toggles} times: " + ", ".join(changes_desc)

        return summary

    @staticmethod
    def detect_protocol_handshake(req_times, req_values, ack_times, ack_values, start, end):
        """
        Advanced: Check valid/ready handshake.
        """
        pass
