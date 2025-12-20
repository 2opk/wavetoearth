import numpy as np
from typing import List, Tuple, Dict, Any, Union

class SemanticAnalyzer:
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
