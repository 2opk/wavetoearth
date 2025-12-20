import re
import sys
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger("wavetoearth.parser")

class VCDParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.signals = {}  # id -> {name, width, type}
        self.data = defaultdict(list)  # id -> [(time, value), ...]
        self.timescale = "1ns"
        self.end_time = 0
        self.id_to_name = {} # id -> hierarchy.name
        self.name_to_id = {} # hierarchy.name -> id

    def parse(self):
        """
        Parses the VCD file line by line.
        Populates self.signals and self.data.
        Constructs numpy arrays for efficient access.
        """
        logger.info(f"Starting parse of {self.file_path}")

        current_time = 0
        current_scope = []

        # Regex for parsing definitions
        # $var type width identifier reference $end
        var_pattern = re.compile(r"^\$var\s+(\w+)\s+(\d+)\s+(\S+)\s+(.+)\s+\$end$")

        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Timestamp
                if line.startswith('#'):
                    try:
                        current_time = int(line[1:])
                        self.end_time = current_time
                    except ValueError:
                        pass
                    continue

                # Value Changes
                # 1-bit: 0!, 1!, x!, z!
                # Multi-bit: b0010 !, b1100 !
                if line[0] in '01xzXZ':
                    # Single bit change
                    val = line[0]
                    sid = line[1:]
                    # Convert x/z to something processable if needed, or keep as string/int
                    # For now keep as is, optimization later
                    self.data[sid].append((current_time, val))
                elif line.startswith('b') or line.startswith('B'):
                    # Multi bit change
                    parts = line.split()
                    if len(parts) >= 2:
                        val = parts[0][1:] # Remove 'b'
                        sid = parts[1]
                        self.data[sid].append((current_time, val))

                # Header parsing (Scope / Vars)
                elif line.startswith('$scope'):
                    parts = line.split()
                    if len(parts) >= 3:
                        current_scope.append(parts[2])
                elif line.startswith('$upscope'):
                    if current_scope:
                        current_scope.pop()
                elif line.startswith('$var'):
                    match = var_pattern.match(line)
                    if match:
                        _, width, sid, name = match.groups()
                        full_name = ".".join(current_scope + [name])
                        self.signals[sid] = {
                            "name": full_name,
                            "width": int(width),
                            "id": sid
                        }
                        self.id_to_name[sid] = full_name
                        self.name_to_id[full_name] = sid
                elif line.startswith('$timescale'):
                     # Simply grab the next line or part usually
                     pass

        logger.info("Raw parsing finished. Converting to numpy...")
        self._convert_to_numpy()
        logger.info("Conversion finished.")

    def _convert_to_numpy(self):
        """
        Converts list of tuples to numpy structure.
        self.numpy_data[name] = (timestamps, values)
        """
        self.numpy_data = {}
        for sid, changes in self.data.items():
            if not changes:
                continue

            # Unzip
            times, vals = zip(*changes)
            name = self.id_to_name.get(sid)
            if not name:
                continue

            # Convert to numpy
            # Times are always uint64
            t_arr = np.array(times, dtype=np.uint64)

            # Values are tricky.
            # For 1-bit, we can map 0->0, 1->1, x->2, z->3 for efficiency?
            # Or just keep object array for correctness first.
            # Using Object array for safety initially.
            v_arr = np.array(vals, dtype=object)

            self.numpy_data[name] = (t_arr, v_arr)

        # Clear raw data to free memory
        self.data.clear()

    def get_signal(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self.numpy_data.get(name)

    def search_signals(self, pattern: str) -> List[str]:
        import re
        pat = re.compile(pattern)
        return [name for name in self.name_to_id.keys() if pat.search(name)]
