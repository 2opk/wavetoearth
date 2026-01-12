# Usage Examples

Real-world analysis scenarios with wavetoearth.

## Example 1: Basic Signal Inspection

**Goal:** Check if an accelerator was busy during a time range.

### CLI

```bash
wavetoearth /path/to/simulation.fst \
  --signals "gemmini.io_busy" \
  --from 50000000 --to 100000000
```

### Python

```python
from wavetoearth import Waveform

with Waveform("/path/to/simulation.fst") as wf:
    result = wf.inspect(
        signals=["gemmini.io_busy"],
        start=50000000,
        end=100000000
    )

    sig = result["signals"][0]
    print(f"Busy signal changed {sig['raw']['change_count']} times")
    print(f"Initial: {sig['raw']['initial_value']}, Final: {sig['raw']['final_value']}")
```

### Output

```json
{
  "range": {"mode": "time", "start": 50000000, "end": 100000000},
  "signals": [{
    "signal": "gemmini.io_busy",
    "resolved_signal": "TOP.chip.gemmini.io_busy",
    "raw": {
      "initial_value": "1",
      "final_value": "0",
      "change_count": 4,
      "changes": [[63547000, "0"], [65165000, "1"], [75109000, "0"], [82927000, "1"]]
    },
    "analysis": {"description": "Signal toggled 4 times between active and idle."}
  }]
}
```

---

## Example 2: Finding Signals with Wildcards

**Goal:** Find all busy signals in the design.

### CLI

```bash
# List matching signals first
wavetoearth signals --pattern ".*busy"

# Then inspect them
wavetoearth /path/to/simulation.fst \
  --signals "*busy" \
  --from 0 --to 1000000000
```

### Python

```python
from wavetoearth import Waveform

with Waveform("/path/to/simulation.fst") as wf:
    # Find all busy signals
    busy_signals = wf.signals(".*busy")
    print(f"Found {len(busy_signals)} busy signals")

    # Inspect all of them
    result = wf.inspect(
        signals="*busy",  # Wildcard pattern
        start=0,
        end=1000000000
    )

    for sig in result["signals"]:
        name = sig["resolved_signal"].split(".")[-1]
        changes = sig["raw"]["change_count"]
        print(f"{name}: {changes} changes")
```

### Output

```
Found 5 busy signals
io_busy: 81 changes
load_controller.io_busy: 9 changes
ex_controller.io_busy: 81 changes
store_controller.io_busy: 1248 changes
cmd_tracker.io_busy: 156 changes
```

---

## Example 3: Cycle-Based Analysis

**Goal:** Analyze signal behavior for specific clock cycles.

### CLI

```bash
wavetoearth /path/to/simulation.fst \
  --signals "gemmini.io_busy" \
  --clock "clock" \
  --start-cycle 100000 \
  --end-cycle 200000
```

### Python

```python
from wavetoearth import Waveform

with Waveform("/path/to/simulation.fst") as wf:
    result = wf.inspect(
        signals=["gemmini.io_busy"],
        clock="clock",
        start_cycle=100000,
        end_cycle=200000
    )

    rng = result["range"]
    print(f"Cycles {rng['start_cycle']}-{rng['end_cycle']}")
    print(f"Timestamps {rng['start_timestamp']}-{rng['end_timestamp']}")
```

### Output

```json
{
  "range": {
    "mode": "cycle",
    "clock": "TOP.chip.clock",
    "edge": "rising",
    "start_cycle": 100000,
    "end_cycle": 200000,
    "start_timestamp": 100000500,
    "end_timestamp": 200001499
  }
}
```

---

## Example 4: Comparing Two Simulations

**Goal:** Compare accelerator utilization between two implementations.

### Python

```python
from wavetoearth import Waveform

def analyze_utilization(file_path, name):
    with Waveform(file_path) as wf:
        result = wf.inspect(
            signals=["gemmini.ex_controller.io_busy"],
            start=50000000,
            end=500000000
        )

        sig = result["signals"][0]
        changes = sig["raw"]["changes"]

        # Calculate busy time
        busy_time = 0
        total_time = 500000000 - 50000000

        prev_time = 50000000
        prev_val = sig["raw"]["initial_value"]

        for ts, val in changes:
            if prev_val == "1":
                busy_time += ts - prev_time
            prev_time = ts
            prev_val = val

        # Handle final segment
        if prev_val == "1":
            busy_time += 500000000 - prev_time

        utilization = busy_time / total_time
        print(f"{name}: {utilization:.1%} utilization ({sig['raw']['change_count']} state changes)")
        return utilization

# Compare
util_a = analyze_utilization("/path/to/baremetal.fst", "Baremetal")
util_b = analyze_utilization("/path/to/optimized.fst", "Optimized")

print(f"\nImprovement: {util_b/util_a:.2f}x")
```

### Output

```
Baremetal: 62.4% utilization (81 state changes)
Optimized: 85.6% utilization (81 state changes)

Improvement: 1.37x
```

---

## Example 5: Analyzing Handshake Protocol

**Goal:** Check valid/ready handshake efficiency.

### Python

```python
from wavetoearth import Waveform

with Waveform("/path/to/simulation.fst") as wf:
    result = wf.inspect(
        signals=["io_cmd_valid", "io_cmd_ready"],
        start=50000000,
        end=500000000,
        include_changes=True
    )

    valid_sig = result["signals"][0]["raw"]
    ready_sig = result["signals"][1]["raw"]

    print(f"Valid changes: {valid_sig['change_count']}")
    print(f"Ready changes: {ready_sig['change_count']}")

    # Check relations if detected
    for rel in result.get("relations", []):
        if rel["type"] == "handshake":
            print(f"\nHandshake detected:")
            print(f"  Transactions: {rel['transactions']}")
            print(f"  Throughput: {rel['throughput_ratio']:.1%}")
            print(f"  Stall ratio: {rel['stall_ratio']:.1%}")
```

---

## Example 6: Finding Idle Gaps

**Goal:** Find periods where accelerator was idle.

### Python

```python
from wavetoearth import Waveform

with Waveform("/path/to/simulation.fst") as wf:
    result = wf.inspect(
        signals=["gemmini.ex_controller.io_busy"],
        start=50000000,
        end=500000000,
        include_changes=True
    )

    sig = result["signals"][0]["raw"]
    changes = sig["raw"]["changes"]

    # Find idle gaps (transitions to 0)
    idle_gaps = []
    for i, (ts, val) in enumerate(changes):
        if val == "0" and i + 1 < len(changes):
            next_ts, next_val = changes[i + 1]
            if next_val == "1":
                gap_duration = next_ts - ts
                idle_gaps.append((ts, next_ts, gap_duration))

    # Sort by duration
    idle_gaps.sort(key=lambda x: x[2], reverse=True)

    print("Top 5 longest idle gaps:")
    for start, end, duration in idle_gaps[:5]:
        print(f"  {start:,} - {end:,}: {duration:,} ticks ({duration/1000:.1f}us)")
```

### Output

```
Top 5 longest idle gaps:
  225,789,000 - 261,125,000: 35,336,000 ticks (35336.0us)
  181,957,000 - 205,245,000: 23,288,000 ticks (23288.0us)
  140,307,000 - 160,617,000: 20,310,000 ticks (20310.0us)
  100,303,000 - 119,947,000: 19,644,000 ticks (19644.0us)
  63,547,000 - 82,927,000: 19,380,000 ticks (19380.0us)
```

---

## Example 7: Batch Analysis of Multiple Files

**Goal:** Analyze multiple simulation results.

### CLI

```bash
# Using glob pattern
wavetoearth /path/to/results/*.fst \
  --signals "gemmini.io_busy" \
  --from 0 --to 1000000000 \
  --output json > results.json
```

### Python

```python
from wavetoearth import Waveform
from pathlib import Path
import json

results = []
for fst_file in Path("/path/to/results").glob("*.fst"):
    with Waveform(str(fst_file)) as wf:
        result = wf.inspect(
            signals=["gemmini.io_busy"],
            start=0,
            end=1000000000
        )
        results.append({
            "file": fst_file.name,
            "changes": result["signals"][0]["raw"]["change_count"]
        })

# Sort by activity
results.sort(key=lambda x: x["changes"], reverse=True)

print("Files ranked by activity:")
for r in results:
    print(f"  {r['file']}: {r['changes']} changes")
```

---

## Example 8: Quick Health Check

**Goal:** Verify simulation completed successfully.

### Python

```python
from wavetoearth import Waveform

def check_simulation(file_path):
    with Waveform(file_path) as wf:
        # Check reset deasserted
        reset = wf.inspect(["reset"], start=0, end=100000)
        if reset["signals"][0]["raw"]["final_value"] != "0":
            return "FAIL: Reset still asserted"

        # Check clock is toggling
        clock = wf.inspect(["clock"], start=0, end=100000)
        if clock["signals"][0]["raw"]["change_count"] < 100:
            return "FAIL: Clock not toggling"

        # Check for activity
        busy = wf.inspect(["*busy"], start=0, end=1000000000)
        total_changes = sum(s["raw"]["change_count"] for s in busy["signals"])
        if total_changes == 0:
            return "WARN: No activity detected"

        return f"OK: {total_changes} state changes detected"

print(check_simulation("/path/to/simulation.fst"))
```

---

## Tips for Effective Analysis

### 1. Start Broad, Then Narrow

```python
# First: Find interesting time range
result = wf.inspect(["io_busy"], start=0, end=None)  # Full range
first_activity = result["signals"][0]["raw"]["first_change"]

# Then: Focus on that range
result = wf.inspect(["io_busy"], start=first_activity, end=first_activity + 1000000)
```

### 2. Use `--no-include-changes` for Speed

```bash
# Fast summary (no change list)
wavetoearth file.fst --signals "*busy" --from 0 --to 1000000000 --no-include-changes

# Detailed analysis (with changes)
wavetoearth file.fst --signals "ex_controller.io_busy" --from 50000000 --to 60000000
```

### 3. Combine Multiple Signals for Context

```python
# Analyze related signals together
result = wf.inspect([
    "io_cmd_valid",
    "io_cmd_ready",
    "ex_controller.io_busy",
    "load_controller.io_busy"
], start=50000000, end=100000000)
```

### 4. Use Cycle Mode for Clock-Domain Analysis

```python
# When exact cycle boundaries matter
result = wf.inspect(
    ["pipeline_stage1_valid", "pipeline_stage2_valid"],
    clock="core_clock",
    start_cycle=1000,
    end_cycle=1100
)
```
