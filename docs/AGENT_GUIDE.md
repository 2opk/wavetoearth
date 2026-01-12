# Agent Quick Reference

Fast-reference guide for LLM agents using wavetoearth.

## TL;DR

wavetoearth analyzes hardware simulation recordings (VCD/FST files) to extract signal behavior over time.

**One-liner:**
```bash
wavetoearth file.fst --signals "signal_name" --from START --to END
```

**Python:**
```python
from wavetoearth import Waveform
with Waveform("file.fst") as wf:
    result = wf.inspect(["signal_name"], start=START, end=END)
```

---

## What You Need to Know

### Signals = Named Wires

- Hardware has thousands of named wires (signals)
- Names are hierarchical: `chip.module.submodule.signal_name`
- You usually only need the short name: `signal_name` or `submodule.signal_name`

### Time = Timestamps or Cycles

- **Timestamps**: Absolute time in simulation ticks (e.g., `50000000`)
- **Cycles**: Clock periods (e.g., cycle `100` to `200`)

### Values = Digital States

- `0` = low/false/off
- `1` = high/true/on
- `x` = unknown
- `z` = disconnected

---

## Common Tasks

### Task 1: Check if Something Was Active

```bash
wavetoearth file.fst --signals "busy" --from 0 --to 1000000000
```

Look at:
- `change_count`: How many times it toggled
- `initial_value` / `final_value`: Start and end states

### Task 2: Find All Related Signals

```bash
wavetoearth file.fst --signals "*controller*" --from 0 --to 1000000
```

Wildcards:
- `*busy` = ends with "busy"
- `gemmini.*` = starts with "gemmini."
- `*control*` = contains "control"

### Task 3: Compare Behavior in Two Ranges

```python
result1 = wf.inspect(["io_busy"], start=1000000, end=2000000)
result2 = wf.inspect(["io_busy"], start=5000000, end=6000000)

changes1 = result1["signals"][0]["raw"]["change_count"]
changes2 = result2["signals"][0]["raw"]["change_count"]
```

### Task 4: Find When Something First Happened

```python
result = wf.inspect(["io_busy"], start=0, end=None)  # Full file
first_change = result["signals"][0]["raw"]["first_change"]
print(f"First activity at timestamp {first_change}")
```

---

## Signal Name Matching

wavetoearth finds signals even with partial names:

| You Type | What It Finds |
|----------|---------------|
| `io_busy` | `TOP.chip.gemmini.ex_controller.io_busy` |
| `ex_controller.io_busy` | Same signal (more specific) |
| `*busy` | All signals ending in "busy" |
| `gemmini.*` | All signals under gemmini module |

**If signal not found:** Try wildcards or check available signals:
```bash
wavetoearth signals --pattern "busy"
```

---

## Output Quick Reference

### Key Fields in Response

```json
{
  "signals": [{
    "signal": "what you asked for",
    "resolved_signal": "full.hierarchical.name",
    "raw": {
      "initial_value": "0",       // Value at start of range
      "final_value": "1",         // Value at end of range
      "change_count": 42,         // Number of transitions
      "first_change": 1000500,    // When first changed
      "last_change": 1999000,     // When last changed
      "changes": [[ts, val], ...] // List of changes
    },
    "analysis": {
      "description": "Human readable summary"
    }
  }]
}
```

### Interpreting Results

| Field | What It Tells You |
|-------|-------------------|
| `change_count = 0` | Signal was constant in this range |
| `change_count` high | Signal was very active |
| `initial_value = final_value` | Returned to original state |
| `first_change` near start | Activity began immediately |
| `first_change` far from start | Long idle period before activity |

---

## Common Signal Patterns

### Busy Signals (`*busy`, `*active`)
- `1` = module is working
- `0` = module is idle
- High change_count = efficient utilization
- Long periods of `0` = wasted time

### Valid/Ready Handshakes
- `valid=1`: Sender has data
- `ready=1`: Receiver can accept
- Both `1`: Transfer happening
- `valid=1, ready=0`: Stalled (backpressure)

### Stall Signals (`*stall`)
- `1` = pipeline blocked
- Want this to be `0` most of the time

---

## Choosing Time Ranges

### Full Simulation
```bash
--from 0 --to 999999999999  # Or omit for full range
```

### Finding the Interesting Part

1. First, get overview:
```python
result = wf.inspect(["*busy"], start=0, end=None)
```

2. Find first activity:
```python
first_activity = result["signals"][0]["raw"]["first_change"]
```

3. Focus on that region:
```python
result = wf.inspect(["*"], start=first_activity, end=first_activity + 10000000)
```

### Using Cycles Instead of Timestamps

```bash
wavetoearth file.fst --signals "io_busy" \
  --clock "clock" --start-cycle 1000 --end-cycle 2000
```

---

## Performance Tips

| Situation | Recommendation |
|-----------|----------------|
| Large file, quick check | Use `--no-include-changes` |
| Many signals | Limit with `--max-expand 50` |
| Detailed analysis | Focus on small time range |
| Multiple queries | Reuse `Waveform` object |

```python
# Good: Reuse connection
with Waveform("file.fst") as wf:
    r1 = wf.inspect(["sig1"], start=0, end=1000)
    r2 = wf.inspect(["sig2"], start=0, end=1000)

# Bad: Reconnects each time
r1 = inspect("file.fst", ["sig1"], start=0, end=1000)
r2 = inspect("file.fst", ["sig2"], start=0, end=1000)
```

---

## Error Handling

| Error | Meaning | Fix |
|-------|---------|-----|
| `No signals matched pattern: X` | Signal name not found | Try wildcards, check spelling |
| `Pattern matched N signals (> max)` | Too many matches | Be more specific or increase `--max-expand` |
| `start must be less than end` | Invalid range | Swap start/end values |

---

## Quick Recipes

### "Is this module doing anything?"
```bash
wavetoearth file.fst --signals "*MODULE*busy" --from 0 --to 1000000000
# Check change_count > 0
```

### "When did the error happen?"
```bash
wavetoearth file.fst --signals "*error*,*fault*,*exception*" --from 0 --to END
# Look at first_change timestamps
```

### "How efficient is this pipeline?"
```python
result = wf.inspect(["pipeline.busy"], start=START, end=END)
# Calculate: time_busy / total_time from changes list
```

### "Are these two signals coordinated?"
```python
result = wf.inspect(["signal_a", "signal_b"], start=START, end=END)
# Compare change patterns, check relations array
```

### "What signals exist in this design?"
```bash
wavetoearth signals --pattern ".*"  # List all
wavetoearth signals --pattern "gemmini"  # Filter by module
```

---

## File Format Notes

| Format | Extension | Notes |
|--------|-----------|-------|
| VCD | `.vcd` | Text format, larger files |
| FST | `.fst` | Binary format, compressed, faster |

Both formats work identically with wavetoearth.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `WAVETOEART_SERVER_URL` | `http://localhost:8000` | Server address |
| `WAVETOEART_USE_SHARDS` | `1` | Enable parallel processing |
| `WAVETOEART_CACHE_TAG` | `v1` | Cache version tag |

Usually no need to change these.
