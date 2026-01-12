# Output Schema Reference

Complete specification of wavetoearth JSON output structures.

## `inspect` Response

The main analysis endpoint returns this structure:

```json
{
  "range": { ... },
  "signals": [ ... ],
  "relations": [ ... ]
}
```

### `range` Object

Describes the queried time/cycle range.

#### Timestamp Mode

```json
{
  "range": {
    "mode": "time",
    "start": 1000000,
    "end": 2000000
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | Always `"time"` for timestamp mode |
| `start` | int | Start timestamp (inclusive) |
| `end` | int | End timestamp (inclusive) |

#### Cycle Mode

```json
{
  "range": {
    "mode": "cycle",
    "clock": "TOP.chip.clock",
    "edge": "rising",
    "start_cycle": 100,
    "end_cycle": 200,
    "start_timestamp": 100500,
    "end_timestamp": 200499
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | Always `"cycle"` for cycle mode |
| `clock` | string | Full clock signal name used |
| `edge` | string | `"rising"` or `"falling"` |
| `start_cycle` | int | Start cycle number |
| `end_cycle` | int | End cycle number |
| `start_timestamp` | int | Computed start timestamp |
| `end_timestamp` | int | Computed end timestamp |

### `signals` Array

Array of signal analysis results.

```json
{
  "signals": [
    {
      "signal": "io_busy",
      "resolved_signal": "TOP.chip.gemmini.ex_controller.io_busy",
      "match": "suffix",
      "aliases": ["gemmini.ex_controller.io_busy", "ex_controller.io_busy"],
      "signal_id": 13275,
      "raw": { ... },
      "analysis": { ... }
    }
  ]
}
```

#### Signal Metadata

| Field | Type | Description |
|-------|------|-------------|
| `signal` | string | Input signal name (what you requested) |
| `resolved_signal` | string | Full hierarchical signal name found |
| `match` | string | How it was matched: `"exact"`, `"suffix"`, `"fuzzy"`, or `"cached"` |
| `aliases` | array | Alternative names that resolve to same signal |
| `signal_id` | int | Internal signal ID in waveform file |

#### `raw` Object

Raw statistical data for the signal.

```json
{
  "raw": {
    "start_timestamp": 1000000,
    "end_timestamp": 2000000,
    "initial_value": "0",
    "final_value": "1",
    "change_count": 42,
    "unique_values": ["0", "1"],
    "first_change": 1000500,
    "last_change": 1999000,
    "changes": [
      [1000500, "1"],
      [1001000, "0"],
      [1001500, "1"]
    ],
    "truncated": false
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `start_timestamp` | int | Actual start of queried range |
| `end_timestamp` | int | Actual end of queried range |
| `initial_value` | string | Signal value at start_timestamp |
| `final_value` | string | Signal value at end_timestamp |
| `change_count` | int | Total number of value changes |
| `unique_values` | array | List of distinct values (up to `max_unique`) |
| `first_change` | int\|null | Timestamp of first change (null if no changes) |
| `last_change` | int\|null | Timestamp of last change (null if no changes) |
| `changes` | array | List of `[timestamp, value]` pairs |
| `truncated` | bool | True if changes list was limited by `max_changes` |

**Note:** `changes` is only included if `include_changes=true` (default).

#### `analysis` Object

Semantic interpretation of the signal behavior.

```json
{
  "analysis": {
    "description": "Signal io_busy toggled 42 times between active (1) and idle (0) states.",
    "pattern": "toggle",
    "duty_cycle": 0.73,
    "frequency": 21000
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `description` | string | Human-readable summary |
| `pattern` | string | Detected pattern (see below) |
| `duty_cycle` | float | Ratio of time signal was high (0.0-1.0) |
| `frequency` | int | Approximate toggle frequency |

**Pattern types:**

| Pattern | Description |
|---------|-------------|
| `"constant"` | Signal never changed |
| `"toggle"` | Signal switched between values |
| `"pulse"` | Brief spikes in value |
| `"ramp"` | Steadily increasing/decreasing (buses) |
| `"unknown"` | No clear pattern detected |

### `relations` Array

Detected relationships between signals.

```json
{
  "relations": [
    {
      "type": "handshake",
      "valid": "io_cmd_valid",
      "ready": "io_cmd_ready",
      "transactions": 125,
      "throughput_ratio": 0.85,
      "stall_ratio": 0.15,
      "avg_latency": 3.2
    },
    {
      "type": "coactivity",
      "a": "ex_controller.io_busy",
      "b": "load_controller.io_busy",
      "same_state_ratio": 0.62,
      "correlation": 0.45
    }
  ]
}
```

#### Handshake Relation

Detected valid/ready protocol pair.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"handshake"` |
| `valid` | string | Valid signal name |
| `ready` | string | Ready signal name |
| `transactions` | int | Number of completed transactions (valid AND ready) |
| `throughput_ratio` | float | Fraction of time with active transaction |
| `stall_ratio` | float | Fraction of time valid=1 but ready=0 |
| `avg_latency` | float | Average cycles from valid to transaction |

#### Coactivity Relation

Correlation between two signals.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"coactivity"` |
| `a` | string | First signal name |
| `b` | string | Second signal name |
| `same_state_ratio` | float | Fraction of time both signals had same value |
| `correlation` | float | Statistical correlation (-1.0 to 1.0) |

---

## `probe` Response

Lightweight version without analysis.

```json
{
  "range": {
    "mode": "time",
    "start": 1000000,
    "end": 2000000
  },
  "signals": [
    {
      "signal": "io_busy",
      "resolved_signal": "TOP.chip.gemmini.ex_controller.io_busy",
      "signal_id": 13275,
      "initial_value": "0",
      "final_value": "1",
      "change_count": 42,
      "unique_values": ["0", "1"],
      "first_change": 1000500,
      "last_change": 1999000,
      "changes": [[1000500, "1"], [1001000, "0"]],
      "truncated": false
    }
  ]
}
```

**Differences from `inspect`:**
- No `analysis` object per signal
- No `relations` array
- Flattened structure (no `raw` wrapper)

---

## `query` Response

Single signal query result.

```json
{
  "signal": "io_busy",
  "resolved_signal": "TOP.chip.gemmini.ex_controller.io_busy",
  "start": 1000000,
  "end": 2000000,
  "changes": [
    [1000500, "1"],
    [1001000, "0"],
    [1001500, "1"]
  ]
}
```

---

## `analyze` Response

Semantic analysis only.

```json
{
  "signal": "io_busy",
  "window": [1000000, 2000000],
  "initial_value": "0",
  "is_stable": false,
  "description": "Signal io_busy toggled 42 times."
}
```

---

## `query_cycles` Response

Cycle-based query result.

```json
{
  "signal": "io_busy",
  "resolved_signal": "TOP.chip.gemmini.ex_controller.io_busy",
  "clock": "TOP.chip.clock",
  "edge": "rising",
  "start_cycle": 100,
  "end_cycle": 200,
  "start_timestamp": 100500,
  "end_timestamp": 200499,
  "changes": [
    [100750, "1"],
    [101250, "0"]
  ]
}
```

---

## `signals` Response

List of available signals.

```json
{
  "signals": [
    "TOP.TestDriver.testHarness.chiptop.clock",
    "TOP.TestDriver.testHarness.chiptop.reset",
    "TOP.TestDriver.testHarness.chiptop.system.tile.gemmini.io_busy"
  ]
}
```

---

## Error Responses

### 400 Bad Request

Invalid parameters.

```json
{
  "detail": "start must be less than end"
}
```

### 404 Not Found

Signal or resource not found.

```json
{
  "detail": "No signals matched pattern: nonexistent_signal"
}
```

### 500 Internal Server Error

Server-side error.

```json
{
  "detail": "Failed to parse waveform file"
}
```

---

## Value Formats

### Signal Values

| Type | Format | Example |
|------|--------|---------|
| Single bit | `"0"`, `"1"`, `"x"`, `"z"` | `"1"` |
| Multi-bit binary | Binary string | `"10110"` |
| Multi-bit hex | Hex with prefix | `"0x2f"` |
| Unknown bits | Contains `x` | `"1x0x"` |
| High-Z bits | Contains `z` | `"zzzz"` |

### Timestamps

All timestamps are integers representing simulation ticks.

```json
{
  "start_timestamp": 1000000,    // 1 million ticks
  "end_timestamp": 2000000       // 2 million ticks
}
```

The time unit depends on the simulation (typically nanoseconds).

### Ratios

All ratios are floats between 0.0 and 1.0.

```json
{
  "duty_cycle": 0.73,       // 73% of time was high
  "throughput_ratio": 0.85, // 85% utilization
  "stall_ratio": 0.15       // 15% stalled
}
```
