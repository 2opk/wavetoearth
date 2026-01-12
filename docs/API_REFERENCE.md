# API Reference

Complete reference for wavetoearth CLI and Python API.

## CLI Usage

### Basic Syntax

```bash
wavetoearth <file_path> --signals <signal_list> [options]
```

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `file_path` | path | Yes | VCD or FST file path. Supports glob patterns (`*.fst`) |

### Options

#### Signal Selection

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--signals` | string | Required | Comma-separated signal names or patterns |
| `--max-expand` | int | 200 | Maximum signals from wildcard expansion |

**Signal name formats:**
```bash
# Exact match (full path)
--signals "TOP.chip.gemmini.io_busy"

# Suffix match (partial path)
--signals "gemmini.io_busy"

# Short name (just signal name)
--signals "io_busy"

# Wildcard patterns
--signals "*busy"           # ends with "busy"
--signals "gemmini.*"       # starts with "gemmini."
--signals "*controller*"    # contains "controller"

# Multiple signals
--signals "io_busy,io_stall,io_valid"
```

#### Time Range (Timestamp Mode)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--from`, `--start` | int | None | Start timestamp (inclusive) |
| `--to`, `--end` | int | None | End timestamp (inclusive) |

```bash
# Query timestamps 1M to 2M
wavetoearth file.fst --signals "io_busy" --from 1000000 --to 2000000
```

#### Time Range (Cycle Mode)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--clock` | string | None | Clock signal name for cycle conversion |
| `--start-cycle` | int | None | Start cycle number |
| `--end-cycle` | int | None | End cycle number |
| `--edge` | string | "rising" | Clock edge: "rising" or "falling" |

```bash
# Query cycles 100-200 based on clock signal
wavetoearth file.fst --signals "io_busy" \
  --clock "clock" --start-cycle 100 --end-cycle 200
```

**Note:** Cycle mode requires all three: `--clock`, `--start-cycle`, `--end-cycle`

#### Output Control

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | string | "json" | Output format: "json" or "text" |
| `--include-changes` | flag | True | Include raw change list |
| `--no-include-changes` | flag | - | Exclude raw change list (faster) |
| `--max-changes` | int | 200 | Max changes to return per signal |
| `--max-unique` | int | 32 | Max unique values to list |

```bash
# JSON output (default, best for programmatic use)
wavetoearth file.fst --signals "io_busy" --from 0 --to 1000000

# Text output (human-readable)
wavetoearth file.fst --signals "io_busy" --from 0 --to 1000000 --output text

# Skip changes for faster response
wavetoearth file.fst --signals "io_busy" --from 0 --to 1000000 --no-include-changes
```

### Subcommands

#### `serve`
Start the server daemon manually.

```bash
wavetoearth serve [--port 8000] [--host 0.0.0.0] [--background]
```

#### `load`
Load a file into running server.

```bash
wavetoearth load /path/to/file.fst
```

#### `signals`
List available signals.

```bash
wavetoearth signals                    # List all (first 50)
wavetoearth signals --pattern "busy"   # Filter by regex
```

#### `query`
Query single signal (low-level).

```bash
wavetoearth query --signal "io_busy" --start 0 --end 1000000
```

#### `analyze`
Get semantic analysis of signal.

```bash
wavetoearth analyze --signal "io_busy" --start 0 --end 1000000
```

---

## Python API

### Quick Start

```python
from wavetoearth import Waveform, Client, inspect, probe

# Method 1: Waveform context manager (recommended)
with Waveform("/path/to/file.fst") as wf:
    result = wf.inspect(["io_busy"], start=0, end=1000000)

# Method 2: One-liner
result = inspect("/path/to/file.fst", ["io_busy"], start=0, end=1000000)

# Method 3: Client for multiple files
with Client() as client:
    client.ensure_loaded("/path/to/file1.fst")
    r1 = client.inspect(["io_busy"], start=0, end=1000000)

    client.ensure_loaded("/path/to/file2.fst")
    r2 = client.inspect(["io_busy"], start=0, end=1000000)
```

### Classes

#### `Waveform`

High-level interface for single file analysis.

```python
class Waveform:
    def __init__(
        self,
        file_path: str,           # Path to VCD/FST file
        server_url: str = None,   # Server URL (default: localhost:8000)
        auto_start: bool = True,  # Auto-start server if not running
        timeout: float = 10.0     # Server startup timeout
    )
```

**Methods:**

| Method | Description |
|--------|-------------|
| `inspect(signals, **kwargs)` | Get raw data + analysis |
| `probe(signals, **kwargs)` | Get summary without analysis |
| `query(signal, start, end)` | Query single signal changes |
| `analyze(signal, start, end)` | Get semantic analysis only |
| `query_cycles(signal, clock, start_cycle, end_cycle, edge)` | Query by cycles |
| `signals(pattern=None)` | List available signals |
| `close()` | Clean up resources |

#### `Client`

Low-level client for advanced use cases.

```python
class Client:
    def __init__(
        self,
        server_url: str = None,
        auto_start: bool = True,
        timeout: float = 10.0
    )
```

**Methods:**

| Method | Description |
|--------|-------------|
| `ensure_server()` | Start server if needed |
| `ensure_loaded(file_path)` | Load file into server |
| `inspect(signals, file_path=None, **kwargs)` | Inspect signals |
| `probe(signals, file_path=None, **kwargs)` | Probe signals |
| `signals(pattern=None)` | List signals |
| `close()` | Clean up |

### Method Parameters

#### `inspect()` / `probe()`

```python
def inspect(
    signals: Union[str, List[str]],  # Signal names or patterns

    # Time range (timestamp mode)
    start: int = None,               # Start timestamp
    end: int = None,                 # End timestamp

    # Time range (cycle mode)
    clock: str = None,               # Clock signal name
    start_cycle: int = None,         # Start cycle
    end_cycle: int = None,           # End cycle
    edge: str = "rising",            # "rising" or "falling"

    # Output control
    include_changes: bool = True,    # Include change list
    max_changes: int = 200,          # Max changes per signal
    max_unique: int = 32,            # Max unique values
    max_expand: int = 200,           # Max wildcard expansion

    # File (for Client only)
    file_path: str = None            # Override loaded file
) -> Dict[str, Any]
```

### Signal Name Resolution

wavetoearth resolves signal names in this order:

1. **Exact match**: Full hierarchical path
2. **Suffix match**: Matches end of full path
3. **Fuzzy match**: Best partial match

```python
# These all find "TOP.chip.gemmini.ex_controller.io_busy"
wf.inspect(["TOP.chip.gemmini.ex_controller.io_busy"])  # exact
wf.inspect(["ex_controller.io_busy"])                    # suffix
wf.inspect(["io_busy"])                                  # fuzzy

# Wildcards expand to multiple signals
wf.inspect(["*busy"])           # All signals ending with "busy"
wf.inspect(["gemmini.*"])       # All signals under gemmini
```

### Error Handling

```python
from wavetoearth import Waveform

try:
    with Waveform("/path/to/file.fst") as wf:
        result = wf.inspect(["nonexistent_signal"])
except RuntimeError as e:
    if "No signals matched" in str(e):
        print("Signal not found")
    elif "Failed to start server" in str(e):
        print("Server startup failed")
    else:
        raise
```

Common errors:

| Error | Cause | Solution |
|-------|-------|----------|
| `No signals matched pattern: X` | Signal not found | Check signal name, use wildcards |
| `Pattern 'X' matched N signals (> max)` | Too many matches | Be more specific or increase `max_expand` |
| `Failed to start server` | Server timeout | Check file path, increase timeout |
| `Could not connect to server` | Server not running | Use `auto_start=True` or start manually |

---

## REST API (Advanced)

The server exposes these endpoints at `http://localhost:8000`:

### `GET /health`
Check server status.

**Response:**
```json
{
  "status": "ok",
  "loaded_file": "/path/to/file.fst"
}
```

### `POST /load`
Load a waveform file.

**Request:**
```json
{
  "file_path": "/path/to/file.fst"
}
```

### `GET /signals`
List signals.

**Query params:** `pattern` (optional regex)

**Response:**
```json
{
  "signals": ["TOP.chip.io_busy", "TOP.chip.io_valid", ...]
}
```

### `POST /inspect`
Main analysis endpoint.

**Request:**
```json
{
  "signals": ["io_busy", "io_valid"],
  "start": 0,
  "end": 1000000,
  "include_changes": true,
  "max_changes": 200
}
```

**Response:** See [OUTPUT_SCHEMA.md](OUTPUT_SCHEMA.md)

### `POST /probe`
Lightweight summary (no analysis).

Same request format as `/inspect`, returns simplified response.

### `POST /query`
Query single signal.

**Request:**
```json
{
  "signal": "io_busy",
  "start": 0,
  "end": 1000000
}
```

### `POST /analyze`
Semantic analysis only.

**Request:**
```json
{
  "signal": "io_busy",
  "start": 0,
  "end": 1000000
}
```

### `POST /query_cycles`
Query by clock cycles.

**Request:**
```json
{
  "signal": "io_busy",
  "clock": "clock",
  "start_cycle": 100,
  "end_cycle": 200,
  "edge": "rising"
}
```
