# wavetoearth

**Agentic RTL Waveform Analyzer** - Bridge heavy VCD/FST files to LLM-based reasoning.

wavetoearth enables AI agents to analyze hardware simulation waveforms without deep RTL knowledge. Query signals by name, get structured JSON responses, understand hardware behavior.

## Quick Start

```bash
# Install
pip install -e .

# Analyze a signal
wavetoearth simulation.fst --signals "io_busy" --from 0 --to 1000000
```

```python
from wavetoearth import Waveform

with Waveform("simulation.fst") as wf:
    result = wf.inspect(["io_busy"], start=0, end=1000000)
    print(result["signals"][0]["raw"]["change_count"])
```

## Documentation

| Document | Description |
|----------|-------------|
| [AGENT_GUIDE.md](docs/AGENT_GUIDE.md) | **Start here** - Quick reference for LLM agents |
| [CONCEPTS.md](docs/CONCEPTS.md) | RTL/waveform basics for non-hardware engineers |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Complete CLI and Python API documentation |
| [OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md) | JSON response structure specification |
| [EXAMPLES.md](docs/EXAMPLES.md) | Real-world analysis scenarios |

## Features

### Signal Discovery
```bash
# Find signals by pattern
wavetoearth signals --pattern "busy"

# Wildcards in queries
wavetoearth file.fst --signals "*controller.io_busy" --from 0 --to 1000000
```

### Automatic Name Resolution
```bash
# These all find "TOP.chip.gemmini.ex_controller.io_busy"
--signals "io_busy"                              # Short name
--signals "ex_controller.io_busy"                # Partial path
--signals "TOP.chip.gemmini.ex_controller.io_busy"  # Full path
```

### Time or Cycle Ranges
```bash
# Timestamp mode
wavetoearth file.fst --signals "io_busy" --from 1000000 --to 2000000

# Cycle mode
wavetoearth file.fst --signals "io_busy" --clock "clock" --start-cycle 100 --end-cycle 200
```

### Structured Output
```bash
# JSON (default) - for programmatic use
wavetoearth file.fst --signals "io_busy" --from 0 --to 1000000

# Text - human readable
wavetoearth file.fst --signals "io_busy" --from 0 --to 1000000 --output text
```

## Installation

### Prerequisites
- Python 3.10+
- Conda (recommended)

### Setup
```bash
# Create environment
conda create -n wavetoearth python=3.10 -y
conda activate wavetoearth

# Install
pip install -r requirements.txt
pip install -e .
```

## Usage Examples

### Check Module Activity
```bash
wavetoearth simulation.fst --signals "gemmini.*busy" --from 0 --to 1000000000
```

### Compare Two Simulations
```python
from wavetoearth import Waveform

for file in ["baseline.fst", "optimized.fst"]:
    with Waveform(file) as wf:
        r = wf.inspect(["io_busy"], start=0, end=1000000000)
        changes = r["signals"][0]["raw"]["change_count"]
        print(f"{file}: {changes} state changes")
```

### Find Performance Bottlenecks
```python
with Waveform("simulation.fst") as wf:
    result = wf.inspect(
        signals=["*stall", "*busy", "*valid", "*ready"],
        start=50000000,
        end=100000000
    )

    for sig in result["signals"]:
        name = sig["signal"]
        changes = sig["raw"]["change_count"]
        print(f"{name}: {changes} transitions")
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────┐
│  VCD/FST    │────▶│ Rust Parser  │────▶│ Parquet │
│   File      │     │ (wavetoearth │     │ Shards  │
└─────────────┘     │    _core)    │     └────┬────┘
                    └──────────────┘          │
                                              ▼
┌─────────────┐     ┌──────────────┐     ┌─────────┐
│  CLI /      │◀────│   FastAPI    │◀────│ DuckDB  │
│  Python API │     │   Server     │     │         │
└─────────────┘     └──────────────┘     └─────────┘
```

## Output Example

```json
{
  "range": {"mode": "time", "start": 0, "end": 1000000},
  "signals": [{
    "signal": "io_busy",
    "resolved_signal": "TOP.chip.gemmini.ex_controller.io_busy",
    "match": "suffix",
    "raw": {
      "initial_value": "0",
      "final_value": "1",
      "change_count": 42,
      "unique_values": ["0", "1"],
      "first_change": 50000,
      "last_change": 990000,
      "changes": [[50000, "1"], [60000, "0"], ...]
    },
    "analysis": {
      "description": "Signal toggled 42 times between active and idle states."
    }
  }],
  "relations": []
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAVETOEART_SERVER_URL` | `http://localhost:8000` | Server address |
| `WAVETOEART_USE_SHARDS` | `1` | Enable sharded Parquet |
| `WAVETOEART_CACHE_TAG` | `v1` | Cache version identifier |

## Contributing

See [claude_handoff.md](claude_handoff.md) for development context.

## License

MIT
