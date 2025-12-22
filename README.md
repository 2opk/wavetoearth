# wavetoearth: Agentic RTL Waveform Analyzer

High-performance, Agent-friendly RTL waveform analysis tool designed to bridge the gap between heavy VCD files and LLM-based reasoning.

## Logic
`wavetoearth` uses a Client-Server architecture, but the server is now **transparent** to the user.
The CLI auto-starts the daemon, loads the file, and returns structured JSON for agentic analysis.

## Installation

### Prerequisites
- Conda (Miniconda or Anaconda)

### Setup
```bash
# 1. Create Conda Environment
conda create -n wavetoearth python=3.10 -y
conda activate wavetoearth

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Install CLI entrypoint
pip install -e .
```

## Usage (Transparent Mode)
One command = start server + load file + inspect.

```bash
wavetoearth /path/to/file.vcd \
  --from 750000 --to 800000 \
  --signals "chiptop.gemmini.io_busy,chiptop.core.csr.io_stall"
```

Cycle-based range:
```bash
wavetoearth /path/to/file.fst \
  --clock chiptop.clock --start-cycle 100 --end-cycle 200 \
  --signals "chiptop.gemmini.io_busy"
```

### Multi-file (shell glob)
```bash
wavetoearth /path/to/*.fst --from 750000 --to 800000 --signals "chiptop.gemmini.io_busy"
```

## Usage (Explicit Mode)

### 1. Start the Server
Run this in a separate terminal (or background):
```bash
python cli.py serve
```

### 2. Load a Waveform
Tell the server to load a specific VCD file.
```bash
python cli.py load /path/to/your/file.vcd
```

### 3. Query Signals
Get values for a signal in a specific time range.
```bash
python cli.py query --signal "chiptop.gemmini.io_busy" --start 1000 --end 2000
```

### Search Signals
```bash
python cli.py signals --pattern "busy"
```

## Environment
- `WAVETOEART_SERVER_URL`: Override server URL (default `http://localhost:8000`)
