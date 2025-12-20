# wavetoearth: Agentic RTL Waveform Analyzer

High-performance, Agent-friendly RTL waveform analysis tool designed to bridge the gap between heavy VCD files and LLM-based reasoning.

## Logic
`wavetoearth` uses a Client-Server architecture.
1. **Server (`serve`)**: Loads the heavy VCD file into RAM (using efficient Numpy arrays).
2. **Client (`query`)**: Sends lightweight JSON queries to the server to get instant signal values.

This allows an LLM Agent to explore the waveform interactively without waiting for parsing every time.

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
```

## Usage

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
