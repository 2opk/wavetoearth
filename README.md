# wavetoearth

**Agentic RTL Waveform Analyzer** - MCP Server for AI-powered hardware debugging.

wavetoearth enables Claude Code to analyze VCD/FST simulation waveforms directly. Find deadlocks, trace signal changes, debug hardware hangs - all through natural language.

```
VCD/FST File → Rust Parser → DuckDB → MCP Server → Claude Code
```

## Installation

### Quick Install (Recommended)

One command to install everything (Rust, Python deps, Claude Code MCP):

```bash
curl -fsSL https://raw.githubusercontent.com/2opk/wavetoearth/master/install.sh | bash
```

Or manually:

```bash
git clone https://github.com/2opk/wavetoearth.git
cd wavetoearth
./install.sh
```

### Prerequisites

- **Python 3.10+**
- **Rust** (installed automatically by script, or manually via [rustup](https://rustup.rs))
- **Claude Code** CLI

### Manual Installation

If you prefer step-by-step:

```bash
# 1. Clone
git clone https://github.com/2opk/wavetoearth.git
cd wavetoearth

# 2. Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 3. Build Rust extension
pip install maturin
cd wavetoearth_core && maturin develop --release && cd ..

# 4. Install Python package
pip install -e .

# 5. Add to Claude Code
claude mcp add wavetoearth --scope user -- python3 $(pwd)/mcp_server.py
```

### Verify Installation

In Claude Code, type `/mcp` to see the server status. You should see `wavetoearth` listed with 16 tools.

### Start Using

```
You: Load /path/to/simulation.vcd and find where the system hangs

Claude: [Uses wave_load, wave_find_stall, wave_last_activity tools]
        Found hang at cycle 2270941. The io_cmd_valid signal stopped
        at timestamp 424925000 and never recovered...
```

## Available MCP Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `wave_load` | Load VCD/FST file | Start analysis |
| `wave_unload` | Free memory | Cleanup |
| `wave_status` | Server status | Check loaded file |
| `wave_signals` | List signals (flat/tree) | Signal discovery |
| `wave_query` | Query by timestamp | Value inspection |
| `wave_query_cycles` | Query by clock cycle | Cycle-based analysis |
| `wave_probe` | Multi-signal summary | Quick overview |
| `wave_inspect` | Deep semantic analysis | Handshake detection |
| `wave_find_stall` | Find signal stalls | **Hang debugging** |
| `wave_last_activity` | Last change times | **Hang debugging** |
| `wave_compare_points` | Compare two timestamps | State diffing |
| `wave_find_event` | Find specific values | Event location |

## Usage Examples

### Debug a Hang

```
You: My simulation hangs at cycle 2270941. Load /tmp/sim.vcd and help me find why.

Claude: Let me analyze the waveform...

[wave_load /tmp/sim.vcd]
Loaded 46,342 signals, time range 0 - 471,415,500

[wave_find_stall io_cmd_valid]
Found stall: io_cmd_valid held at 0 from 424,925,000 to end of trace (46.5M units)
This extends to end of trace - POTENTIAL HANG

[wave_last_activity pattern=".*valid.*"]
Last activity on valid signals:
- serial_tl_out_d_valid: 471,161,000
- l2_sinkC_valid: 471,153,000
- io_cmd_valid: 424,925,000 ← Stopped much earlier!

The hang appears to be in the command path. io_cmd_valid stopped
46M time units before other signals...
```

### Analyze Handshakes

```
You: Check the ready/valid handshake on the memory interface

Claude: [wave_inspect mem_req_valid, mem_req_ready, mem_resp_valid]

Handshake Analysis:
- Throughput: 78% (both high)
- Stall ratio: 15% (valid high, ready low)
- Backpressure detected at cycles 1000-1500
```

### Signal Discovery

```
You: What gemmini signals are available?

Claude: [wave_signals pattern=".*gemmini.*" tree=true]

TOP (41,968 signals)
└── TestDriver
    └── testHarness
        └── chiptop0
            └── system
                └── tile_prci_domain
                    └── gemmini
                        ├── ex_controller (234 signals)
                        ├── load_controller (189 signals)
                        └── store_controller (156 signals)
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  VCD/FST    │────▶│ Rust Parser  │────▶│   Parquet   │
│   File      │     │ (wavetoearth │     │   Shards    │
│  (GB-sized) │     │    _core)    │     │  (cached)   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│ Claude Code │◀────│  MCP Server  │◀────│   DuckDB    │
│   (Agent)   │ MCP │  (stdio)     │ SQL │  (in-mem)   │
└─────────────┘     └──────────────┘     └─────────────┘
```

### Key Features

- **High Performance**: Rust-based VCD/FST parsing with SIMD acceleration
- **Lazy Loading**: Parquet shards cached for repeated queries
- **Fuzzy Matching**: `io_valid` automatically resolves to `TOP.chip.gemmini.io_valid`
- **Context Efficient**: Pagination and limits prevent context overflow
- **Semantic Analysis**: Automatic handshake and stall detection

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAVETOEART_USE_SHARDS` | `1` | Enable sharded Parquet |
| `WAVETOEART_SHARDS` | `auto` | Number of shards |
| `WAVETOEART_CACHE_TAG` | `v2` | Cache version |

### MCP Server Configuration

For project-specific settings, create `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "wavetoearth": {
      "command": "wavetoearth-mcp",
      "env": {
        "WAVETOEART_CACHE_TAG": "project_v1"
      }
    }
  }
}
```

## Requirements

- Python 3.10+
- Rust toolchain (for building from source)
- ~2GB RAM per 1GB of VCD file

## Troubleshooting

### MCP Server Not Showing in `/mcp`

1. Check if the command works directly:
   ```bash
   wavetoearth-mcp
   # Should start and wait for MCP handshake
   ```

2. Verify configuration:
   ```bash
   claude mcp list
   ```

3. Restart Claude Code after configuration changes

### Out of Memory

Large VCD files (>10GB) may require:
```bash
export WAVETOEART_SHARDS=64
export WAVETOEART_MAX_SHARDS=128
```

### Slow Initial Load

First load converts VCD→Parquet (cached). Subsequent loads are fast:
- Initial: ~30s per GB
- Cached: ~1s regardless of size

## Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT
