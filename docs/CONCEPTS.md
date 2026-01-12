# RTL Waveform Concepts for Agents

This document explains fundamental concepts needed to use wavetoearth effectively, **without requiring prior RTL or hardware knowledge**.

## What is a Waveform File?

A **waveform file** (VCD or FST) is a recording of how digital signals change over time during a hardware simulation.

Think of it like a **flight data recorder** for chips:
- Records every signal's value at every moment
- Used to debug why hardware behaved a certain way
- Can contain millions of signals and billions of time points

### File Formats

| Format | Full Name | Characteristics |
|--------|-----------|-----------------|
| **VCD** | Value Change Dump | Text-based, human-readable, large files |
| **FST** | Fast Signal Trace | Binary, compressed, 10-50x smaller than VCD |

wavetoearth supports both formats identically.

## Signals

A **signal** is a named wire or register in the hardware design.

### Signal Names are Hierarchical

Signal names use dots (`.`) to show the hardware hierarchy:

```
TOP.TestDriver.testHarness.chiptop.system.tile.gemmini.ex_controller.io_busy
│   │          │           │       │      │    │       │             └─ signal name
│   │          │           │       │      │    │       └─ module: execution controller
│   │          │           │       │      │    └─ module: Gemmini accelerator
│   │          │           │       │      └─ module: processor tile
│   │          │           │       └─ module: system
│   │          │           └─ module: chip top
│   │          └─ module: test harness
│   └─ module: test driver
└─ root
```

### You Don't Need Full Names

wavetoearth automatically resolves short names:

| You type | wavetoearth finds |
|----------|-------------------|
| `io_busy` | `...gemmini.ex_controller.io_busy` |
| `gemmini.io_busy` | `...gemmini.io_busy` |
| `*controller.io_busy` | All signals matching pattern |

### Signal Values

Digital signals typically have these values:

| Value | Meaning |
|-------|---------|
| `0` | Logic low (false, off) |
| `1` | Logic high (true, on) |
| `x` | Unknown (not yet initialized) |
| `z` | High impedance (disconnected) |

Multi-bit signals (buses) show as binary or hex: `10110`, `0x2f`

## Time in Waveforms

### Timestamps

Time is measured in **simulation ticks** (typically nanoseconds or picoseconds).

```
timestamp: 50000000  →  50 million ticks (e.g., 50ms if 1 tick = 1ns)
```

### Cycles

Hardware operates in **clock cycles**. One cycle = one rising edge of the clock signal.

```
Clock:    _/‾\_/‾\_/‾\_/‾\_/‾\_
Cycle:     0   1   2   3   4
```

wavetoearth can convert between timestamps and cycles using `--clock`.

## Common Signal Patterns

### Control Signals

| Pattern | Meaning | Example |
|---------|---------|---------|
| `*_busy` | Module is working | `ex_controller.io_busy` |
| `*_valid` | Data is available | `io_cmd_valid` |
| `*_ready` | Receiver can accept | `io_cmd_ready` |
| `*_stall` | Pipeline is blocked | `io_csr_stall` |
| `*_fire` | Transaction occurs | `cmd_fire` |

### Handshake Protocol

Many hardware interfaces use **valid/ready handshake**:

```
valid: ____/‾‾‾‾‾‾‾‾\____
ready: ________/‾‾‾‾\____
fire:  ________/‾‾‾‾\____  (valid AND ready)
              ↑
        Transaction happens here
```

- `valid=1`: Sender has data
- `ready=1`: Receiver can accept
- `fire` (valid AND ready): Actual transfer

### Busy/Idle Pattern

```
busy:  ___/‾‾‾‾‾‾‾‾‾‾‾‾‾\___/‾‾‾‾‾‾‾\___
           ↑ working      ↑ idle    ↑ working again
```

- High busy ratio = good utilization
- Long idle gaps = inefficiency

## Key Metrics

### Change Count

How many times a signal toggled in the time range.

```
signal: 0 → 1 → 0 → 1 → 1 → 0
changes: 4 (value changed 4 times)
```

### Unique Values

The set of distinct values the signal took.

```
signal: 0 → 1 → 0 → 1 → 0
unique_values: [0, 1]  (only two distinct values)
```

### First/Last Change

When the signal first and last changed within the queried range.

## Putting It Together

When you query:
```bash
wavetoearth file.fst --signals "gemmini.io_busy" --from 1000000 --to 2000000
```

You're asking: "Show me how the Gemmini accelerator's busy signal behaved between timestamp 1M and 2M."

The response tells you:
- **initial_value**: What was it at timestamp 1M?
- **final_value**: What was it at timestamp 2M?
- **change_count**: How many times did it toggle?
- **changes**: List of (timestamp, new_value) pairs
- **analysis**: Human-readable summary

## Glossary

| Term | Definition |
|------|------------|
| **RTL** | Register Transfer Level - hardware design abstraction |
| **Simulation** | Running hardware design in software to verify behavior |
| **Waveform** | Time-series recording of all signals during simulation |
| **Timestamp** | Absolute time point in simulation ticks |
| **Cycle** | One period of the clock signal |
| **Handshake** | Protocol where valid/ready signals coordinate transfers |
| **Hierarchy** | Nested structure of hardware modules |
| **Resolve** | Finding full signal name from partial/short name |
