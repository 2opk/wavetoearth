import typer
import requests
import sys
import subprocess
import time
import os
import argparse
import socket
from typing import Optional, List, Dict, Any
from pathlib import Path
from urllib.parse import urlparse
import json

app = typer.Typer(help="WaveToEarth: Agentic RTL Waveform Analyzer CLI")

SUBCOMMANDS = {
    "serve",
    "load",
    "signals",
    "query",
    "analyze",
    "query-cycles",
    "probe",
    "inspect",
}

SERVER_URL = os.environ.get("WAVETOEART_SERVER_URL", "http://localhost:8000")

def _parse_server_url() -> Dict[str, Any]:
    parsed = urlparse(SERVER_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    return {"host": host, "port": port}

def _set_server_url(url: str) -> None:
    global SERVER_URL
    SERVER_URL = url

def _alloc_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def _server_health(timeout: float = 0.5) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

def _start_server():
    addr = _parse_server_url()
    app_dir = str(Path(__file__).parent)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server:app",
        "--app-dir",
        app_dir,
        "--host",
        addr["host"],
        "--port",
        str(addr["port"]),
    ]
    subprocess.Popen(cmd, cwd=app_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _ensure_server(timeout: float = 10.0) -> Dict[str, Any]:
    health = _server_health()
    if health:
        return health
    _start_server()
    deadline = time.time() + timeout
    while time.time() < deadline:
        health = _server_health()
        if health:
            return health
        time.sleep(0.2)
    raise RuntimeError("Failed to start server")

def _ensure_loaded(file_path: str) -> None:
    file_path = str(Path(file_path).resolve())
    health = _ensure_server()
    if health.get("loaded_file") == file_path:
        return
    resp = requests.post(f"{SERVER_URL}/load", json={"file_path": file_path})
    resp.raise_for_status()

def _parse_signals(signals: str) -> List[str]:
    return [s.strip() for s in signals.split(",") if s.strip()]

def _run_inspect(
    file_path: str,
    signals: str,
    start: Optional[int],
    end: Optional[int],
    clock: Optional[str],
    start_cycle: Optional[int],
    end_cycle: Optional[int],
    edge: str,
    include_changes: bool,
    max_changes: int,
    max_unique: int,
):
    signal_list = _parse_signals(signals)
    if not signal_list:
        raise RuntimeError("signals must be a comma-separated list")

    _ensure_loaded(file_path)

    payload = {
        "signals": signal_list,
        "start": start,
        "end": end,
        "clock": clock,
        "start_cycle": start_cycle,
        "end_cycle": end_cycle,
        "edge": edge,
        "include_changes": include_changes,
        "max_changes": max_changes,
        "max_unique": max_unique,
    }
    resp = requests.post(f"{SERVER_URL}/inspect", json=payload)
    if resp.status_code == 404:
        new_port = _alloc_free_port()
        _set_server_url(f"http://localhost:{new_port}")
        _ensure_loaded(file_path)
        resp = requests.post(f"{SERVER_URL}/inspect", json=payload)
    if not resp.ok:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"{resp.status_code} {detail}")
    return resp.json()

def _render_inspect_output(data: Dict[str, Any], include_changes: bool) -> None:
    range_info = data.get("range", {})
    mode = range_info.get("mode", "time")
    if mode == "cycle":
        typer.echo(
            f"Range: cycles {range_info.get('start_cycle')}..{range_info.get('end_cycle')} "
            f"({range_info.get('clock')} {range_info.get('edge')}) "
            f"ts={range_info.get('start_timestamp')}..{range_info.get('end_timestamp')}"
        )
    else:
        typer.echo(f"Range: ts={range_info.get('start')}..{range_info.get('end')}")

    for item in data.get("signals", []):
        raw = item.get("raw", {})
        analysis = item.get("analysis", {})
        typer.echo("")
        typer.echo(f"Signal: {item.get('signal')} (id={item.get('signal_id')})")
        typer.echo(f"  initial: {raw.get('initial_value')}  final: {raw.get('final_value')}")
        typer.echo(f"  changes: {raw.get('change_count')}  unique: {raw.get('unique_values')}")
        typer.echo(f"  first_change: {raw.get('first_change')}  last_change: {raw.get('last_change')}")
        if include_changes and raw.get("changes") is not None:
            typer.echo("  raw_changes:")
            for t, v in raw.get("changes", []):
                typer.echo(f"    @{t}: {v}")
            if raw.get("truncated"):
                typer.echo("  ... truncated ...")
        typer.echo(f"  analysis: {analysis.get('description')}")

    relations = data.get("relations", [])
    if relations:
        typer.echo("")
        typer.echo("Relations:")
        for rel in relations:
            if rel.get("type") == "handshake":
                typer.echo(
                    f"  handshake {rel.get('valid')} / {rel.get('ready')}: "
                    f"throughput={rel.get('throughput_ratio')}, stall={rel.get('stall_ratio')}"
                )
            elif rel.get("type") == "coactivity":
                typer.echo(
                    f"  coactivity {rel.get('a')} / {rel.get('b')}: "
                    f"same_state_ratio={rel.get('same_state_ratio')}"
                )

def _parse_default_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wavetoearth",
        description="WaveToEarth: Agentic RTL Waveform Analyzer",
    )
    parser.add_argument("file_paths", nargs="+", help="VCD/FST files (glob OK)")
    parser.add_argument("--from", "--start", dest="from_ts", type=int, default=None)
    parser.add_argument("--to", "--end", dest="to_ts", type=int, default=None)
    parser.add_argument("--signals", required=True)
    parser.add_argument("--clock", default=None)
    parser.add_argument("--start-cycle", dest="start_cycle", type=int, default=None)
    parser.add_argument("--end-cycle", dest="end_cycle", type=int, default=None)
    parser.add_argument("--edge", default="rising")
    parser.add_argument("--include-changes", dest="include_changes", action="store_true", default=True)
    parser.add_argument("--no-include-changes", dest="include_changes", action="store_false")
    parser.add_argument("--max-changes", dest="max_changes", type=int, default=200)
    parser.add_argument("--max-unique", dest="max_unique", type=int, default=32)
    parser.add_argument("--output", default="json")
    return parser.parse_args(argv)

def _run_default(argv: List[str]) -> int:
    try:
        args = _parse_default_args(argv)
        results = []
        for file_path in args.file_paths:
            abs_path = str(Path(file_path).resolve())
            data = _run_inspect(
                abs_path,
                args.signals,
                args.from_ts,
                args.to_ts,
                args.clock,
                args.start_cycle,
                args.end_cycle,
                args.edge,
                args.include_changes,
                args.max_changes,
                args.max_unique,
            )
            results.append({"file": abs_path, "data": data})

        if args.output == "json":
            if len(results) == 1:
                typer.echo(json.dumps(results[0]["data"], indent=2))
            else:
                typer.echo(json.dumps(results, indent=2))
        else:
            for item in results:
                if len(results) > 1:
                    typer.echo("")
                    typer.echo(f"File: {item['file']}")
                _render_inspect_output(item["data"], args.include_changes)
        return 0
    except Exception as e:
        typer.echo(f"Error: {e}")
        return 1

def main_cli() -> None:
    argv = sys.argv[1:]
    if not argv:
        app()
        return
    first_token = next((arg for arg in argv if not arg.startswith("-")), None)
    if first_token in SUBCOMMANDS:
        app()
        return
    raise SystemExit(_run_default(argv))

@app.command()
def serve(port: int = 8000, host: str = "0.0.0.0", background: bool = False):
    """
    Start the WaveToEarth server daemon.
    """
    cmd = [sys.executable, "-m", "uvicorn", "server:app", "--host", host, "--port", str(port)]

    if background:
        # Start in background
        subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
        typer.echo(f"Server started in background on {host}:{port}")
    else:
        # Run in foreground
        subprocess.run(cmd, cwd=str(Path(__file__).parent))

@app.command()
def load(file_path: str):
    """
    Load a VCD file into the running server.
    """
    url = f"{SERVER_URL}/load"
    try:
        abs_path = str(Path(file_path).resolve())
        response = requests.post(url, json={"file_path": abs_path})
        response.raise_for_status()
        typer.echo(f"Success: {response.json()}")
    except requests.exceptions.ConnectionError:
        typer.echo("Error: Could not connect to server. Is it running? (Try 'wavetoearth serve')")
    except Exception as e:
        typer.echo(f"Error: {e}")

@app.command()
def signals(pattern: Optional[str] = None):
    """
    List available signals, optionally filtered by regex pattern.
    """
    url = f"{SERVER_URL}/signals"
    params = {}
    if pattern:
        params["pattern"] = pattern

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        sigs = response.json()["signals"]
        typer.echo(f"Found {len(sigs)} signals.")
        for s in sigs[:50]: # Limit output
            typer.echo(s)
        if len(sigs) > 50:
            typer.echo(f"... and {len(sigs)-50} more.")
    except Exception as e:
        typer.echo(f"Error: {e}")

@app.command()
def query(signal: str, start: int = 0, end: int = 1000, output: str = "json"):
    """
    Query signal values in a time range.
    """
    url = f"{SERVER_URL}/query"
    payload = {
        "signal": signal,
        "start": start,
        "end": end
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        if output == "json":
            typer.echo(json.dumps(data, indent=2))
        else:
            # Simple text format
            typer.echo(f"Signal: {data['signal']}")
            typer.echo(f"Range: {data['start']} - {data['end']}")
            typer.echo("Changes:")
            for t, v in data['changes']:
                typer.echo(f"  @{t}: {v}")

    except Exception as e:
        typer.echo(f"Error: {e}")

@app.command()
def analyze(signal: str, start: int = 0, end: int = 1000):
    """
    Get a semantic summary of the signal in the time range.
    """
    url = f"{SERVER_URL}/analyze"
    payload = {
        "signal": signal,
        "start": start,
        "end": end
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        typer.echo(f"--- Analysis of {signal} ---")
        typer.echo(f"Time Window: {data['window'][0]} to {data['window'][1]}")
        typer.echo(f"Initial Value: {data['initial_value']}")
        typer.echo(f"Stability: {'Stable' if data['is_stable'] else 'Changing'}")
        typer.echo(f"Description: {data['description']}")

    except Exception as e:
        typer.echo(f"Error: {e}")

@app.command()
def query_cycles(
    signal: str,
    clock: str,
    start_cycle: int,
    end_cycle: int,
    edge: str = "rising",
    output: str = "json",
):
    """
    Query signal values in a cycle range using a clock signal.
    """
    url = f"{SERVER_URL}/query_cycles"
    payload = {
        "signal": signal,
        "clock": clock,
        "start_cycle": start_cycle,
        "end_cycle": end_cycle,
        "edge": edge,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        if output == "json":
            typer.echo(json.dumps(data, indent=2))
        else:
            typer.echo(f"Signal: {data['signal']}")
            typer.echo(f"Clock: {data['clock']} ({data['edge']})")
            typer.echo(f"Cycle Range: {data['start_cycle']} - {data['end_cycle']}")
            typer.echo(f"Time Range: {data['start_timestamp']} - {data['end_timestamp']}")
            typer.echo("Changes:")
            for t, v in data['changes']:
                typer.echo(f"  @{t}: {v}")

    except Exception as e:
        typer.echo(f"Error: {e}")

@app.command()
def probe(
    signals: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    clock: Optional[str] = None,
    start_cycle: Optional[int] = None,
    end_cycle: Optional[int] = None,
    edge: str = "rising",
    include_changes: bool = False,
    max_changes: int = 200,
    max_unique: int = 32,
    output: str = "json",
):
    """
    Summarize multiple signals for a time or cycle range.
    """
    url = f"{SERVER_URL}/probe"
    signal_list = [s.strip() for s in signals.split(",") if s.strip()]
    if not signal_list:
        typer.echo("Error: signals must be a comma-separated list")
        raise typer.Exit(code=1)

    payload = {
        "signals": signal_list,
        "start": start,
        "end": end,
        "clock": clock,
        "start_cycle": start_cycle,
        "end_cycle": end_cycle,
        "edge": edge,
        "include_changes": include_changes,
        "max_changes": max_changes,
        "max_unique": max_unique,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        if output == "json":
            typer.echo(json.dumps(data, indent=2))
            return

        range_info = data.get("range", {})
        mode = range_info.get("mode", "time")
        if mode == "cycle":
            typer.echo(
                f"Range: cycles {range_info.get('start_cycle')}..{range_info.get('end_cycle')} "
                f"({range_info.get('clock')} {range_info.get('edge')}) "
                f"ts={range_info.get('start_timestamp')}..{range_info.get('end_timestamp')}"
            )
        else:
            typer.echo(f"Range: ts={range_info.get('start')}..{range_info.get('end')}")

        for item in data.get("signals", []):
            typer.echo("")
            typer.echo(f"Signal: {item.get('signal')} (id={item.get('signal_id')})")
            typer.echo(f"  initial: {item.get('initial_value')}  final: {item.get('final_value')}")
            typer.echo(f"  changes: {item.get('change_count')}  unique: {item.get('unique_values')}")
            typer.echo(f"  first_change: {item.get('first_change')}  last_change: {item.get('last_change')}")
            if include_changes and item.get("changes") is not None:
                typer.echo("  sample_changes:")
                for t, v in item.get("changes"):
                    typer.echo(f"    @{t}: {v}")
                if item.get("truncated"):
                    typer.echo("  ... truncated ...")
    except Exception as e:
        typer.echo(f"Error: {e}")

@app.command()
def inspect(
    file_paths: List[str],
    signals: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    clock: Optional[str] = None,
    start_cycle: Optional[int] = None,
    end_cycle: Optional[int] = None,
    edge: str = "rising",
    include_changes: bool = True,
    max_changes: int = 200,
    max_unique: int = 32,
    output: str = "json",
):
    """
    Inspect signals with raw data + analysis in one call.
    """
    try:
        results = []
        for file_path in file_paths:
            abs_path = str(Path(file_path).resolve())
            data = _run_inspect(
                abs_path,
                signals,
                start,
                end,
                clock,
                start_cycle,
                end_cycle,
                edge,
                include_changes,
                max_changes,
                max_unique,
            )
            results.append({"file": abs_path, "data": data})

        if output == "json":
            if len(results) == 1:
                typer.echo(json.dumps(results[0]["data"], indent=2))
            else:
                typer.echo(json.dumps(results, indent=2))
        else:
            for item in results:
                if len(results) > 1:
                    typer.echo("")
                    typer.echo(f"File: {item['file']}")
                _render_inspect_output(item["data"], include_changes)
    except Exception as e:
        typer.echo(f"Error: {e}")

if __name__ == "__main__":
    main_cli()
