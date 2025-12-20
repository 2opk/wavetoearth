import typer
import requests
import sys
import subprocess
import time
from typing import Optional
from pathlib import Path
import json

app = typer.Typer(help="WaveToEarth: Agentic RTL Waveform Analyzer CLI")

SERVER_URL = "http://localhost:8000"

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

if __name__ == "__main__":
    app()
