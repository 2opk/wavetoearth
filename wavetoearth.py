import os
import sys
import time
import json
import socket
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Sequence, Union
from urllib.parse import urlparse

import requests

DEFAULT_SERVER_URL = os.environ.get("WAVETOEART_SERVER_URL", "http://localhost:8000")


def _parse_server_url(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    return {"host": host, "port": port}


def _alloc_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _normalize_signals(signals: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(signals, str):
        return [s.strip() for s in signals.split(",") if s.strip()]
    return [str(s).strip() for s in signals if str(s).strip()]


def _wildcard_to_regex(pattern: str) -> str:
    buf = []
    for ch in pattern:
        if ch == "*":
            buf.append(".*")
        elif ch == "?":
            buf.append(".")
        else:
            buf.append(re.escape(ch))
    return "".join(buf)


def _expand_signals(
    server_url: str,
    signals: List[str],
    max_expand: int,
) -> List[str]:
    expanded: List[str] = []
    for sig in signals:
        if "*" in sig or "?" in sig:
            regex = _wildcard_to_regex(sig)
            resp = requests.get(f"{server_url}/signals", params={"pattern": regex})
            resp.raise_for_status()
            matches = resp.json().get("signals", [])
            if not matches:
                raise RuntimeError(f"No signals matched pattern: {sig}")
            if max_expand > 0 and len(matches) > max_expand:
                raise RuntimeError(f"Pattern '{sig}' matched {len(matches)} signals (> {max_expand})")
            expanded.extend(matches)
        else:
            expanded.append(sig)
    seen = set()
    ordered: List[str] = []
    for sig in expanded:
        if sig in seen:
            continue
        seen.add(sig)
        ordered.append(sig)
    return ordered


class Client:
    def __init__(
        self,
        server_url: Optional[str] = None,
        auto_start: bool = True,
        timeout: float = 10.0,
    ):
        self.server_url = server_url or DEFAULT_SERVER_URL
        self.auto_start = auto_start
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None

    def _server_health(self, timeout: float = 0.5) -> Optional[Dict[str, Any]]:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def _start_server(self) -> None:
        addr = _parse_server_url(self.server_url)
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
        self._proc = subprocess.Popen(cmd, cwd=app_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def ensure_server(self) -> Dict[str, Any]:
        health = self._server_health()
        if health:
            return health
        if not self.auto_start:
            raise RuntimeError("Server is not running")
        self._start_server()
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            health = self._server_health()
            if health:
                return health
            time.sleep(0.2)
        raise RuntimeError("Failed to start server")

    def ensure_loaded(self, file_path: str) -> None:
        abs_path = str(Path(file_path).resolve())
        health = self.ensure_server()
        if health.get("loaded_file") == abs_path:
            return
        resp = requests.post(f"{self.server_url}/load", json={"file_path": abs_path})
        resp.raise_for_status()

    def _post(self, path: str, json_payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(f"{self.server_url}{path}", json=json_payload)
        if resp.status_code == 404 and path in ("/inspect", "/probe"):
            new_port = _alloc_free_port()
            self.server_url = f"http://localhost:{new_port}"
            self.ensure_server()
            resp = requests.post(f"{self.server_url}{path}", json=json_payload)
        if not resp.ok:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise RuntimeError(f"{resp.status_code} {detail}")
        return resp.json()

    def signals(self, pattern: Optional[str] = None) -> List[str]:
        params = {}
        if pattern:
            params["pattern"] = pattern
        resp = requests.get(f"{self.server_url}/signals", params=params)
        resp.raise_for_status()
        return resp.json().get("signals", [])

    def query(self, signal: str, start: int, end: int) -> Dict[str, Any]:
        return self._post("/query", {"signal": signal, "start": start, "end": end})

    def analyze(self, signal: str, start: int, end: int) -> Dict[str, Any]:
        return self._post("/analyze", {"signal": signal, "start": start, "end": end})

    def query_cycles(self, signal: str, clock: str, start_cycle: int, end_cycle: int, edge: str = "rising") -> Dict[str, Any]:
        return self._post(
            "/query_cycles",
            {
                "signal": signal,
                "clock": clock,
                "start_cycle": start_cycle,
                "end_cycle": end_cycle,
                "edge": edge,
            },
        )

    def probe(
        self,
        signals: Union[str, Sequence[str]],
        start: Optional[int] = None,
        end: Optional[int] = None,
        clock: Optional[str] = None,
        start_cycle: Optional[int] = None,
        end_cycle: Optional[int] = None,
        edge: str = "rising",
        include_changes: bool = False,
        max_changes: int = 200,
        max_unique: int = 32,
        max_expand: int = 200,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if file_path:
            self.ensure_loaded(file_path)
        signal_list = _normalize_signals(signals)
        signal_list = _expand_signals(self.server_url, signal_list, max_expand)
        return self._post(
            "/probe",
            {
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
            },
        )

    def inspect(
        self,
        signals: Union[str, Sequence[str]],
        start: Optional[int] = None,
        end: Optional[int] = None,
        clock: Optional[str] = None,
        start_cycle: Optional[int] = None,
        end_cycle: Optional[int] = None,
        edge: str = "rising",
        include_changes: bool = True,
        max_changes: int = 200,
        max_unique: int = 32,
        max_expand: int = 200,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if file_path:
            self.ensure_loaded(file_path)
        signal_list = _normalize_signals(signals)
        signal_list = _expand_signals(self.server_url, signal_list, max_expand)
        return self._post(
            "/inspect",
            {
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
            },
        )

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        self._proc = None

    def __enter__(self) -> "Client":
        self.ensure_server()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class Waveform:
    def __init__(
        self,
        file_path: str,
        server_url: Optional[str] = None,
        auto_start: bool = True,
        timeout: float = 10.0,
    ):
        self.file_path = str(Path(file_path).resolve())
        self.client = Client(server_url=server_url, auto_start=auto_start, timeout=timeout)
        self.client.ensure_loaded(self.file_path)

    def inspect(self, signals: Union[str, Sequence[str]], **kwargs: Any) -> Dict[str, Any]:
        return self.client.inspect(signals, file_path=self.file_path, **kwargs)

    def probe(self, signals: Union[str, Sequence[str]], **kwargs: Any) -> Dict[str, Any]:
        return self.client.probe(signals, file_path=self.file_path, **kwargs)

    def query(self, signal: str, start: int, end: int) -> Dict[str, Any]:
        return self.client.query(signal, start, end)

    def analyze(self, signal: str, start: int, end: int) -> Dict[str, Any]:
        return self.client.analyze(signal, start, end)

    def query_cycles(self, signal: str, clock: str, start_cycle: int, end_cycle: int, edge: str = "rising") -> Dict[str, Any]:
        return self.client.query_cycles(signal, clock, start_cycle, end_cycle, edge=edge)

    def signals(self, pattern: Optional[str] = None) -> List[str]:
        return self.client.signals(pattern)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Waveform":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def inspect(file_path: str, signals: Union[str, Sequence[str]], **kwargs: Any) -> Dict[str, Any]:
    wf = Waveform(file_path)
    return wf.inspect(signals, **kwargs)


def probe(file_path: str, signals: Union[str, Sequence[str]], **kwargs: Any) -> Dict[str, Any]:
    wf = Waveform(file_path)
    return wf.probe(signals, **kwargs)


__all__ = ["Client", "Waveform", "inspect", "probe"]
