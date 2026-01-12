#!/usr/bin/env python3
import argparse
import sys
from typing import List, Tuple, Optional

try:
    from vcdvcd import VCDVCD
except Exception as exc:
    print("vcdvcd is required for this verification. Install with: pip install vcdvcd")
    raise SystemExit(1) from exc

from server import WaveformDatabase


def normalize_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "".join(str(v) for v in value)
    return str(value)


def extract_changes(tv: List[Tuple[int, str]], start: int, end: int) -> Tuple[Optional[str], List[Tuple[int, str]]]:
    initial = None
    changes: List[Tuple[int, str]] = []
    for t, v in tv:
        v = normalize_value(v)
        if t < start:
            initial = v
            continue
        if t > end:
            break
        changes.append((t, v))
    return initial, changes


def compare_changes(db_changes: List[Tuple[int, str]], vcd_changes: List[Tuple[int, str]]) -> Optional[str]:
    if len(db_changes) != len(vcd_changes):
        return f"length mismatch: wavetoearth={len(db_changes)} vcdvcd={len(vcd_changes)}"
    for idx, (db_item, vcd_item) in enumerate(zip(db_changes, vcd_changes)):
        if db_item != vcd_item:
            return f"diff at {idx}: wavetoearth={db_item} vcdvcd={vcd_item}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify wavetoearth values against vcdvcd")
    parser.add_argument("--file", required=True, help="VCD file path")
    parser.add_argument("--signals", required=True, help="Comma-separated signal names (short names OK)")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    raw_signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    if not raw_signals:
        print("No signals provided")
        return 1

    db = WaveformDatabase()
    db.load(args.file)

    resolved = []
    for sig in raw_signals:
        res = db._resolve_signal(sig)
        resolved.append(res["signal_name"])

    vcd = VCDVCD(args.file, only_sigs=False, signals=resolved, store_tvs=True)

    failed = False
    for sig in resolved:
        if sig not in vcd.references_to_ids:
            print(f"[FAIL] vcdvcd missing signal: {sig}")
            failed = True
            continue
        sig_id = vcd.references_to_ids[sig]
        tv = vcd.data[sig_id].tv

        vcd_initial, vcd_changes = extract_changes(tv, args.start, args.end)
        wav = db.query(sig, args.start, args.end)
        wav_initial = normalize_value(wav["initial_value"])
        wav_changes = [(t, normalize_value(v)) for t, v in wav["changes"]]

        if vcd_initial is None:
            vcd_initial = ""
        if wav_initial is None:
            wav_initial = ""

        if vcd_initial != wav_initial:
            print(f"[FAIL] initial mismatch for {sig}: wavetoearth={wav_initial} vcdvcd={vcd_initial}")
            failed = True

        diff = compare_changes(wav_changes, vcd_changes)
        if diff:
            print(f"[FAIL] changes mismatch for {sig}: {diff}")
            failed = True
        else:
            print(f"[OK] {sig} ({len(wav_changes)} changes)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
