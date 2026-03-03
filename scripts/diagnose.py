#!/usr/bin/env python3
"""
diagnose.py — Cross-check DBC signal definitions against what's actually
decoded from your MF4 files. Catches DBC mismatches early.

Usage:
  python scripts/diagnose.py data/raw/can0_20260302_1215.mf4
  python scripts/diagnose.py data/raw/          # runs on all files
  python scripts/diagnose.py data/raw/ --vehicle sprayer_v1
"""
import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from asammdf import MDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def diagnose_file(path: Path, vehicle_name: str = "sprayer_v2") -> dict:
    """
    Decode every CAN frame in the file and collect per-signal statistics.
    Compares against REQUIRED_SIGNALS and reports coverage and anomalies.
    """
    # Import here so vehicle can be overridden before loading
    from sprayer_energy.utils.config import Config, REQUIRED_SIGNALS
    from sprayer_energy.decoders.dbc_decoder import decode_frame, get_all_signal_names

    Config.load(vehicle_name)

    report = {
        "file": path.name,
        "vehicle": vehicle_name,
        "signal_stats": {},
        "required_coverage": {},
        "anomalies": [],
        "dbc_signals_never_seen": [],
    }

    try:
        mdf = MDF(path)
        time   = mdf.get("time", group=0).samples
        can_id = mdf.get("CAN_DataFrame.ID", group=0).samples
        data   = mdf.get("CAN_DataFrame.DataBytes", group=0).samples
        mdf.close()
    except Exception as e:
        report["anomalies"].append(f"Cannot read file: {e}")
        return report

    # Decode every frame and collect signal values
    signal_values: dict[str, list] = defaultdict(list)
    signal_times:  dict[str, list] = defaultdict(list)

    for i, (t, cid, dbytes) in enumerate(zip(time, can_id, data)):
        decoded = decode_frame(int(cid), bytes(dbytes))
        if decoded:
            for k, v in decoded.items():
                signal_values[k].append(float(v) if isinstance(v, (int, float)) else 0)
                signal_times[k].append(float(t))

    # Per-signal statistics
    for sig, vals in signal_values.items():
        arr = np.array(vals, dtype=float)
        report["signal_stats"][sig] = {
            "count":    len(arr),
            "min":      round(float(arr.min()), 4),
            "max":      round(float(arr.max()), 4),
            "mean":     round(float(arr.mean()), 4),
            "std":      round(float(arr.std()), 4),
            "zeros_pct": round(100.0 * (arr == 0).mean(), 1),
            "nan_pct":   round(100.0 * np.isnan(arr).mean(), 1),
        }

        # Anomaly checks
        if (arr == 0).mean() > 0.95:
            report["anomalies"].append(
                f"WARN: '{sig}' is zero >95% of the time — may be unconnected"
            )
        if np.isnan(arr).mean() > 0.5:
            report["anomalies"].append(f"WARN: '{sig}' has >50% NaN values")
        if arr.std() == 0 and len(arr) > 10:
            report["anomalies"].append(
                f"WARN: '{sig}' is perfectly constant ({arr[0]}) — may be stuck"
            )

    # Required signal coverage
    required = REQUIRED_SIGNALS
    seen     = set(signal_values.keys())

    for sig in sorted(required):
        if sig in seen:
            stats = report["signal_stats"][sig]
            report["required_coverage"][sig] = {
                "found": True,
                "count": stats["count"],
                "range": f"{stats['min']} → {stats['max']}",
                "zeros_pct": stats["zeros_pct"],
            }
        else:
            report["required_coverage"][sig] = {"found": False}

    # DBC signals defined but never seen in this file
    all_dbc_sigs = get_all_signal_names()
    report["dbc_signals_never_seen"] = sorted(all_dbc_sigs - seen)

    return report


def print_diagnosis(report: dict):
    w = 70
    print(f"\n{'═'*w}")
    print(f"  DIAGNOSIS: {report['file']}  [vehicle={report['vehicle']}]")
    print(f"{'─'*w}")

    # Required signal table
    cov = report["required_coverage"]
    found   = [(k, v) for k, v in cov.items() if v.get("found")]
    missing = [(k, v) for k, v in cov.items() if not v.get("found")]

    print(f"\n  Required signals found: {len(found)}/{len(cov)}")
    print(f"  {'Signal':<35} {'Count':>7} {'Range':<30} {'Zeros%':>7}")
    print(f"  {'─'*35} {'─'*7} {'─'*30} {'─'*7}")
    for sig, info in sorted(found):
        print(f"  ✓ {sig:<33} {info['count']:>7,} {info['range']:<30} {info['zeros_pct']:>6.1f}%")

    if missing:
        print(f"\n  Missing required signals ({len(missing)}):")
        for sig, _ in missing:
            print(f"  ✗ {sig}")

    if report["anomalies"]:
        print(f"\n  Anomalies ({len(report['anomalies'])}):")
        for a in report["anomalies"]:
            print(f"  ⚠  {a}")

    never_seen = report.get("dbc_signals_never_seen", [])
    if never_seen:
        print(f"\n  DBC signals never seen in this file ({len(never_seen)}):")
        for s in never_seen[:20]:   # cap at 20 to keep readable
            print(f"    · {s}")
        if len(never_seen) > 20:
            print(f"    ... and {len(never_seen)-20} more")

    print(f"\n{'═'*w}\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose DBC vs MF4 signal coverage.")
    parser.add_argument("path",      type=str, help=".mf4 file or directory")
    parser.add_argument("--vehicle", type=str, default="sprayer_v2",
                        help="Vehicle config name (default: sprayer_v2)")
    parser.add_argument("--save",    action="store_true",
                        help="Save summary CSV to outputs/diagnosis.csv")
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_dir():
        files = sorted(target.glob("*.mf4"))
        if not files:
            logger.error(f"No .mf4 files in {target}")
            sys.exit(1)

        rows = []
        for f in files:
            logger.info(f"Diagnosing {f.name}...")
            r = diagnose_file(f, args.vehicle)
            print_diagnosis(r)
            cov     = r["required_coverage"]
            n_found = sum(1 for v in cov.values() if v.get("found"))
            rows.append({
                "file":             r["file"],
                "signals_found":    n_found,
                "signals_required": len(cov),
                "coverage_pct":     round(100 * n_found / max(len(cov), 1), 1),
                "anomaly_count":    len(r["anomalies"]),
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

        if args.save:
            out = Path("outputs/diagnosis.csv")
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            logger.info(f"Saved → {out}")

    elif target.is_file():
        r = diagnose_file(target, args.vehicle)
        print_diagnosis(r)
    else:
        logger.error(f"Not a valid path: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()