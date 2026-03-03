#!/usr/bin/env python3
"""
mf4_inspect.py — Detailed inspection of MF4 files.

Usage:
  python scripts/mf4_inspect.py data/raw/can0_20260302_1215.mf4
  python scripts/mf4_inspect.py data/raw/          # inspect all files
  python scripts/mf4_inspect.py data/raw/ --save   # save report to outputs/
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from asammdf import MDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Signals we expect to find in every healthy file
EXPECTED_SIGNALS = [
    "BatteryVoltage", "Battery_current", "ActualSOCPercentage",
    "Speed", "Motor_Rpm", "Gradient",
    "Spray_Pump_Status", "Ctrl_power", "Overall_Cutback",
    "GNSS_Latitude", "GNSS_Longitude",
]


def inspect_file(path: Path) -> dict:
    """
    Open one MF4 file and return a full quality report dict.
    """
    report = {"file": path.name, "status": "ok", "issues": []}

    try:
        mdf = MDF(path)
    except Exception as e:
        report["status"] = "UNREADABLE"
        report["issues"].append(str(e))
        return report

    try:
        # ── Basic file info ───────────────────────────────────────────────
        channels = mdf.channels_db
        all_channel_names = list(channels.keys())
        report["total_channels"] = len(all_channel_names)

        # Try to read the main CAN group
        try:
            time   = mdf.get("time", group=0).samples
            can_id = mdf.get("CAN_DataFrame.ID", group=0).samples
            data   = mdf.get("CAN_DataFrame.DataBytes", group=0).samples

            duration_sec  = float(time[-1] - time[0]) if len(time) > 1 else 0.0
            sample_count  = len(time)
            avg_hz        = sample_count / duration_sec if duration_sec > 0 else 0.0
            unique_ids    = len(set(can_id))

            report["duration_sec"]    = round(duration_sec, 2)
            report["sample_count"]    = sample_count
            report["avg_sample_hz"]   = round(avg_hz, 1)
            report["unique_can_ids"]  = unique_ids
            report["time_start"]      = round(float(time[0]), 3)
            report["time_end"]        = round(float(time[-1]), 3)

            # Unique CAN IDs seen
            report["can_ids_hex"] = sorted(set(f"0x{i:X}" for i in can_id))

        except Exception as e:
            report["issues"].append(f"Cannot read CAN_DataFrame group: {e}")
            report["duration_sec"] = 0
            report["sample_count"] = 0

        # ── Check expected signals vs available channels ──────────────────
        report["expected_signals_found"]   = []
        report["expected_signals_missing"] = []

        for sig in EXPECTED_SIGNALS:
            if sig in all_channel_names:
                report["expected_signals_found"].append(sig)
            else:
                report["expected_signals_missing"].append(sig)
                report["issues"].append(f"MISSING expected signal: {sig}")

        report["signal_coverage_pct"] = round(
            100 * len(report["expected_signals_found"]) / len(EXPECTED_SIGNALS), 1
        )

        # ── Data quality checks ───────────────────────────────────────────
        if report.get("duration_sec", 0) < 10:
            report["issues"].append("WARNING: File duration < 10 seconds (too short)")

        if report.get("sample_count", 0) < 100:
            report["issues"].append("WARNING: Less than 100 samples (very sparse)")

        if report.get("avg_sample_hz", 0) < 5:
            report["issues"].append("WARNING: Sample rate < 5 Hz (low frequency)")

        report["healthy"] = len(report["issues"]) == 0

    finally:
        mdf.close()

    return report


def print_report(report: dict):
    w = 60
    healthy = report.get("healthy", False)
    status  = "✓ HEALTHY" if healthy else "✗ ISSUES FOUND"

    print(f"\n{'═'*w}")
    print(f"  {report['file']}")
    print(f"  Status: {status}")
    print(f"{'─'*w}")

    if "duration_sec" in report:
        print(f"  Duration        : {report['duration_sec']:.1f} s  "
              f"({report['duration_sec']/60:.1f} min)")
        print(f"  Samples         : {report['sample_count']:,}")
        print(f"  Sample rate     : {report['avg_sample_hz']:.1f} Hz")
        print(f"  Unique CAN IDs  : {report['unique_can_ids']}")
        print(f"  Signal coverage : {report['signal_coverage_pct']}% "
              f"({len(report['expected_signals_found'])}/{len(EXPECTED_SIGNALS)} expected)")

    if report["expected_signals_missing"]:
        print(f"\n  Missing signals:")
        for s in report["expected_signals_missing"]:
            print(f"    ✗ {s}")

    if report["expected_signals_found"]:
        print(f"\n  Found signals:")
        for s in report["expected_signals_found"]:
            print(f"    ✓ {s}")

    if report["issues"]:
        print(f"\n  Issues ({len(report['issues'])}):")
        for issue in report["issues"]:
            print(f"    ⚠  {issue}")

    print(f"{'═'*w}")


def batch_inspect(directory: Path, save: bool = False) -> pd.DataFrame:
    files = sorted(directory.glob("*.mf4"))
    if not files:
        logger.error(f"No .mf4 files found in {directory}")
        sys.exit(1)

    logger.info(f"Inspecting {len(files)} MF4 files in {directory}...")
    reports = []

    for f in files:
        r = inspect_file(f)
        print_report(r)
        reports.append(r)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  BATCH SUMMARY — {len(files)} files")
    print(f"{'─'*70}")
    print(f"  {'File':<35} {'Duration':>8} {'Samples':>8} {'Coverage':>10} {'Status':>10}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")

    for r in reports:
        status = "✓" if r.get("healthy") else f"✗ {len(r['issues'])} issues"
        print(f"  {r['file']:<35} "
              f"{r.get('duration_sec', 0):>7.1f}s "
              f"{r.get('sample_count', 0):>8,} "
              f"{r.get('signal_coverage_pct', 0):>9.1f}% "
              f"{status:>10}")

    healthy_count = sum(1 for r in reports if r.get("healthy"))
    total_dur     = sum(r.get("duration_sec", 0) for r in reports)
    total_samples = sum(r.get("sample_count", 0) for r in reports)
    print(f"{'─'*70}")
    print(f"  Healthy files   : {healthy_count}/{len(files)}")
    print(f"  Total duration  : {total_dur:.1f}s  ({total_dur/60:.1f} min)")
    print(f"  Total samples   : {total_samples:,}")
    print(f"{'═'*70}\n")

    df = pd.DataFrame(reports)

    if save:
        out = Path("outputs") / "mf4_inspection_report.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Flatten list columns for CSV
        for col in ["expected_signals_found", "expected_signals_missing", "issues", "can_ids_hex"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: "|".join(x) if isinstance(x, list) else x)
        df.to_csv(out, index=False)
        logger.info(f"Report saved → {out}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Inspect MF4 files for signal quality.")
    parser.add_argument("path",   type=str, help="Path to .mf4 file or directory")
    parser.add_argument("--save", action="store_true", help="Save batch report to outputs/")
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_dir():
        batch_inspect(target, save=args.save)
    elif target.is_file() and target.suffix.lower() == ".mf4":
        report = inspect_file(target)
        print_report(report)
    else:
        logger.error(f"Path must be a .mf4 file or directory containing .mf4 files: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()