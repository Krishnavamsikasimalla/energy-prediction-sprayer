import logging
import pandas as pd

# Correct absolute import referencing the installed package
from sprayer_energy.data_pipeline.physics import compute_mission_summary

logger = logging.getLogger(__name__)

def split_into_windows(df: pd.DataFrame, window_size_sec: int = 30) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    df = df.sort_values("Timestamp")
    start_time = df["Timestamp"].min()
    end_time = df["Timestamp"].max()

    windows = []
    current = start_time

    while current < end_time:
        # Extract the 30-second chunk
        window_df = df[
            (df["Timestamp"] >= current) &
            (df["Timestamp"] < current + window_size_sec)
        ]

        # Require at least 10 samples to consider it a valid window
        if len(window_df) > 10:
            summary = compute_mission_summary(window_df)
            
            # If physics calculation succeeded, append it
            if summary:
                summary["Window_Start"] = current
                windows.append(summary)
        else:
            logger.debug(f"Skipping tiny window at {current}s (only {len(window_df)} rows).")

        current += window_size_sec

    if not windows:
        logger.warning("No valid windows extracted from this file.")
        return None

    return pd.DataFrame(windows)