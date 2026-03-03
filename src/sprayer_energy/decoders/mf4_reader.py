import logging
from pathlib import Path
import pandas as pd
import numpy as np
from asammdf import MDF

logger = logging.getLogger(__name__)


def extract_can_frames(path: str | Path, group_index: int = 0) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")

    logger.debug(f"Loading MDF: {path_obj.name}")
    try:
        mdf = MDF(path_obj)
    except Exception as e:
        logger.error(f"Failed to read MDF {path_obj.name}: {e}")
        raise

    try:
        time       = mdf.get("time",                    group=group_index).samples
        can_id     = mdf.get("CAN_DataFrame.ID",        group=group_index).samples
        data       = mdf.get("CAN_DataFrame.DataBytes", group=group_index).samples
        data_bytes = [bytes(row) for row in data]

        try:
            bus_ch = mdf.get("CAN_DataFrame.BusChannel", group=group_index).samples
        except Exception:
            logger.debug("No BusChannel signal — defaulting all frames to channel 1")
            bus_ch = np.ones(len(time), dtype=int)

        df = pd.DataFrame({
            "Timestamp":  time,
            "CAN_ID":     can_id.astype(int),
            "DataBytes":  data_bytes,
            "BusChannel": bus_ch.astype(int),
        })

        logger.debug(f"{path_obj.name}: {len(df)} frames, channels: {sorted(df['BusChannel'].unique())}")
        return df

    except Exception as e:
        logger.error(f"Failed to extract from {path_obj.name}: {e}")
        raise
    finally:
        mdf.close()
