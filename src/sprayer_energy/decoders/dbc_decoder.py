"""
DBC decoder — uses REQUIRED_SIGNALS from central config.
Supports vehicle-specific DBC files.
"""
import logging
from pathlib import Path
import pandas as pd
import cantools

from sprayer_energy.utils.config import Config, PATHS, REQUIRED_SIGNALS

logger = logging.getLogger(__name__)

_db_cache: dict[str, cantools.database.Database] = {}


def _load_databases():
    """Load and cache DBC files for the active vehicle config."""
    vehicle = Config.get()
    dbc_dir = PATHS["dbc_dir"]

    can0_key = vehicle.can0_dbc
    can1_key = vehicle.can1_dbc

    if can0_key not in _db_cache:
        path = dbc_dir / can0_key
        if not path.exists():
            raise FileNotFoundError(f"Missing DBC: {path.absolute()}")
        _db_cache[can0_key] = cantools.database.load_file(path)
        logger.debug(f"Loaded DBC: {path}")

    if can1_key not in _db_cache:
        path = dbc_dir / can1_key
        if not path.exists():
            raise FileNotFoundError(f"Missing DBC: {path.absolute()}")
        _db_cache[can1_key] = cantools.database.load_file(path)
        logger.debug(f"Loaded DBC: {path}")

    return _db_cache[can0_key], _db_cache[can1_key]


def decode_frame(can_id: int, data_bytes: bytes):
    """Decode a CAN frame — tries can0 then can1."""
    db0, db1 = _load_databases()
    for db in (db0, db1):
        try:
            return db.decode_message(can_id, data_bytes)
        except KeyError:
            continue
        except cantools.database.errors.DecodeError as e:
            logger.debug(f"DecodeError ID {can_id:#x}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error ID {can_id:#x}: {e}")
            return None
    return None


def decode_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode raw CAN frames into a signal DataFrame.
    Only keeps REQUIRED_SIGNALS + any vehicle extra_signals.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    vehicle  = Config.get()
    want     = REQUIRED_SIGNALS | set(vehicle.extra_signals)
    records  = []
    _logged  = False

    for _, row in raw_df.iterrows():
        can_id     = int(row["CAN_ID"])
        data_bytes = bytes(row["DataBytes"])
        decoded    = decode_frame(can_id, data_bytes)

        if decoded:
            if not _logged:
                logger.debug(f"Sample decoded keys: {list(decoded.keys())[:8]}")
                _logged = True

            filtered = {k: decoded[k] for k in want if k in decoded}
            if filtered:
                filtered["Timestamp"] = row["Timestamp"]
                records.append(filtered)

    if not records:
        logger.debug("No required signals found in DataFrame.")
        return pd.DataFrame()

    return pd.DataFrame(records)


def get_all_signal_names() -> set[str]:
    """Return all signal names defined in both DBC files."""
    db0, db1 = _load_databases()
    names = set()
    for db in (db0, db1):
        for msg in db.messages:
            for sig in msg.signals:
                names.add(sig.name)
    return names