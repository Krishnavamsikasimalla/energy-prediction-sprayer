"""
test_decode.py — Unit tests for the CAN decoder pipeline.
Run with: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_raw_df(n=50, seed=42):
    """Create a synthetic raw CAN DataFrame for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Timestamp": np.linspace(0, 30, n),
        "CAN_ID":    rng.integers(256, 2600, size=n),
        "DataBytes": [bytes(rng.integers(0, 255, 8).tolist()) for _ in range(n)],
    })


def make_decoded_df(n=100, seed=0):
    """Create a synthetic decoded signal DataFrame."""
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 30, n)
    return pd.DataFrame({
        "Timestamp":           t,
        "BatteryVoltage":      rng.uniform(48, 54, n),
        "Battery_current":     rng.uniform(5, 30, n),
        "ActualSOCPercentage": np.linspace(90, 88, n),
        "Speed":               rng.uniform(0, 6, n),
        "Motor_Rpm":           rng.uniform(500, 2000, n),
        "Gradient":            rng.uniform(-3, 3, n),
        "Spray_Pump_Status":   rng.integers(0, 2, n).astype(float),
        "Ctrl_power":          rng.uniform(200, 800, n),
        "Overall_Cutback":     rng.uniform(0, 10, n),
        "GNSS_Latitude":       rng.uniform(17.3, 17.4, n),
        "GNSS_Longitude":      rng.uniform(78.4, 78.5, n),
        "BMS_Error_Code":      np.zeros(n),
        "Tr_ctrl_fault":       np.zeros(n),
        "RSOC":                np.full(n, 90.0),
        "RSOC_2":              np.full(n, 90.0),
        "Travel_Mode":         np.zeros(n),
        "Field_Mode":          np.ones(n),
        "Gears":               np.ones(n),
        "Vehicle_Acceleration":rng.uniform(-0.1, 0.1, n),
        "Accel_X":             rng.uniform(-0.5, 0.5, n),
        "Accel_Y":             rng.uniform(-0.5, 0.5, n),
        "BatteryVoltage_spray":rng.uniform(48, 54, n),
        "Battery_current_spray":rng.uniform(1, 10, n),
        "RPM":                 rng.uniform(400, 1200, n),
        "hyd_Motor_temperature":rng.uniform(30, 60, n),
        "Tr_Mtr_Temp":         rng.uniform(35, 70, n),
        "Ctrl_Temperature":    rng.uniform(30, 65, n),
        "Motor_ctrl_efficiency":rng.uniform(75, 95, n),
        "Spray_Pressure":      rng.uniform(2, 5, n),
        "Spray_Flowrate":      rng.uniform(10, 20, n),
    })


# ── Config tests ──────────────────────────────────────────────────────────────

class TestConfig:

    def test_default_vehicle_is_sprayer_v2(self):
        from sprayer_energy.utils.config import Config
        cfg = Config.get()
        assert cfg.name == "sprayer_v2"

    def test_load_sprayer_v1(self):
        from sprayer_energy.utils.config import Config
        cfg = Config.load("sprayer_v1")
        assert cfg.battery_capacity_wh == 3000.0
        assert cfg.spray_width_m == 1.5

    def test_load_sprayer_v2(self):
        from sprayer_energy.utils.config import Config
        cfg = Config.load("sprayer_v2")
        assert cfg.battery_capacity_wh == 20000.0
        assert cfg.spray_width_m == 10.0
        assert cfg.num_battery_packs == 2

    def test_unknown_vehicle_raises(self):
        from sprayer_energy.utils.config import Config
        with pytest.raises(ValueError, match="Unknown vehicle"):
            Config.load("nonexistent_vehicle_xyz")

    def test_list_vehicles_returns_both(self):
        from sprayer_energy.utils.config import Config
        vehicles = Config.list_vehicles()
        assert "sprayer_v1" in vehicles
        assert "sprayer_v2" in vehicles

    def test_required_signals_is_nonempty_set(self):
        from sprayer_energy.utils.config import REQUIRED_SIGNALS
        assert isinstance(REQUIRED_SIGNALS, set)
        assert len(REQUIRED_SIGNALS) > 10
        assert "BatteryVoltage" in REQUIRED_SIGNALS
        assert "Speed" in REQUIRED_SIGNALS


# ── Physics tests ─────────────────────────────────────────────────────────────

class TestPhysics:

    def setup_method(self):
        from sprayer_energy.utils.config import Config
        Config.load("sprayer_v2")

    def test_returns_dict_on_valid_input(self):
        from sprayer_energy.data_pipeline.physics import compute_mission_summary
        df = make_decoded_df(100)
        result = compute_mission_summary(df)
        assert result is not None
        assert isinstance(result, dict)

    def test_returns_none_on_empty(self):
        from sprayer_energy.data_pipeline.physics import compute_mission_summary
        result = compute_mission_summary(pd.DataFrame())
        assert result is None

    def test_energy_is_positive(self):
        from sprayer_energy.data_pipeline.physics import compute_mission_summary
        df = make_decoded_df(100)
        result = compute_mission_summary(df)
        assert result["Energy_Wh"] > 0

    def test_distance_is_positive(self):
        from sprayer_energy.data_pipeline.physics import compute_mission_summary
        df = make_decoded_df(100)
        result = compute_mission_summary(df)
        assert result["Distance_m"] > 0

    def test_soc_bounds(self):
        from sprayer_energy.data_pipeline.physics import compute_mission_summary
        df = make_decoded_df(100)
        result = compute_mission_summary(df)
        assert 0 <= result["Start_SOC"] <= 100
        assert 0 <= result["End_SOC"] <= 100

    def test_spray_ratio_between_0_and_1(self):
        from sprayer_energy.data_pipeline.physics import compute_mission_summary
        df = make_decoded_df(100)
        result = compute_mission_summary(df)
        assert 0 <= result["Spray_Ratio"] <= 1.0

    def test_gnss_captured(self):
        from sprayer_energy.data_pipeline.physics import compute_mission_summary
        df = make_decoded_df(100)
        result = compute_mission_summary(df)
        assert "Start_Lat" in result
        assert "Start_Lon" in result
        assert not np.isnan(result["Start_Lat"])

    def test_all_required_keys_present(self):
        from sprayer_energy.data_pipeline.physics import compute_mission_summary
        df = make_decoded_df(100)
        result = compute_mission_summary(df)
        for key in ["Energy_Wh", "Distance_m", "Avg_Speed", "Start_SOC",
                    "Spray_Ratio", "Avg_RPM", "Window_Healthy", "Total_Faults"]:
            assert key in result, f"Missing key: {key}"

    def test_missing_signals_handled_gracefully(self):
        """Physics should not crash if optional signals are missing."""
        from sprayer_energy.data_pipeline.physics import compute_mission_summary
        df = make_decoded_df(100)
        # Drop several signals
        df = df.drop(columns=["Motor_ctrl_efficiency", "Overall_Cutback", "Tr_Mtr_Temp"])
        result = compute_mission_summary(df)
        assert result is not None


# ── Fault detection tests ─────────────────────────────────────────────────────

class TestFaultDetection:

    def setup_method(self):
        from sprayer_energy.utils.config import Config
        Config.load("sprayer_v2")

    def test_healthy_window_has_no_faults(self):
        from sprayer_energy.data_pipeline.physics import detect_faults
        df = make_decoded_df(100)
        faults = detect_faults(df)
        assert faults["Window_Healthy"] == 1
        assert faults["Total_Faults"] == 0

    def test_bms_error_detected(self):
        from sprayer_energy.data_pipeline.physics import detect_faults
        df = make_decoded_df(100)
        df["BMS_Error_Code"] = 5   # non-zero = fault
        faults = detect_faults(df)
        assert faults["Fault_BMS"] == 1
        assert faults["Window_Healthy"] == 0

    def test_cutback_fault_detected(self):
        from sprayer_energy.data_pipeline.physics import detect_faults
        df = make_decoded_df(100)
        df["Overall_Cutback"] = 75.0   # > threshold of 50%
        faults = detect_faults(df)
        assert faults["Fault_Cutback"] == 1

    def test_low_voltage_detected(self):
        from sprayer_energy.data_pipeline.physics import detect_faults
        df = make_decoded_df(100)
        df["BatteryVoltage"] = 35.0    # below min_battery_voltage_v = 40
        faults = detect_faults(df)
        assert faults["Fault_Low_Voltage"] == 1

    def test_multiple_faults_counted(self):
        from sprayer_energy.data_pipeline.physics import detect_faults
        df = make_decoded_df(100)
        df["BMS_Error_Code"]  = 1
        df["Overall_Cutback"] = 80.0
        faults = detect_faults(df)
        assert faults["Total_Faults"] >= 2


# ── Windowing tests ───────────────────────────────────────────────────────────

class TestWindowing:

    def setup_method(self):
        from sprayer_energy.utils.config import Config
        Config.load("sprayer_v2")

    def test_windows_created_from_long_df(self):
        from sprayer_energy.data_pipeline.windowing import split_into_windows
        df = make_decoded_df(n=500)
        # Extend time to 120 seconds to get multiple windows
        df["Timestamp"] = np.linspace(0, 120, 500)
        result = split_into_windows(df, window_size_sec=30)
        assert result is not None
        assert len(result) >= 3

    def test_returns_none_on_empty(self):
        from sprayer_energy.data_pipeline.windowing import split_into_windows
        result = split_into_windows(pd.DataFrame())
        assert result is None

    def test_returns_none_on_tiny_df(self):
        from sprayer_energy.data_pipeline.windowing import split_into_windows
        df = make_decoded_df(n=5)
        result = split_into_windows(df, window_size_sec=30)
        # 5 rows < MIN_WINDOW_ROWS=10, should produce nothing
        assert result is None or len(result) == 0


# ── Decode dataframe tests (mocked DBC) ──────────────────────────────────────

class TestDecodeDataframe:

    def test_empty_input_returns_empty(self):
        from sprayer_energy.decoders.dbc_decoder import decode_dataframe
        result = decode_dataframe(pd.DataFrame())
        assert result.empty

    def test_output_has_timestamp(self):
        """With a mocked decode_frame that returns known signals."""
        from sprayer_energy.utils.config import Config
        Config.load("sprayer_v2")

        raw = make_raw_df(20)
        mock_decoded = {
            "BatteryVoltage": 52.0,
            "Battery_current": 10.0,
            "Speed": 4.5,
        }

        with patch("sprayer_energy.decoders.dbc_decoder._load_databases"), \
             patch("sprayer_energy.decoders.dbc_decoder.decode_frame",
                   return_value=mock_decoded):
            from sprayer_energy.decoders import dbc_decoder
            result = dbc_decoder.decode_dataframe(raw)

        assert not result.empty
        assert "Timestamp" in result.columns
        assert "BatteryVoltage" in result.columns