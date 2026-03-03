"""
Central configuration for sprayer_energy.
All constants, paths, vehicle profiles, and signal names live here.
Switch vehicles with: Config.load("sprayer_v1")
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# ── Project root (2 levels up from this file) ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Default paths ─────────────────────────────────────────────────────────────
PATHS = {
    "raw_data":       PROJECT_ROOT / "data" / "raw",
    "processed_data": PROJECT_ROOT / "data" / "processed",
    "outputs":        PROJECT_ROOT / "outputs",
    "plots":          PROJECT_ROOT / "outputs" / "plots",
    "dbc_dir":        PROJECT_ROOT / "docs" / "dbc",
    "vehicle_dir":    PROJECT_ROOT / "docs" / "vehicles",
    "model":          PROJECT_ROOT / "outputs" / "energy_model.pkl",
    "dataset":        PROJECT_ROOT / "data" / "processed" / "training_dataset.csv",
}

# ── Fault thresholds ──────────────────────────────────────────────────────────
FAULT_THRESHOLDS = {
    "overall_cutback_pct":   50.0,   # flag if cutback > 50%
    "bms_error_code":         0,     # flag if BMS_Error_Code != 0
    "max_ctrl_temp_c":       80.0,   # flag if controller temp > 80°C
    "max_motor_temp_c":      90.0,   # flag if motor temp > 90°C
    "max_hyd_temp_c":        85.0,
    "min_battery_voltage_v": 40.0,   # flag if voltage drops below this
    "min_soc_pct":           20.0,   # flag low SOC windows
}

# ── Windowing ─────────────────────────────────────────────────────────────────
WINDOW_SIZE_SEC  = 30
MIN_WINDOW_ROWS  = 10

# ── Required CAN signals for energy model ─────────────────────────────────────
REQUIRED_SIGNALS = {
    # Electrical / Battery
    "BatteryVoltage", "Battery_current", "ActualSOCPercentage",
    "RSOC", "RSOC_2",
    # Powertrain
    "Speed", "Motor_Rpm", "Ctrl_power",
    "Ctrl_Temperature", "Ctrl_Bat_Voltage", "Ctrl_Bat_Current",
    # Motor
    "Tr_Mtr_Temp", "Mtr_RMS_currents", "Motor_ctrl_efficiency", "Overall_Cutback",
    # Terrain
    "Gradient",
    # Sprayer
    "Spray_Pump_Status", "Spray_Pressure", "Spray_Flowrate",
    # Drive / Mode
    "Travel_Mode", "Field_Mode", "Gears", "Vehicle_Acceleration",
    # IMU
    "Accel_X", "Accel_Y",
    # Spray battery (CAN1)
    "BatteryVoltage_spray", "Battery_current_spray",
    # Hydraulics
    "RPM", "hyd_Motor_temperature",
    # GNSS
    "GNSS_Latitude", "GNSS_Longitude",
    # Faults
    "BMS_Error_Code", "Tr_ctrl_fault",
}

# ── Vehicle profiles ──────────────────────────────────────────────────────────
@dataclass
class VehicleConfig:
    name:                  str
    description:           str
    battery_capacity_wh:   float        # Total usable energy
    battery_voltage_v:     float        # Nominal voltage
    battery_capacity_ah:   float        # Total Ah
    num_battery_packs:     int
    spray_width_m:         float
    can0_dbc:              str = "can0.dbc"
    can1_dbc:              str = "can1.dbc"
    extra_signals:         list = field(default_factory=list)
    notes:                 str = ""

    def to_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Vehicle config saved → {path}")

    @classmethod
    def from_json(cls, path: Path) -> "VehicleConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ── Built-in vehicle profiles ─────────────────────────────────────────────────
VEHICLES: dict[str, VehicleConfig] = {

    "sprayer_v1": VehicleConfig(
        name                = "sprayer_v1",
        description         = "Original test vehicle — single battery pack",
        battery_capacity_wh = 3000.0,
        battery_voltage_v   = 52.1,
        battery_capacity_ah = 206.0,
        num_battery_packs   = 1,
        spray_width_m       = 1.5,
        notes               = "First prototype. Limited field data.",
    ),

    "sprayer_v2": VehicleConfig(
        name                = "sprayer_v2",
        description         = "Production vehicle — dual 10kWh packs, 10m boom",
        battery_capacity_wh = 20000.0,    # 2 × 10 kWh
        battery_voltage_v   = 52.1,
        battery_capacity_ah = 412.0,      # 2 × 206 Ah
        num_battery_packs   = 2,
        spray_width_m       = 10.0,
        can0_dbc            = "can0.dbc",
        can1_dbc            = "can1.dbc",
        notes               = "Dual pack parallel. CAN1 = spray battery.",
    ),
}


# ── Active config (singleton) ─────────────────────────────────────────────────
class Config:
    _active: VehicleConfig = VEHICLES["sprayer_v2"]   # default

    @classmethod
    def load(cls, name: str) -> VehicleConfig:
        """Switch active vehicle by name. Falls back to JSON in docs/vehicles/."""
        if name in VEHICLES:
            cls._active = VEHICLES[name]
            logger.info(f"Loaded built-in vehicle config: {name}")
        else:
            json_path = PATHS["vehicle_dir"] / f"{name}.json"
            if json_path.exists():
                cls._active = VehicleConfig.from_json(json_path)
                logger.info(f"Loaded vehicle config from: {json_path}")
            else:
                raise ValueError(
                    f"Unknown vehicle '{name}'. "
                    f"Available built-ins: {list(VEHICLES.keys())}. "
                    f"Or create {json_path}."
                )
        return cls._active

    @classmethod
    def get(cls) -> VehicleConfig:
        return cls._active

    @classmethod
    def list_vehicles(cls) -> list[str]:
        names = list(VEHICLES.keys())
        vdir  = PATHS["vehicle_dir"]
        if vdir.exists():
            names += [p.stem for p in vdir.glob("*.json")]
        return names