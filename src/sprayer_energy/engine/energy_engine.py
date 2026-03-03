"""
Advanced EnergyEngine — reads vehicle config from central Config.
Supports mission planning, range estimation, and sensitivity analysis.
"""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sprayer_energy.utils.config import Config, PATHS

ACRE_TO_M2 = 4046.86

_FEATURE_DEFAULTS = {
    "Avg_Speed": 4.0, "Max_Speed": 6.0, "Speed_Std": 1.0,
    "RMS_Accel_Derived": 0.1, "Avg_Accel_g": 0.02,
    "Avg_Gradient": 0.0, "Max_Gradient": 0.0, "Uphill_Frac": 0.0,
    "Avg_Power_W": 1500.0, "Peak_Power_W": 2500.0, "Spray_Energy_Frac": 0.3,
    "Start_SOC": 90.0, "SOC_Drop": 1.0, "Start_RSOC": 90.0, "Start_RSOC2": 90.0,
    "Avg_RPM": 1200.0, "Max_RPM": 2000.0, "Avg_Hyd_RPM": 800.0,
    "Avg_Efficiency": 85.0, "Avg_Cutback": 2.0,
    "Avg_Ctrl_Temp": 40.0, "Avg_Motor_Temp": 50.0, "Avg_Hyd_Temp": 45.0,
    "Dominant_Gear": 1, "Low_Gear_Frac": 0.7, "High_Gear_Frac": 0.3,
    "Travel_Frac": 0.1, "Field_Frac": 0.9,
    "Spray_Ratio": 0.85, "Avg_Spray_Pressure": 3.0, "Avg_Spray_Flow": 15.0,
    "Vibration": 0.5,
}


@dataclass
class MissionResult:
    total_energy_wh:   float
    required_soc_pct:  float
    distance_m:        float
    area_acres:        float
    energy_per_meter:  float
    feasible:          bool
    margin_soc_pct:    float
    start_soc_pct:     float
    vehicle:           str


class EnergyEngine:

    def __init__(
        self,
        model_path: str | None = None,
        battery_capacity_wh: float | None = None,
        spray_width_m: float | None = None,
        vehicle: str | None = None,
    ):
        # Load vehicle config
        if vehicle:
            Config.load(vehicle)
        cfg = Config.get()

        self.battery_capacity = battery_capacity_wh or cfg.battery_capacity_wh
        self.spray_width      = spray_width_m      or cfg.spray_width_m
        self.vehicle_name     = cfg.name

        pkl = model_path or str(PATHS["model"])
        self.pipeline = joblib.load(pkl)
        self._feature_names = list(
            self.pipeline.named_steps["imputer"].feature_names_in_
        )

    def _build_input(self, overrides: dict) -> pd.DataFrame:
        row = {**_FEATURE_DEFAULTS, **overrides}
        row = {k: row.get(k, 0.0) for k in self._feature_names}
        return pd.DataFrame([row])

    def predict_energy_per_meter(self, **kwargs) -> float:
        return float(self.pipeline.predict(self._build_input(kwargs))[0])

    def plan_mission(
        self,
        acres: float,
        start_soc: float,
        avg_speed: float = 4.0,
        avg_gradient: float = 0.0,
        spray_ratio: float = 0.85,
        **extra,
    ) -> MissionResult:
        distance_m = (acres * ACRE_TO_M2) / self.spray_width
        epm = self.predict_energy_per_meter(
            Avg_Speed=avg_speed, Avg_Gradient=avg_gradient,
            Spray_Ratio=spray_ratio, Start_SOC=start_soc, **extra,
        )
        total_energy   = epm * distance_m
        required_soc   = (total_energy / self.battery_capacity) * 100
        available_wh   = (start_soc / 100.0) * self.battery_capacity

        return MissionResult(
            total_energy_wh  = round(total_energy, 2),
            required_soc_pct = round(required_soc, 2),
            distance_m       = round(distance_m, 1),
            area_acres       = round(acres, 3),
            energy_per_meter = round(epm, 5),
            feasible         = total_energy <= available_wh,
            margin_soc_pct   = round(start_soc - required_soc, 2),
            start_soc_pct    = start_soc,
            vehicle          = self.vehicle_name,
        )

    def estimate_range(
        self,
        current_soc: float,
        avg_speed: float = 4.0,
        avg_gradient: float = 0.0,
        spray_ratio: float = 0.85,
        **extra,
    ) -> dict:
        available_wh  = (current_soc / 100.0) * self.battery_capacity
        epm = self.predict_energy_per_meter(
            Avg_Speed=avg_speed, Avg_Gradient=avg_gradient,
            Spray_Ratio=spray_ratio, Start_SOC=current_soc, **extra,
        )
        max_dist_m = available_wh / max(epm, 1e-9)
        area_m2    = max_dist_m * self.spray_width
        max_acres  = area_m2 / ACRE_TO_M2

        return {
            "Max_Acres":           round(max_acres, 3),
            "Max_Distance_m":      round(max_dist_m, 1),
            "Energy_per_meter":    round(epm, 5),
            "Available_Energy_Wh": round(available_wh, 1),
        }

    def sensitivity_analysis(
        self,
        base_acres: float,
        base_soc: float = 80.0,
        param_ranges: dict | None = None,
    ) -> pd.DataFrame:
        if param_ranges is None:
            param_ranges = {
                "avg_speed":    [2, 3, 4, 5, 6],
                "avg_gradient": [-5, -2, 0, 2, 5],
                "spray_ratio":  [0.5, 0.7, 0.85, 1.0],
                "start_soc":    [50, 60, 70, 80, 90],
            }
        rows = []
        for param, values in param_ranges.items():
            for v in values:
                kwargs = {"avg_speed": 4.0, "avg_gradient": 0.0,
                          "spray_ratio": 0.85, "start_soc": base_soc}
                kwargs[param] = v
                r = self.plan_mission(acres=base_acres, **kwargs)
                rows.append({
                    "Parameter":       param,
                    "Value":           v,
                    "Required_SOC_%":  r.required_soc_pct,
                    "Energy_Wh":       r.total_energy_wh,
                    "Feasible":        r.feasible,
                })
        return pd.DataFrame(rows)