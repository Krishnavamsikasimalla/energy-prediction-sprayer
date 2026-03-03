"""
Advanced physics computation.
Extracts 40+ features per window including fault flags and GNSS.
"""
import logging
import pandas as pd
import numpy as np
from sprayer_energy.utils.config import Config, FAULT_THRESHOLDS

logger = logging.getLogger(__name__)


def detect_faults(df: pd.DataFrame) -> dict:
    """
    Scan a window for fault conditions.
    Returns a dict of boolean flags and fault counts.
    """
    def safe_get(col, default=0.0):
        return df[col].fillna(default) if col in df.columns else pd.Series(default, index=df.index)

    faults = {}

    # BMS error
    bms_err = safe_get("BMS_Error_Code", 0)
    faults["Fault_BMS"]          = int((bms_err != 0).any())
    faults["Fault_BMS_Count"]    = int((bms_err != 0).sum())

    # Traction controller fault
    tr_fault = safe_get("Tr_ctrl_fault", 0)
    faults["Fault_Traction"]     = int((tr_fault != 0).any())

    # Cutback > threshold
    cutback = safe_get("Overall_Cutback", 0)
    faults["Fault_Cutback"]      = int((cutback > FAULT_THRESHOLDS["overall_cutback_pct"]).any())
    faults["Max_Cutback"]        = float(cutback.max())

    # Thermal faults
    ctrl_temp  = safe_get("Ctrl_Temperature", 0)
    motor_temp = safe_get("Tr_Mtr_Temp", 0)
    hyd_temp   = safe_get("hyd_Motor_temperature", 0)
    faults["Fault_Ctrl_Overheat"]  = int((ctrl_temp  > FAULT_THRESHOLDS["max_ctrl_temp_c"]).any())
    faults["Fault_Motor_Overheat"] = int((motor_temp > FAULT_THRESHOLDS["max_motor_temp_c"]).any())
    faults["Fault_Hyd_Overheat"]   = int((hyd_temp   > FAULT_THRESHOLDS["max_hyd_temp_c"]).any())

    # Low voltage
    voltage = safe_get("BatteryVoltage", 999)
    faults["Fault_Low_Voltage"]  = int((voltage < FAULT_THRESHOLDS["min_battery_voltage_v"]).any())
    faults["Min_Voltage"]        = float(voltage.min())

    # Low SOC
    soc = safe_get("ActualSOCPercentage", 100)
    faults["Fault_Low_SOC"]      = int((soc < FAULT_THRESHOLDS["min_soc_pct"]).any())

    # Total fault count for this window
    fault_flags = [v for k, v in faults.items() if k.startswith("Fault_") and isinstance(v, int)]
    faults["Total_Faults"]       = sum(fault_flags)
    faults["Window_Healthy"]     = int(faults["Total_Faults"] == 0)

    return faults


def compute_mission_summary(df: pd.DataFrame, spray_width_m: float | None = None) -> dict | None:
    """
    Compute a feature-rich summary for a single 30-second window.
    spray_width_m defaults to active vehicle config if not passed.
    """
    if df.empty:
        return None

    if spray_width_m is None:
        spray_width_m = Config.get().spray_width_m

    try:
        df = df.copy().sort_values("Timestamp")
        df = df.ffill().bfill()

        def safe_get(col, default=0.0):
            if col in df.columns:
                return df[col].fillna(default)
            logger.debug(f"Missing signal: {col}")
            return pd.Series(default, index=df.index)

        dt = df["Timestamp"].diff().fillna(0).clip(lower=0, upper=5)

        # ── Electrical ───────────────────────────────────────────────────────
        voltage          = safe_get("BatteryVoltage")
        current          = safe_get("Battery_current")
        power_w          = (voltage * current).abs()
        total_energy_wh  = (power_w * dt).sum() / 3600.0

        ctrl_power       = safe_get("Ctrl_power").abs()
        ctrl_energy_wh   = (ctrl_power * dt).sum() / 3600.0

        spray_v          = safe_get("BatteryVoltage_spray")
        spray_i          = safe_get("Battery_current_spray")
        spray_power      = (spray_v * spray_i).abs()
        spray_energy_wh  = (spray_power * dt).sum() / 3600.0

        avg_power_w      = float(power_w.mean())
        peak_power_w     = float(power_w.quantile(0.95))

        soc              = safe_get("ActualSOCPercentage")
        start_soc        = float(soc.iloc[0])
        end_soc          = float(soc.iloc[-1])
        soc_drop         = start_soc - end_soc
        start_rsoc       = float(safe_get("RSOC").iloc[0])
        start_rsoc2      = float(safe_get("RSOC_2").iloc[0])

        # ── Kinematics ───────────────────────────────────────────────────────
        speed_kmh        = safe_get("Speed")
        speed_mps        = (speed_kmh / 3.6).clip(lower=0)
        distance_m       = float((speed_mps * dt).sum())
        avg_speed        = float(speed_kmh.mean())
        max_speed        = float(speed_kmh.max())
        speed_std        = float(speed_kmh.std()) if len(speed_kmh) > 1 else 0.0

        accel_derived    = speed_mps.diff() / dt.replace(0, np.nan)
        rms_accel        = float(np.sqrt((accel_derived ** 2).mean(skipna=True)))
        accel_g          = safe_get("Vehicle_Acceleration")
        avg_accel_g      = float(accel_g.abs().mean())

        # ── Terrain ──────────────────────────────────────────────────────────
        gradient         = safe_get("Gradient")
        avg_gradient     = float(gradient.mean())
        max_gradient     = float(gradient.abs().max())
        uphill_frac      = float((gradient > 1.0).mean())

        # ── Motor / Drivetrain ───────────────────────────────────────────────
        motor_rpm        = safe_get("Motor_Rpm").abs()
        avg_rpm          = float(motor_rpm.mean())
        max_rpm          = float(motor_rpm.max())
        avg_hyd_rpm      = float(safe_get("RPM").mean())

        avg_ctrl_temp    = float(safe_get("Ctrl_Temperature").mean())
        avg_motor_temp   = float(safe_get("Tr_Mtr_Temp").mean())
        avg_hyd_temp     = float(safe_get("hyd_Motor_temperature").mean())
        avg_efficiency   = float(safe_get("Motor_ctrl_efficiency").mean())
        avg_cutback      = float(safe_get("Overall_Cutback").mean())

        gears            = safe_get("Gears")
        dominant_gear    = int(gears.mode().iloc[0]) if not gears.empty else 0
        low_gear_frac    = float((gears == 1).mean())
        high_gear_frac   = float((gears == 2).mean())

        # ── Spray ────────────────────────────────────────────────────────────
        pump_status      = safe_get("Spray_Pump_Status")
        spray_mask       = pump_status == 1
        spray_frac       = float(spray_mask.mean())
        spray_dist       = float((speed_mps[spray_mask] * dt[spray_mask]).sum())
        area_m2          = spray_dist * spray_width_m
        area_acres       = area_m2 / 4046.86

        avg_spray_pressure = float(safe_get("Spray_Pressure")[spray_mask].mean()) if spray_mask.any() else 0.0
        avg_spray_flow     = float(safe_get("Spray_Flowrate")[spray_mask].mean()) if spray_mask.any() else 0.0

        travel_frac      = float(safe_get("Travel_Mode").mean())
        field_frac       = float(safe_get("Field_Mode").mean())

        # ── IMU ──────────────────────────────────────────────────────────────
        ax, ay           = safe_get("Accel_X"), safe_get("Accel_Y")
        vibration        = float(np.sqrt((ax ** 2 + ay ** 2).mean()))

        # ── GNSS ─────────────────────────────────────────────────────────────
        lat              = safe_get("GNSS_Latitude",  default=float("nan"))
        lon              = safe_get("GNSS_Longitude", default=float("nan"))
        start_lat        = float(lat.dropna().iloc[0])  if lat.notna().any() else float("nan")
        start_lon        = float(lon.dropna().iloc[0])  if lon.notna().any() else float("nan")
        end_lat          = float(lat.dropna().iloc[-1]) if lat.notna().any() else float("nan")
        end_lon          = float(lon.dropna().iloc[-1]) if lon.notna().any() else float("nan")

        # ── Derived ratios ───────────────────────────────────────────────────
        energy_per_meter   = (total_energy_wh / distance_m) if distance_m > 0.5 else float("nan")
        spray_energy_frac  = (spray_energy_wh / total_energy_wh) if total_energy_wh > 0 else 0.0

        # ── Fault detection ──────────────────────────────────────────────────
        fault_info = detect_faults(df)

        result = {
            # Targets
            "Energy_Wh":              total_energy_wh,
            "Energy_per_meter":       energy_per_meter,
            # Distance / Area
            "Distance_m":             distance_m,
            "Area_acres":             area_acres,
            # Speed
            "Avg_Speed":              avg_speed,
            "Max_Speed":              max_speed,
            "Speed_Std":              speed_std,
            # Gradient
            "Avg_Gradient":           avg_gradient,
            "Max_Gradient":           max_gradient,
            "Uphill_Frac":            uphill_frac,
            # Power
            "Avg_Power_W":            avg_power_w,
            "Peak_Power_W":           peak_power_w,
            "Ctrl_Energy_Wh":         ctrl_energy_wh,
            "Spray_Energy_Wh":        spray_energy_wh,
            "Spray_Energy_Frac":      spray_energy_frac,
            # SOC
            "Start_SOC":              start_soc,
            "End_SOC":                end_soc,
            "SOC_Drop":               soc_drop,
            "Start_RSOC":             start_rsoc,
            "Start_RSOC2":            start_rsoc2,
            # Motor
            "Avg_RPM":                avg_rpm,
            "Max_RPM":                max_rpm,
            "Avg_Hyd_RPM":            avg_hyd_rpm,
            "Avg_Efficiency":         avg_efficiency,
            "Avg_Cutback":            avg_cutback,
            # Thermal
            "Avg_Ctrl_Temp":          avg_ctrl_temp,
            "Avg_Motor_Temp":         avg_motor_temp,
            "Avg_Hyd_Temp":           avg_hyd_temp,
            # Gear / Mode
            "Dominant_Gear":          dominant_gear,
            "Low_Gear_Frac":          low_gear_frac,
            "High_Gear_Frac":         high_gear_frac,
            "Travel_Frac":            travel_frac,
            "Field_Frac":             field_frac,
            # Spray
            "Spray_Ratio":            spray_frac,
            "Avg_Spray_Pressure":     avg_spray_pressure,
            "Avg_Spray_Flow":         avg_spray_flow,
            # Acceleration / IMU
            "Avg_Accel_g":            avg_accel_g,
            "RMS_Accel_Derived":      rms_accel,
            "Vibration":              vibration,
            # GNSS
            "Start_Lat":              start_lat,
            "Start_Lon":              start_lon,
            "End_Lat":                end_lat,
            "End_Lon":                end_lon,
        }
        result.update(fault_info)
        return result

    except Exception as e:
        logger.error(f"Error computing mission summary: {e}", exc_info=True)
        return None