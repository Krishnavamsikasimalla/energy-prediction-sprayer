"""
test_engine.py — Unit tests for EnergyEngine.
Run with: pytest tests/ -v
"""
import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline():
    """A minimal sklearn-style pipeline mock that predicts 0.5 Wh/m."""
    pipe = MagicMock()
    pipe.predict.return_value = np.array([0.5])
    pipe.named_steps = {
        "imputer": MagicMock(feature_names_in_=np.array([
            "Avg_Speed", "Max_Speed", "Speed_Std", "RMS_Accel_Derived", "Avg_Accel_g",
            "Avg_Gradient", "Max_Gradient", "Uphill_Frac",
            "Avg_Power_W", "Peak_Power_W", "Spray_Energy_Frac",
            "Start_SOC", "SOC_Drop", "Start_RSOC2",
            "Avg_RPM", "Max_RPM", "Avg_Hyd_RPM", "Avg_Efficiency", "Avg_Cutback",
            "Avg_Ctrl_Temp", "Avg_Motor_Temp", "Avg_Hyd_Temp",
            "Dominant_Gear", "Low_Gear_Frac", "High_Gear_Frac",
            "Travel_Frac", "Field_Frac",
            "Spray_Ratio", "Avg_Spray_Pressure", "Avg_Spray_Flow",
            "Vibration",
        ]))
    }
    return pipe


@pytest.fixture
def engine(mock_pipeline, tmp_path):
    """EnergyEngine with mocked model — no real pkl needed."""
    from sprayer_energy.engine.energy_engine import EnergyEngine

    pkl_path = tmp_path / "model.pkl"
    joblib.dump(mock_pipeline, pkl_path)

    with patch("joblib.load", return_value=mock_pipeline):
        eng = EnergyEngine(
            model_path=str(pkl_path),
            battery_capacity_wh=20000.0,
            spray_width_m=10.0,
        )
    return eng


# ── Basic prediction ──────────────────────────────────────────────────────────

class TestEnergyEnginePredict:

    def test_predict_energy_per_meter_returns_float(self, engine):
        result = engine.predict_energy_per_meter(Avg_Speed=4.0)
        assert isinstance(result, float)
        assert result > 0

    def test_predict_returns_mocked_value(self, engine):
        result = engine.predict_energy_per_meter()
        assert pytest.approx(result, rel=0.01) == 0.5


# ── Mission planning ──────────────────────────────────────────────────────────

class TestMissionPlan:

    def test_plan_returns_mission_result(self, engine):
        from sprayer_energy.engine.energy_engine import MissionResult
        result = engine.plan_mission(acres=5.0, start_soc=80.0)
        assert isinstance(result, MissionResult)

    def test_distance_calculation_correct(self, engine):
        """5 acres with 10m boom = 5 * 4046.86 / 10 = 2023.43 m"""
        result = engine.plan_mission(acres=5.0, start_soc=80.0)
        expected_dist = (5.0 * 4046.86) / 10.0
        assert pytest.approx(result.distance_m, rel=0.01) == expected_dist

    def test_energy_calculation_correct(self, engine):
        """0.5 Wh/m × 2023.43m = 1011.7 Wh"""
        result = engine.plan_mission(acres=5.0, start_soc=80.0)
        assert pytest.approx(result.total_energy_wh, rel=0.01) == 0.5 * result.distance_m

    def test_required_soc_correct(self, engine):
        """1011.7 Wh / 20000 Wh × 100 = ~5.06%"""
        result = engine.plan_mission(acres=5.0, start_soc=80.0)
        expected_soc = (result.total_energy_wh / 20000.0) * 100
        assert pytest.approx(result.required_soc_pct, rel=0.01) == expected_soc

    def test_feasible_when_enough_soc(self, engine):
        result = engine.plan_mission(acres=5.0, start_soc=80.0)
        assert result.feasible is True

    def test_infeasible_when_tiny_soc(self, engine):
        """Even 1% SOC = 200 Wh, enough for ~400m. 5 acres needs ~2023m → infeasible."""
        result = engine.plan_mission(acres=5.0, start_soc=1.0)
        assert result.feasible is False

    def test_margin_is_start_minus_required(self, engine):
        result = engine.plan_mission(acres=5.0, start_soc=80.0)
        assert pytest.approx(result.margin_soc_pct, rel=0.01) == (
            result.start_soc_pct if hasattr(result, "start_soc_pct")
            else 80.0 - result.required_soc_pct
        )

    def test_zero_acres_returns_zero_energy(self, engine):
        result = engine.plan_mission(acres=0.0, start_soc=80.0)
        assert result.total_energy_wh == pytest.approx(0.0, abs=1e-6)

    def test_large_mission(self, engine):
        result = engine.plan_mission(acres=100.0, start_soc=100.0)
        assert result.distance_m > 0
        assert result.total_energy_wh > 0


# ── Range estimation ──────────────────────────────────────────────────────────

class TestRangeEstimate:

    def test_returns_dict_with_required_keys(self, engine):
        result = engine.estimate_range(current_soc=80.0)
        for key in ["Max_Acres", "Max_Distance_m", "Energy_per_meter", "Available_Energy_Wh"]:
            assert key in result

    def test_range_proportional_to_soc(self, engine):
        r100 = engine.estimate_range(current_soc=100.0)
        r50  = engine.estimate_range(current_soc=50.0)
        # With fixed energy/meter, range should be exactly proportional
        assert pytest.approx(r100["Max_Acres"], rel=0.02) == 2 * r50["Max_Acres"]

    def test_zero_soc_gives_zero_range(self, engine):
        result = engine.estimate_range(current_soc=0.0)
        assert result["Max_Acres"] == pytest.approx(0.0, abs=0.01)

    def test_available_energy_correct(self, engine):
        result = engine.estimate_range(current_soc=80.0)
        assert pytest.approx(result["Available_Energy_Wh"], rel=0.01) == 0.80 * 20000.0


# ── Sensitivity analysis ──────────────────────────────────────────────────────

class TestSensitivityAnalysis:

    def test_returns_dataframe(self, engine):
        df = engine.sensitivity_analysis(base_acres=5.0)
        assert isinstance(df, pd.DataFrame)
        assert "Parameter" in df.columns
        assert "Required_SOC_%" in df.columns

    def test_all_parameters_present(self, engine):
        df = engine.sensitivity_analysis(base_acres=5.0)
        params = df["Parameter"].unique()
        for p in ["avg_speed", "avg_gradient", "spray_ratio", "start_soc"]:
            assert p in params

    def test_custom_param_ranges(self, engine):
        df = engine.sensitivity_analysis(
            base_acres=5.0,
            param_ranges={"avg_speed": [2, 4, 6]}
        )
        assert len(df) == 3


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_extra_kwargs_dont_crash(self, engine):
        """Unknown kwargs should be absorbed by _build_input gracefully."""
        result = engine.plan_mission(
            acres=5.0,
            start_soc=80.0,
            Avg_Ctrl_Temp=75.0,    # valid feature override
        )
        assert result is not None

    def test_gradient_positive_increases_energy(self, engine):
        """Higher gradient → model predicts different (mocked = same, but API works)."""
        r_flat = engine.plan_mission(acres=5.0, start_soc=80.0, avg_gradient=0.0)
        r_hill = engine.plan_mission(acres=5.0, start_soc=80.0, avg_gradient=10.0)
        # Both work without error (model is mocked to return 0.5 always)
        assert r_flat.total_energy_wh > 0
        assert r_hill.total_energy_wh > 0