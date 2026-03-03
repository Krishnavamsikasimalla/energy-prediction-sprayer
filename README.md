# sprayer-energy — Advanced Pipeline

## Quick Start

```bash
# 1. Inspect new MF4 files before processing
python scripts/mf4_inspect.py data/raw/ --save

# 2. Diagnose DBC signal coverage vs actual MF4 data
python scripts/diagnose.py data/raw/ --vehicle sprayer_v2

# 3. Build training dataset
python scripts/build_dataset.py

# 4. Explore data in notebook
jupyter notebook notebooks/eda.ipynb

# 5. Train model
python scripts/train.py --trials 50

# 6. Validate
python scripts/validate.py

# 7. Predict
python scripts/predict.py --acres 5 --soc 80
python scripts/predict.py --acres 10 --soc 70 --vehicle sprayer_v1

# 8. Plot GNSS track
python scripts/plot_track.py --color energy
python scripts/plot_track.py --color spray
python scripts/plot_track.py --color fault
python scripts/plot_track.py --all

# 9. Run tests
pytest tests/ -v
```

## Vehicle Configs

| Config | Battery | Boom | Description |
|---|---|---|---|
| `sprayer_v2` | 20 kWh (2×10kWh, 412Ah @ 52.1V) | 10m | Production vehicle |
| `sprayer_v1` | 3 kWh | 1.5m | Original prototype |

Switch vehicle:
```bash
python scripts/predict.py --acres 5 --soc 80 --vehicle sprayer_v1
python scripts/predict.py --list-vehicles
```

## Key Files Changed

| File | What changed |
|---|---|
| `utils/config.py` | **NEW** — all constants centralised, multi-vehicle support |
| `data_pipeline/physics.py` | +GNSS, +fault detection, +spray battery power, 40+ features |
| `decoders/dbc_decoder.py` | Uses config for signal list and DBC paths |
| `engine/energy_engine.py` | Uses config, `MissionResult` dataclass, sensitivity analysis |
| `scripts/mf4_inspect.py` | Full signal quality report with batch mode |
| `scripts/diagnose.py` | **NEW** — DBC vs MF4 cross-check with per-signal stats |
| `scripts/plot_track.py` | **NEW** — GNSS track coloured by energy/speed/spray/fault |
| `scripts/predict.py` | Vehicle switching, deficit warnings, range table |
| `tests/test_decode.py` | **NEW** — 20 real unit tests |
| `tests/test_engine.py` | **NEW** — 18 real unit tests |
| `notebooks/eda.ipynb` | **NEW** — Full EDA notebook |

## Adding a New Vehicle

```python
# Option A: Add to utils/config.py VEHICLES dict
"sprayer_v3": VehicleConfig(
    name="sprayer_v3",
    description="Next gen vehicle",
    battery_capacity_wh=30000.0,
    battery_voltage_v=72.0,
    battery_capacity_ah=416.0,
    num_battery_packs=3,
    spray_width_m=12.0,
),

# Option B: Create docs/vehicles/sprayer_v3.json
# (use VehicleConfig.to_json() to generate template)
```

## Fault Thresholds (config.py)

```python
FAULT_THRESHOLDS = {
    "overall_cutback_pct": 50.0,
    "max_ctrl_temp_c":     80.0,
    "max_motor_temp_c":    90.0,
    "min_battery_voltage_v": 40.0,
    "min_soc_pct":         20.0,
}
```