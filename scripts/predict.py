#!/usr/bin/env python3
"""
predict.py — Mission planning CLI.

Examples:
  python scripts/predict.py --acres 5 --soc 80
  python scripts/predict.py --acres 10 --soc 70 --vehicle sprayer_v1
  python scripts/predict.py --acres 5 --soc 80 --gradient 3 --speed 3
  python scripts/predict.py --list-vehicles
"""
import argparse
from sprayer_energy.utils.config import Config
from sprayer_energy.engine.energy_engine import EnergyEngine


def main():
    parser = argparse.ArgumentParser(description="Sprayer energy mission planner.")
    parser.add_argument("--acres",    type=float, default=5.0)
    parser.add_argument("--soc",      type=float, default=80.0,  help="Starting SOC %%")
    parser.add_argument("--speed",    type=float, default=4.0,   help="Avg speed km/h")
    parser.add_argument("--gradient", type=float, default=0.0,   help="Avg gradient degrees")
    parser.add_argument("--spray",    type=float, default=0.85,  help="Spray ratio 0-1")
    parser.add_argument("--vehicle",  type=str,   default="sprayer_v2",
                        help="Vehicle config name")
    parser.add_argument("--model",    type=str,   default=None,
                        help="Override model .pkl path")
    parser.add_argument("--list-vehicles", action="store_true",
                        help="List available vehicle configs and exit")
    args = parser.parse_args()

    if args.list_vehicles:
        print("\nAvailable vehicle configs:")
        for v in Config.list_vehicles():
            cfg = Config.load(v)
            print(f"  {v:<20} {cfg.battery_capacity_wh/1000:.0f} kWh  "
                  f"{cfg.spray_width_m}m boom  — {cfg.description}")
        print()
        return

    engine = EnergyEngine(
        model_path=args.model,
        vehicle=args.vehicle,
    )
    cfg = Config.get()

    result = engine.plan_mission(
        acres=args.acres,
        start_soc=args.soc,
        avg_speed=args.speed,
        avg_gradient=args.gradient,
        spray_ratio=args.spray,
    )

    w = 50
    print(f"\n{'═'*w}")
    print(f"  Mission Plan — {args.acres} acres  [{cfg.name}]")
    print(f"  Battery: {cfg.battery_capacity_wh/1000:.0f} kWh  |  Boom: {cfg.spray_width_m}m")
    print(f"{'─'*w}")
    print(f"  Distance       : {result.distance_m:>10,.0f} m")
    print(f"  Energy needed  : {result.total_energy_wh:>10,.1f} Wh")
    print(f"  Required SOC   : {result.required_soc_pct:>9.1f} %")
    print(f"  SOC Margin     : {result.margin_soc_pct:>+9.1f} %")
    print(f"  Feasible       : {'✓ YES' if result.feasible else '✗ NO — recharge first':>10}")
    print(f"{'─'*w}")

    if not result.feasible:
        deficit_wh  = result.total_energy_wh - (args.soc / 100 * cfg.battery_capacity_wh)
        min_soc_pct = result.required_soc_pct
        print(f"  ⚠  Need {deficit_wh:,.0f} Wh more  |  Min SOC needed: {min_soc_pct:.1f}%")
        print(f"{'─'*w}")

    print(f"\n  Range table at current conditions (speed={args.speed}, grad={args.gradient}):")
    print(f"  {'SOC':>5}  {'Max Acres':>10}  {'Max Distance':>13}  {'Avail Wh':>10}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*13}  {'─'*10}")
    for soc_level in [100, 90, 80, 70, 60, 50, 30]:
        r = engine.estimate_range(
            current_soc=soc_level,
            avg_speed=args.speed,
            avg_gradient=args.gradient,
            spray_ratio=args.spray,
        )
        marker = " ◄" if soc_level == int(args.soc) else ""
        print(f"  {soc_level:>4}%  {r['Max_Acres']:>9.1f}a  "
              f"{r['Max_Distance_m']:>11,.0f} m  "
              f"{r['Available_Energy_Wh']:>9,.0f} W{marker}")
    print(f"{'═'*w}\n")


if __name__ == "__main__":
    main()