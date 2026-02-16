#!/usr/bin/env python3
"""
MILP day-ahead scheduling for a "producer" that makes "widgets" over 24 hours.

Copyright (c) 2025 FullStackEnergy.com
Licensed under the MIT License. See LICENSE file for details.

- Input: hourly energy prices from CSV (24 rows)
- Output: optimized 24h production schedule to CSV
- Config: YAML controls min up/down time, daily widget requirement, and input (inlet) constraints
- Producer rate is continuously variable when ON
- Energy use depends on production rate via a nonlinear curve approximated by piecewise-linear MILP

Dependencies:
  pip install pyomo pandas pyyaml

Solver:
  - CBC (recommended) or GLPK
    Ubuntu: sudo apt-get install coinor-cbc  OR  sudo apt-get install glpk-utils

Run:
  python producer_milp.py --config config.yaml --prices prices.csv --out schedule.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import yaml
import pyomo.environ as pyo


# -----------------------------
# Config + curve handling
# -----------------------------

@dataclass
class CurveSpec:
    kind: str  # "polynomial" or "breakpoints"
    # polynomial
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    # breakpoints
    rate_points: Optional[List[float]] = None
    power_points: Optional[List[float]] = None
    # sampling
    n_points: int = 12


@dataclass
class ModelSpec:
    solver: str
    timestep_hours: float

    rate_min: float
    rate_max: float

    daily_widget_target: float
    daily_target_mode: str  # "equality" or "minimum"

    startup_cost: float

    min_up_hours: int
    min_down_hours: int

    ramp_up: Optional[float]    # widgets/hour per hour
    ramp_down: Optional[float]

    input_constraints: Dict     # limits max rate each hour (e.g., raw material availability)

    curve: CurveSpec


def load_yaml_config(path: str) -> ModelSpec:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    curve_cfg = cfg.get("curve", {})
    curve = CurveSpec(
        kind=str(curve_cfg.get("type", "polynomial")).strip().lower(),
        a=float(curve_cfg.get("a", 0.0)),
        b=float(curve_cfg.get("b", 0.0)),
        c=float(curve_cfg.get("c", 0.0)),
        rate_points=curve_cfg.get("rate_points"),
        power_points=curve_cfg.get("power_points"),
        n_points=int(curve_cfg.get("n_points", 12)),
    )

    input_cfg = cfg.get("input_constraints", {"mode": "none"})

    return ModelSpec(
        solver=str(cfg.get("solver", "cbc")),
        timestep_hours=float(cfg.get("timestep_hours", 1.0)),

        rate_min=float(cfg["rate_min"]),
        rate_max=float(cfg["rate_max"]),

        daily_widget_target=float(cfg["daily_widget_target"]),
        daily_target_mode=str(cfg.get("daily_target_mode", "equality")).strip().lower(),

        startup_cost=float(cfg.get("startup_cost", 0.0)),

        min_up_hours=int(cfg.get("min_up_hours", 0)),
        min_down_hours=int(cfg.get("min_down_hours", 0)),

        ramp_up=(None if cfg.get("ramp_up") in (None, "", "null") else float(cfg.get("ramp_up"))),
        ramp_down=(None if cfg.get("ramp_down") in (None, "", "null") else float(cfg.get("ramp_down"))),

        input_constraints=input_cfg,
        curve=curve,
    )


def power_curve_value(curve: CurveSpec, rate: float) -> float:
    """Evaluate the power curve (kW) at a given production rate."""
    if curve.kind == "polynomial":
        return curve.a * rate * rate + curve.b * rate + curve.c
    raise ValueError("Direct evaluation only supported for polynomial; breakpoints handled separately.")


def build_piecewise_points(
    curve: CurveSpec,
    rate_max: float,
) -> Tuple[List[float], List[float]]:
    """
    Build (rate_points, power_points) for piecewise-linear approximation.
    Always includes rate=0 so that 'off' is feasible with power=0.
    """
    if curve.kind == "breakpoints":
        if not curve.rate_points or not curve.power_points:
            raise ValueError("curve.type=breakpoints requires curve.rate_points and curve.power_points")
        if len(curve.rate_points) != len(curve.power_points):
            raise ValueError("curve.rate_points and curve.power_points must have same length")

        pts = [float(x) for x in curve.rate_points]
        vals = [float(y) for y in curve.power_points]
        if min(pts) > 0.0:
            pts = [0.0] + pts
            vals = [0.0] + vals
        return pts, vals

    if curve.kind == "polynomial":
        n_pts = max(2, int(curve.n_points))
        pts = [0.0 + i * (rate_max - 0.0) / (n_pts - 1) for i in range(n_pts)]
        vals = [float(power_curve_value(curve, r)) for r in pts]
        vals[0] = max(0.0, vals[0])
        return pts, vals

    raise ValueError(f"Unknown curve.type: {curve.kind}")


# -----------------------------
# Data I/O
# -----------------------------

def load_prices_csv(path: str, price_col: str, hour_col: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if hour_col and hour_col in df.columns:
        df = df.sort_values(hour_col).reset_index(drop=True)

    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in {path}. Columns: {list(df.columns)}")

    if len(df) != 24:
        raise ValueError(f"Expected 24 rows in prices CSV, got {len(df)}")

    return df.reset_index(drop=True)


def get_input_max_series(df: pd.DataFrame, input_cfg: Dict, rate_max: float) -> List[float]:
    """
    Hourly cap on production rate (e.g., raw material, staffing, upstream capacity).
    """
    mode = str(input_cfg.get("mode", "none")).strip().lower()
    """
    "none"
    No input constraints applied
    Uses rate_max from config for all 24 hours
    Example: All hours limited to 100 widgets/hr

    "constant"
    Same limit for all 24 hours (different from rate_max)
    Requires: max_rate_constant parameter
    Example: Raw material limits production to 50 widgets/hr every hour

    "csv_column" (currently configured)
    Different limit for each hour, read from CSV
    Requires: max_rate_column parameter pointing to column name
    Example: prices.csv has input_max_rate column with varying limits (60-100)
    """
    if mode == "none":
        return [rate_max] * 24

    if mode == "constant":
        v = float(input_cfg["max_rate_constant"])
        return [v] * 24

    if mode == "csv_column":
        col = str(input_cfg["max_rate_column"])
        if col not in df.columns:
            raise ValueError(f"input_constraints.mode=csv_column but column '{col}' not found in prices CSV.")
        return [float(x) for x in df[col].tolist()]

    raise ValueError(f"Unknown input_constraints.mode: {mode}")


# -----------------------------
# MILP model
# -----------------------------

def solve_schedule(
    spec: ModelSpec,
    prices_df: pd.DataFrame,
    price_col: str,
) -> pd.DataFrame:
    prices = [float(x) for x in prices_df[price_col].tolist()]
    input_max = get_input_max_series(prices_df, spec.input_constraints, spec.rate_max)

    dt = float(spec.timestep_hours)
    T = list(range(24))

    pw_pts, pw_vals = build_piecewise_points(spec.curve, spec.rate_max)

    m = pyo.ConcreteModel("producer_24h_milp")
    m.T = pyo.RangeSet(0, 23)

    # Decision vars
    m.on = pyo.Var(m.T, domain=pyo.Binary)
    m.start = pyo.Var(m.T, domain=pyo.Binary)
    m.stop = pyo.Var(m.T, domain=pyo.Binary)

    # rate: widgets/hour
    m.rate = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, spec.rate_max))
    # power: kW
    m.power = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # Link rate to on/off, enforce min rate when on
    def _rate_upper(m, t):
        return m.rate[t] <= spec.rate_max * m.on[t]
    m.rate_upper = pyo.Constraint(m.T, rule=_rate_upper)

    def _rate_lower(m, t):
        return m.rate[t] >= spec.rate_min * m.on[t]
    m.rate_lower = pyo.Constraint(m.T, rule=_rate_lower)

    # Input constraint: hourly max feasible rate
    def _input_rule(m, t):
        return m.rate[t] <= float(input_max[t])
    m.input_limit = pyo.Constraint(m.T, rule=_input_rule)

    # Startup/stop logic (assume off before t=0)
    def _start_rule(m, t):
        if t == 0:
            return m.start[t] >= m.on[t]
        return m.start[t] >= m.on[t] - m.on[t - 1]
    m.start_logic = pyo.Constraint(m.T, rule=_start_rule)

    def _stop_rule(m, t):
        if t == 0:
            return m.stop[t] >= 0
        return m.stop[t] >= m.on[t - 1] - m.on[t]
    m.stop_logic = pyo.Constraint(m.T, rule=_stop_rule)

    # Min up time
    U = int(spec.min_up_hours)
    if U > 0:
        def _min_up(m, t):
            end = min(23, t + U - 1)
            return sum(m.on[k] for k in range(t, end + 1)) >= (end - t + 1) * m.start[t]
        m.min_up = pyo.Constraint(m.T, rule=_min_up)

    # Min down time
    D = int(spec.min_down_hours)
    if D > 0:
        def _min_down(m, t):
            end = min(23, t + D - 1)
            return sum(1 - m.on[k] for k in range(t, end + 1)) >= (end - t + 1) * m.stop[t]
        m.min_down = pyo.Constraint(m.T, rule=_min_down)

    # Optional ramp constraints (rate changes)
    if spec.ramp_up is not None:
        RU = float(spec.ramp_up)
        def _ramp_up(m, t):
            if t == 0:
                return pyo.Constraint.Skip
            return m.rate[t] - m.rate[t - 1] <= RU * dt
        m.ramp_up = pyo.Constraint(m.T, rule=_ramp_up)

    if spec.ramp_down is not None:
        RD = float(spec.ramp_down)
        def _ramp_down(m, t):
            if t == 0:
                return pyo.Constraint.Skip
            return m.rate[t - 1] - m.rate[t] <= RD * dt
        m.ramp_down = pyo.Constraint(m.T, rule=_ramp_down)

    # Daily widgets constraint
    """
    "equality" (default)

    Must produce exactly the target number of widgets
    Constraint: total_widgets == daily_widget_target
    Example: If target is 200, must produce exactly 200 (not 199 or 201)
    "minimum"

    Must produce at least the target (can produce more)
    Constraint: total_widgets >= daily_widget_target
    Example: If target is 200, can produce 200, 250, 300, etc.
    Useful when overproduction is allowed
    """
    total_widgets = sum(m.rate[t] * dt for t in m.T)
    mode = spec.daily_target_mode
    if mode == "equality":
        m.daily_target = pyo.Constraint(expr=total_widgets == float(spec.daily_widget_target))
    elif mode == "minimum":
        m.daily_target = pyo.Constraint(expr=total_widgets >= float(spec.daily_widget_target))
    else:
        raise ValueError("daily_target_mode must be 'equality' or 'minimum'")

    """
    Implements the relationship: power[t] = f(rate[t]) for the nonlinear power 
    consumption curve.

    The actual curve is quadratic:

    power = 0.015 × rate² + 0.6 × rate + 5.0
    But quadratic constraints aren't linear! So it approximates the curve using 
    connected line segments.
    """
    m.pw = pyo.Piecewise(
        m.T,                    # 1. Index set: apply to all hours (0-23)
        m.power,                # 2. Dependent variable (y-axis): power output
        m.rate,                 # 3. Independent variable (x-axis): production rate input
        pw_pts=pw_pts,          # 4. Breakpoints (x-coords): [0, 9.09, 18.18, ..., 100]
        f_rule=pw_vals,         # 5. Function values (y-coords): [0, 12.5, 20.8, ..., 215]
        pw_constr_type="EQ",    # 6. Constraint type: power = f(rate) (equality)
        pw_repn="SOS2",         # 7. Representation: Special Ordered Set type 2
    )

    # Objective: energy cost + startup cost
    startup_cost = float(spec.startup_cost)
    def _obj(m):
        energy_cost = sum(prices[t] * (m.power[t] * dt) for t in m.T)  # €/kWh * kWh
        return energy_cost + startup_cost * sum(m.start[t] for t in m.T)
    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    # Solve
    opt = pyo.SolverFactory(spec.solver)
    if not opt.available(False):
        raise RuntimeError(
            f"Solver '{spec.solver}' not available. Install it or change config solver to 'cbc' or 'glpk'."
        )

    res = opt.solve(m, tee=True)
    term = str(res.solver.termination_condition)

    # Output schedule
    out_rows = []
    total_cost = 0.0
    total_widgets_out = 0.0

    for t in T:
        on = int(round(pyo.value(m.on[t])))
        start = int(round(pyo.value(m.start[t])))
        stop = int(round(pyo.value(m.stop[t])))
        rate = float(pyo.value(m.rate[t]))
        power = float(pyo.value(m.power[t]))
        energy_kwh = power * dt
        cost = prices[t] * energy_kwh

        total_cost += cost
        total_widgets_out += rate * dt

        out_rows.append({
            "hour": t,
            "price_eur_per_kwh": prices[t],
            "input_max_rate": float(input_max[t]),
            "on": on,
            "start": start,
            "stop": stop,
            "rate_widgets_per_hour": rate,
            "power_kw": power,
            "energy_kwh": energy_kwh,
            "cost_eur": cost,
        })

    out = pd.DataFrame(out_rows)
    out.attrs["solver_status"] = term
    out.attrs["total_cost_eur"] = total_cost
    out.attrs["total_widgets"] = total_widgets_out
    out.attrs["daily_target"] = float(spec.daily_widget_target)
    return out


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML configuration file")
    ap.add_argument("--prices", required=True, help="CSV file with 24 hourly prices (and optional input constraint column)")
    ap.add_argument("--out", required=True, help="Output schedule CSV path")
    ap.add_argument("--price-col", default="price_eur_per_kwh", help="Column name in prices CSV for €/kWh")
    ap.add_argument("--hour-col", default=None, help="Optional hour column for sorting (e.g., 'hour')")
    args = ap.parse_args()

    spec = load_yaml_config(args.config)

    prices_df = load_prices_csv(args.prices, price_col=args.price_col, hour_col=args.hour_col)

    schedule_df = solve_schedule(spec, prices_df, price_col=args.price_col)

    # Write output CSV with a few metadata comment lines at the top (lines starting with #)
    header_lines = [
        f"# solver_status,{schedule_df.attrs.get('solver_status')}",
        f"# total_cost_eur,{schedule_df.attrs.get('total_cost_eur')}",
        f"# total_widgets,{schedule_df.attrs.get('total_widgets')}",
        f"# daily_target,{schedule_df.attrs.get('daily_target')}",
    ]

    csv_text = schedule_df.to_csv(index=False)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
        f.write(csv_text)

    print("Wrote:", args.out)
    print("Solver:", schedule_df.attrs.get("solver_status"))
    print("Total cost (€):", round(schedule_df.attrs.get("total_cost_eur", 0.0), 4))
    print("Total widgets:", round(schedule_df.attrs.get("total_widgets", 0.0), 4))


if __name__ == "__main__":
    main()