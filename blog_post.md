# Optimizing Production Schedules with Mixed-Integer Linear Programming

## Introduction: The Day-Ahead Scheduling Problem

Imagine you run a manufacturing facility that produces widgets 24/7. Your electricity costs vary by the hour—cheap at night, expensive during peak demand. Your machine can run at different production rates, but higher speeds consume more power (and not always proportionally). You have a daily production target to meet.

**The question:** When should you run the machine, and at what rate, to minimize your energy costs while meeting your production goals?

This is a classic **day-ahead scheduling problem**, and it's exactly the kind of challenge that Mixed-Integer Linear Programming (MILP) excels at solving.

In this post, I'll walk through building a complete MILP optimizer in Python using **Pyomo** and the **CBC solver**. Along the way, we'll discover some fascinating insights about how equipment efficiency curves fundamentally shape optimal production strategies. 

You can see the code [here in github](https://github.com/relihanl/milp-production-scheduler).

## What is MILP?

**Mixed-Integer Linear Programming** combines two types of decision variables:
- **Integer variables**: Discrete choices (e.g., machine ON/OFF, which shift to schedule)
- **Continuous variables**: Quantities that can vary smoothly (e.g., production rate, power consumption)

Unlike pure Linear Programming (LP), MILP can model real-world constraints like:
- Binary on/off decisions
- Minimum up/down times (can't cycle too quickly)
- Startup costs
- Discrete operating modes

The "Linear" part means the objective and constraints must be linear—but we'll see how to handle nonlinear power curves using clever approximations.

## The Problem: Production Scheduling with Nonlinear Power Consumption

### Inputs
1. **24-hour electricity prices** (€/kWh) - varying from €0.09 to €0.22
2. **Production target**: 200 widgets (must produce exactly this amount)
3. **Machine constraints**:
   - Rate limits: 10-100 widgets/hr when ON
   - Minimum up time: 2 hours (once started, must run at least 2 hours)
   - Minimum down time: 1 hour (once stopped, must stay off at least 1 hour)
4. **Power consumption curve** (nonlinear!):
   ```
   power(kW) = a × rate² + b × rate + c
   ```
   where:
   - **a**: Quadratic coefficient (inefficiency at high rates)
   - **b**: Linear coefficient (direct energy per widget)
   - **c**: Base load (fixed overhead when ON)

### Output
An optimal 24-hour schedule specifying:
- Which hours to run the machine (ON/OFF)
- Production rate for each hour
- Total energy cost

## The MILP Formulation

### Decision Variables

**Binary (Integer) Variables** - 72 total (24 hours × 3):
```python
m.on[t]    # 1 if machine ON at hour t, 0 otherwise
m.start[t] # 1 if machine starts at hour t
m.stop[t]  # 1 if machine stops at hour t
```

**Continuous Variables** - 48 total (24 hours × 2):
```python
m.rate[t]  # Production rate (widgets/hour) at hour t
m.power[t] # Power consumption (kW) at hour t
```

### Key Constraints

**1. Link rate to on/off state:**
```python
rate[t] >= rate_min * on[t]  # When ON: rate >= 10
rate[t] <= rate_max * on[t]  # When OFF: rate = 0
```

**2. Startup/shutdown logic:**
```python
start[t] >= on[t] - on[t-1]  # Detects OFF→ON transitions
stop[t] >= on[t-1] - on[t]   # Detects ON→OFF transitions
```

**3. Minimum up/down time:**
```python
# If we start at t, must stay on for min_up_hours
sum(on[k] for k in range(t, t+min_up_hours)) >= min_up_hours * start[t]

# If we stop at t, must stay off for min_down_hours
sum(1-on[k] for k in range(t, t+min_down_hours)) >= min_down_hours * stop[t]
```

**4. Daily production target:**
```python
sum(rate[t] for t in 0..23) == 200  # Equality mode
```

**5. Piecewise-linear power curve (the clever part!):**

Since `power = a×rate² + b×rate + c` is nonlinear, we approximate it using **piecewise-linear segments** with **SOS2 (Special Ordered Set type 2)** variables:

```python
m.pw = pyo.Piecewise(
    m.T,                    # Index: hours 0-23
    m.power,                # Dependent variable: power
    m.rate,                 # Independent variable: rate
    pw_pts=breakpoints,     # e.g., [0, 9, 18, 27, ..., 100]
    f_rule=power_values,    # Power at each breakpoint
    pw_constr_type="EQ",    # Equality: power = f(rate)
    pw_repn="SOS2",         # Use SOS2 for MILP efficiency
)
```

This creates 12 piecewise-linear segments that closely approximate the quadratic curve while keeping everything linear for the MILP solver.

### Objective Function

```python
minimize: energy_cost + startup_penalties

where:
  energy_cost = sum(price[t] * power[t] for t in 0..23)
  startup_penalties = startup_cost * sum(start[t] for t in 0..23)
```

## The Fascinating Role of the Quadratic Coefficient

Here's where things get interesting. By varying just the quadratic coefficient **a** in the power curve, we can simulate different types of machinery—and the optimizer adapts its strategy dramatically!

### Experiment 1: High Quadratic Penalty (a = 0.015)

**Power curve:** `power = 0.015×rate² + 0.6×rate + 5.0`

At maximum rate (100 widgets/hr):
- Quadratic term: 150 kW (70% of total!)
- Linear term: 60 kW (28%)
- Base load: 5 kW (2%)
- **Total: 215 kW**

This represents equipment with **severe inefficiencies at high speeds** (e.g., pumps with quadratic drag losses).

**Optimal Strategy:**
```
Hours ON: 0-7, 15-20, 23 (15 hours total)
Max rate: 27 widgets/hr (conservative)
Total cost: €35.62
```

The optimizer spreads production across many hours at low rates to avoid the crushing quadratic penalty.

### Experiment 2: Low Quadratic Penalty (a = 0.001)

**Power curve:** `power = 0.001×rate² + 0.6×rate + 5.0`

At maximum rate (100 widgets/hr):
- Quadratic term: 10 kW (13%)
- Linear term: 60 kW (80%)
- Base load: 5 kW (7%)
- **Total: 75 kW**

This represents **highly scalable equipment** with mostly linear power consumption.

**Optimal Strategy:**
```
Hours ON: 2-6 (5 hours only!)
Max rate: 60 widgets/hr (aggressive burst)
Total cost: €29.25 (18% savings!)
```

The optimizer concentrates all production in the cheapest hours (4-5am at €0.09/kWh) and runs at maximum feasible rates. This is a "sprint during cheap hours, stop during expensive hours" strategy.

### Experiment 3: Medium Quadratic Penalty (a = 0.005)

**Power curve:** `power = 0.005×rate² + 0.6×rate + 5.0`

At maximum rate:
- Quadratic term: 50 kW (43%)
- **Total: 115 kW**

**Optimal Strategy:**
```
Hours ON: 0-6, 18-19 (9 hours)
Max rate: 36 widgets/hr (balanced)
Total cost: €31.75
```

A perfect middle ground—moderate rates, focused on cheaper hours.

### The Strategy Spectrum

| a value | Cost | Hours ON | Max Rate | Strategy |
|---------|------|----------|----------|----------|
| 0.001 | €29.25 | 5 | 60 | **High-rate burst** |
| 0.005 | €31.75 | 9 | 36 | **Balanced moderate** |
| 0.015 | €35.62 | 15 | 27 | **Low-rate spread** |

**Visual representation:**
```
a=0.001 (€29.25):  ████████░░░░░░░░░░░░░░  (burst)
a=0.005 (€31.75):  ███████░░░░░░░░░░░██░░  (balanced)
a=0.015 (€35.62):  ████████░░░░░░██████░█  (spread)
                   0        8        16   24 (hours)
```

## Real-World Solver Performance

Using the **CBC (COIN-OR Branch and Cut)** solver, here's what happens:

```
Problem size: 191 constraints, 383 variables (71 binary)
LP relaxation: €29.25 (lower bound)
Cutting planes: 13 cuts added (Gomory, Probing, MIR)
Integer solution: €29.25 (found in 0 nodes!)
Solve time: 0.01 seconds
```

**Key insight:** The cutting planes were so effective that the LP relaxation bound matched the integer solution exactly—no branch-and-bound exploration needed! This is why MILP solvers are so powerful for well-structured problems.

## Practical Takeaways

### 1. Equipment Efficiency Curves Matter—A Lot

The quadratic coefficient **a** acts as a "burst penalty" dial:
- **Low a** (scalable equipment): Burst strategy during cheap periods
- **High a** (inefficient at high rates): Spread strategy across more hours

Before optimizing, **measure your actual equipment's power curve!** Use the breakpoints configuration:

```yaml
curve:
  type: breakpoints
  rate_points: [0, 10, 20, 40, 60, 80, 100]
  power_points: [0, 12, 25, 52, 85, 130, 195]  # From real measurements
```

### 2. MILP Can Handle Complexity

This problem has:
- ✅ Binary on/off decisions
- ✅ Minimum up/down time constraints
- ✅ Nonlinear power curves (approximated)
- ✅ Hourly varying constraints and prices

Yet it solves in **0.01 seconds** on a laptop.

### 3. The Base Load (c) Creates Interesting Trade-offs

The fixed 5 kW base load when ON:
- Makes short, low-rate runs inefficient (overhead dominates)
- Encourages either running longer or at higher rates
- Creates the "sprint vs. marathon" dynamic we observed

### 4. Constraints Shape Solutions in Non-Obvious Ways

The `min_up_hours=2` constraint forces at least 2-hour runs, preventing the optimizer from doing rapid on/off cycling. Without this, it might cycle every hour to chase price fluctuations (unrealistic for real equipment).

## The Code

The complete optimizer is ~400 lines of Python and supports:
- Polynomial or breakpoint power curves
- CSV input for prices and hourly constraints
- Configurable min up/down times, ramp rates, startup costs
- Both equality and minimum production targets

Key dependencies:
```bash
pip install pyomo pandas pyyaml
sudo apt install coinor-cbc  # or: brew install cbc
```

Run it:
```bash
python producer_milp.py \
  --config config.yaml \
  --prices prices.csv \
  --out schedule.csv
```

## Conclusion

Mixed-Integer Linear Programming is a powerful tool for optimization problems with both discrete decisions (on/off) and continuous variables (rates, power). By modeling our manufacturing problem as a MILP:

1. We found **18% cost savings** by tuning equipment strategy to match its efficiency characteristics
2. We discovered how quadratic efficiency curves fundamentally change optimal scheduling strategies
3. We solved a complex 24-hour scheduling problem in **10 milliseconds**

The real magic happens when you combine:
- Domain knowledge (equipment physics, operational constraints)
- Mathematical modeling (MILP formulation)
- Modern solvers (CBC, Gurobi, CPLEX)

Whether you're scheduling production, optimizing energy systems, planning logistics, or routing vehicles—MILP is likely the right tool for the job.

---

**Try it yourself!** The full code is available in this repository. Experiment with:
- Different power curves (try a=0.0 for purely linear!)
- Startup costs (penalize frequent on/off cycling)
- Ramp rate limits (constrain how fast production can change)
- Different price profiles (solar-heavy grids have negative prices!)

Happy optimizing! 🚀

---

## License

Copyright © 2025 FullStackEnergy.com

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Have questions or found an interesting variant? Drop a comment below or open an issue on GitHub!*
