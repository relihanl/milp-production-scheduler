# Producer MILP Scheduler

A 24-hour day-ahead production scheduler using **Mixed-Integer Linear
Programming (MILP)**.

The system optimizes when and how fast a configurable **producer**
should manufacture **widgets**, given:

-   Hourly electricity prices (CSV input)
-   Production rate limits
-   Minimum up / minimum down times
-   Optional ramp limits
-   Optional hourly input constraints
-   Nonlinear power consumption curve (approximated via piecewise linear
    MILP)
-   Daily production target

The output is an optimized 24-hour schedule written to CSV.

------------------------------------------------------------------------

## Features

-   Binary on/off scheduling\
-   Minimum up / minimum down time constraints\
-   Optional ramp rate constraints\
-   Startup cost penalties\
-   Daily production target (equality or minimum)\
-   Hourly production limits from CSV\
-   Nonlinear power curve using SOS2 piecewise linearization\
-   Solver support: **CBC** (recommended) or **GLPK**

------------------------------------------------------------------------

## Project Structure

```
producer_milp.py    # Main optimization script (with inline documentation)
config.yaml         # Configuration file
prices.csv          # 24-hour electricity prices (input)
schedule.csv        # Optimized output schedule (generated)
blog_post.md        # Comprehensive guide and insights
README.md           # This file
requirements.txt    # Python dependencies
LICENSE             # MIT License
```

------------------------------------------------------------------------

## Requirements

### Python Packages

pip install pyomo pandas pyyaml

### MILP Solver (Required)

Install one of the following:

#### CBC (Recommended)

Ubuntu / Debian: sudo apt install coinor-cbc

macOS (Homebrew): brew install cbc

Windows (Conda): conda install -c conda-forge coincbc

#### GLPK (Alternative)

Ubuntu: sudo apt install glpk-utils

macOS: brew install glpk

Windows (Conda): conda install -c conda-forge glpk

------------------------------------------------------------------------

## 🚀 Running the Optimizer

python producer_milp.py\
--config config.yaml\
--prices prices.csv\
--out schedule.csv\
--price-col price_eur_per_kwh\
--hour-col hour

------------------------------------------------------------------------

## 📥 Input Files

### prices.csv

Must contain 24 rows (one per hour).

Columns:

-   price_eur_per_kwh (required)
-   input_max_rate (optional)

------------------------------------------------------------------------

### config.yaml

Example configuration:

```yaml
solver: cbc
timestep_hours: 1.0

# Production rate limits when ON (widgets/hour)
rate_min: 10.0
rate_max: 100.0

# Daily production requirement (widgets)
daily_widget_target: 200.0
daily_target_mode: equality   # "equality" or "minimum"

# Optional startup penalty (€/start)
startup_cost: 0.0

# Min up/down times (hours). Set 0 to disable.
min_up_hours: 2
min_down_hours: 1

# Optional ramping limits
ramp_up: null
ramp_down: null

# Hourly input constraints (raw materials, staffing, etc.)
input_constraints:
  mode: csv_column               # "none", "constant", or "csv_column"
  max_rate_column: input_max_rate

# Power curve: power_kW = f(rate)
curve:
  type: polynomial
  a: 0.005    # Quadratic coefficient (inefficiency at high rates)
  b: 0.6      # Linear coefficient (direct energy per widget)
  c: 5.0      # Base load (fixed overhead when ON)
  n_points: 12

# Alternative: use measured breakpoints
# curve:
#   type: breakpoints
#   rate_points: [0, 20, 40, 60, 80, 100]
#   power_points: [0, 18, 45, 78, 118, 165]
```

**Note:** The `a` coefficient dramatically affects optimal strategy:
- Low `a` (0.001): Burst strategy - run hard during cheap hours
- Medium `a` (0.005): Balanced approach
- High `a` (0.015): Spread strategy - run longer at lower rates

See [blog_post.md](blog_post.md) for detailed analysis.

------------------------------------------------------------------------

## 📤 Output: schedule.csv

Columns:

-   hour\
-   price_eur_per_kwh\
-   input_max_rate\
-   on\
-   start\
-   stop\
-   rate_widgets_per_hour\
-   power_kw\
-   energy_kwh\
-   cost_eur

Header comments include:

-   solver_status\
-   total_cost_eur\
-   total_widgets\
-   daily_target

------------------------------------------------------------------------

## 🧠 Optimization Model Summary

Decision variables:

-   Binary on/off\
-   Production rate (continuous)\
-   Power consumption (continuous)

Objective:

Minimize total electricity cost + startup cost

Subject to:

-   Daily production requirement\
-   Minimum up/down time\
-   Ramp limits (optional)\
-   Input constraints\
-   Piecewise-linear power curve

------------------------------------------------------------------------

## 📊 Key Insights

The quadratic coefficient **a** in the power curve acts as a "burst penalty" dial:

| a value | Strategy | Operating Hours | Cost (200 widgets) |
|---------|----------|-----------------|-------------------|
| 0.001 | High-rate burst | 5 hours | €29.25 |
| 0.005 | Balanced moderate | 9 hours | €31.75 |
| 0.015 | Low-rate spread | 15 hours | €35.62 |

**Key finding:** Equipment efficiency characteristics fundamentally shape optimal scheduling strategies. Machines with high quadratic losses benefit from longer, gentler runs, while linear-scaling equipment should burst at high rates during cheap periods.

For a comprehensive analysis with experiments and insights, see **[blog_post.md](blog_post.md)**.

------------------------------------------------------------------------

## 📚 Learning Resources

- **[blog_post.md](blog_post.md)** - Comprehensive guide with:
  - MILP fundamentals
  - Complete problem formulation
  - Experimental results showing how equipment curves affect strategy
  - Solver performance analysis
  - Practical deployment advice

- **Inline code documentation** - The Python script includes detailed comments explaining mode parameters, constraints, and the piecewise-linear approximation technique.

------------------------------------------------------------------------

## 📜 License

Copyright © 2025 FullStackEnergy.com

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
