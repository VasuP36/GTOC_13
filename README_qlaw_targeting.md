```markdown
# 🚀 Solar Sail Q-Law Targeting Simulation

This repository contains a Python-based numerical simulation of **solar sail trajectory optimization using the Q-law** (Lyapunov-based control). It allows simulation of **planet-to-planet transfers** using a fully dynamic solar sail model, adaptive integration, and automated flyby detection.

---

## 🛰 Overview

The **Q-law** (from Petropoulos, 2004) provides a Lyapunov control framework for continuous low-thrust (or solar sail) trajectory shaping.  
This script implements a **fully numerical version** of that formulation using **Modified Equinoctial Elements (MEE)** with finite-difference dynamics.  

Key features:
- Finite-difference based B-matrix relating RTN accelerations to MEE rates  
- Adaptive RK45 integration with tolerance control  
- Realistic RTN-frame solar sail acceleration model  
- Planetary ephemerides derived from user-supplied CSV  
- Flyby detection with `v_inf` computation  
- Animated 2D orbit GIFs (XY and YZ planes)

---

## ⚙️ Input and Usage

### Required Input
A CSV file `gtoc13_planets.csv` containing:
```

#Planet ID, Name, GM (km3/s2), Radius (km),
Semi-Major Axis (km), Eccentricity (), Inclination (deg),
Longitude of the Ascending Node (deg), Argument of Periapsis (deg),
Mean Anomaly at t=0 (deg), Weight ()

```

For example:
```

1,Vulcan,658906373.320,133020.700,13811982.942,0.000,0.000,0.000,315.372,322.584,0.1
2,Yavin,6363037.484,18013.200,128528229.968,0.050,3.000,110.499,148.135,155.310,1
3,Eden,443853.559,6697.400,179517444.840,0.007,1.000,107.472,356.208,51.897,2

````

### Run the Simulation
```bash
python3 solar_sail_q_law_final.py
````

It will:

1. Read the planet dataset
2. Initialize sail and planetary states
3. Run the Q-law integration from departure to arrival
4. Plot and optionally animate results
5. Stop automatically upon detecting a **flyby**

---

## 🧭 Core Parameters

All tunable constants are located at the top of the script.

| Parameter                 | Description                                 | Recommended Range          |
| ------------------------- | ------------------------------------------- | -------------------------- |
| `FD_DT`                   | Finite-difference step for MEE dynamics (s) | 1–10                       |
| `EPS_A`                   | Perturbation for numeric B-matrix (km/s²)   | 1e-7 – 1e-6                |
| `W_P, W_F, W_G, W_H, W_K` | Q-law weights for p, f, g, h, k             | Adjust to tune convergence |
| `DELTA_P...DELTA_K`       | Scaling values for Q computation            | Usually fixed              |
| `sail_C`                  | Sail light pressure constant (N/m² @ 1 AU)  | ~5.4e-6                    |
| `sail_A`                  | Sail area (m²)                              | 10,000–20,000              |
| `sail_m`                  | Spacecraft mass (kg)                        | 300–700                    |

Example:

```python
self.sail_params = {
    'sail_C': 5.4026e-6,
    'sail_A': 15000,
    'sail_m': 500
}
```

Increasing area or reducing mass increases acceleration.

---

## 🧮 Control Law Details

The Q-law minimizes a Lyapunov function:

$$
Q = \frac{1}{2}\sum_{i} \left( \frac{x_{i} - x_{i}^{*}}{w_{i} \Delta_{i}} \right)^{2}
$$

where ($x_{i}$) are current MEEs and ($x_{i}^{*}$) are target MEEs.

The gradient ($\nabla Q$) is combined with the numeric B-matrix:

$$
\dot{Q} = \mathbf{g}^{\mathsf{T}} \mathbf{a}_{\mathrm{RTN}} + c
$$

The control angles $\alpha$ and $\delta$ are chosen analytically to minimize ($\dot{Q}$) subject to sail geometry:

* $\alpha$ = cone angle (0–90°)
* $\delta$ = clock angle (0–360°)
* Acceleration magnitude $\cos^{2} \alpha$
where (x_i) are current MEEs and (x_i^*) are target MEEs.
The gradient (\nabla Q) is combined with the numeric B-matrix:
[
\dot{Q} = g^T a_{RTN} + c
]

The control angles α and δ are chosen analytically to minimize ( \dot{Q} ) subject to sail geometry:

* α = cone angle (0–90°)
* δ = clock angle (0–360°)
* Acceleration magnitude ∝ cos²(α)

---

## 🧰 Numerical Implementation

| Function                                       | Role                                                       |
| ---------------------------------------------- | ---------------------------------------------------------- |
| `compute_mees_dot_numeric()`                   | Computes finite-difference MEE rates for RTN accelerations |
| `build_B_numeric()`                            | Constructs local B-matrix (6×3)                            |
| `compute_g_and_c()`                            | Computes Lyapunov gradient and offset                      |
| `find_optimal_alpha()`, `find_optimal_delta()` | Compute sail control angles                                |
| `rk45_step()`                                  | Adaptive integrator with `rtol`/`atol` control             |
| `run_q_law()`                                  | Main loop: updates α, δ, integrates, checks flybys         |
| `make_gif()` / `make_gif_yz()`                 | Create orbit animations in XY and YZ planes                |

---

## 🛰 Flyby Detection

During propagation, the spacecraft–arrival distance is checked each step:

```python
if (1.01 * R_arr) <= rel_dist <= (101.0 * R_arr):
    print(f"Flyby detected at {t/86400:.2f} days → "
          f"distance={rel_dist/R_arr:.1f} R, v_inf={v_inf:.3f} km/s")
    break
```

* Stops simulation at first flyby
* `v_inf` is computed from relative velocity vector
* To log multiple encounters, remove `break` and store results in a list

---

## 🎥 Visual Outputs

After simulation, the script generates:

* `solar_sail_xy.gif`: Orbit evolution in the ecliptic plane (x–y)
* `solar_sail_yz.gif`: Inclination and out-of-plane view (y–z)
* `plot_trajectory.png`: Static 2D trajectory
* `control_angles.png`: α and δ vs time
* `Q_vs_time.png`: Lyapunov function evolution

Each GIF shows:

* 🟡 **Sun (Altaira)** at origin
* 🟢 **Departure planet**
* 🔴 **Arrival planet**
* 🔵 **Spacecraft** moving over time

Frames are sampled at 1 year or 1 month intervals.

---

## 🧠 Debugging and Fine-Tuning

| Symptom                        | Likely Cause                               | Adjustment                            |
| ------------------------------ | ------------------------------------------ | ------------------------------------- |
| Spacecraft shoots off straight | Wrong α/δ handling or sail too strong      | Clamp α < 85°, reduce sail area       |
| Orbit doesn’t expand           | Sail too weak / low `W_P`                  | Increase `W_P`, decrease mass         |
| α stuck at 90°                 | Flat gradient (numerical B too small)      | Increase `FD_DT` or `EPS_A`           |
| Simulation too noisy           | FD_DT too small                            | Use 5–10 s                            |
| Never reaches target           | Duration too short or Q weights unbalanced | Extend `T_days`, tune `W_F, W_G`      |
| Oscillating α/δ                | Step too coarse or weights too aggressive  | Increase RK45 `rtol`, reduce W values |

---

## 📊 Typical Parameter Sets

| Scenario             | sail_A (m²) | sail_m (kg) | W_P | FD_DT (s) | Notes                                  |
| -------------------- | ----------- | ----------- | --- | --------- | -------------------------------------- |
| Fast transfer        | 20000       | 300         | 0.5 | 5         | Quick convergence, risk of oscillation |
| Realistic baseline   | 15000       | 500         | 0.2 | 10        | Balanced, stable behavior              |
| Gentle orbit raising | 10000       | 700         | 0.1 | 10        | Smooth evolution, slower transfer      |

---

## 🧩 Extending the Model

* **Multi-leg missions:** Chain multiple `run_q_law()` calls using flyby outputs as initial conditions.
* **Batch optimization:** Loop over sail area/mass and Q-law weights to study performance sensitivity.
* **Alternate dynamics:** Replace numeric `build_B_numeric()` with analytical MEE Jacobian if desired.
* **Integration sweeps:** Automate convergence testing using YAML configs or Jupyter widgets.

---

## 🧾 Outputs and Logs

Example console output:

```
Simulating Vulcan → Yavin
DEBUG: mee=[1.381198e+07, 0.0, 0.0, 0.0, 0.0], g=[-0.000, 0.015, 0.076], Q=0.004201
Flyby detected at t = 1245.3 days → distance = 30.2 R, v_inf = 2.853 km/s
```

---

## 🪶 Notes

* This implementation uses **RTN-based sail acceleration** consistent with physical solar sail geometry (thrust away from Sun).
* α is measured from the anti-sun direction (so 0° = facing Sun, 90° = edge-on).
* Ensure your time horizon `T_days` is large enough (typically 3–15 years) to allow convergence.
* The system is fully nondimensionalizable if scaling for optimization or mission design sweeps is needed.

---

## 📚 References

1. Petropoulos, A.E. *Low-Thrust Trajectory Optimization Using a Q-Law*. AAS 04-108, 2004.
2. McInnes, C.R. *Solar Sailing: Technology, Dynamics, and Mission Applications*, Springer-Praxis, 1999.
3. Yuricst et al., *pyqlaw* ([https://github.com/Yuricst/pyqlaw](https://github.com/Yuricst/pyqlaw))

---

**Author:** Internal Lab Adaptation of the Q-Law Framework
**Version:** 1.0 — Stable baseline with flyby and visualization support
**License:** For academic / research use only

```
```
