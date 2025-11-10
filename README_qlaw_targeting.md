Perfect — that’s the right time to formalize this into a **mini operating manual** so anyone in your lab (or a future you) can pick it up and get meaningful results.
Here’s a clean, concise **instruction manual** for your `solar_sail_q_law_final.py` simulation, written like a proper internal research note.

---

## 🚀 **Solar Sail Q-Law Targeting Simulation Manual**

**Script:** `solar_sail_q_law_final.py`
**Purpose:**
Numerically simulate solar sail trajectory optimization using the **Q-law** (Lyapunov-based control) for **planet-to-planet transfers**, with:

* Finite-difference–based MEE dynamics (`B` matrix)
* Adaptive RK45 integration
* Realistic solar sail acceleration model
* Flyby detection and optional GIF visualizations

---

### ⚙️ 1. **Setup and Input Data**

**Input file required:**
`gtoc13_planets.csv` — must contain:

```
#Planet ID, Name, GM (km3/s2), Radius (km),
Semi-Major Axis (km), Eccentricity (), Inclination (deg),
Longitude of the Ascending Node (deg), Argument of Periapsis (deg),
Mean Anomaly at t=0 (deg), Weight ()
```

> For reference, GTOC13 planet entries (like Vulcan, Yavin, Eden) already work out-of-the-box.

---

### 🧭 2. **Basic Usage**

Run directly:

```bash
python3 solar_sail_q_law_final.py
```

What it does:

1. Reads `gtoc13_planets.csv`.
2. Initializes a solar sail model.
3. Runs Q-law–based transfer from `dep_id` → `arr_id`.
4. Plots trajectory, control angles, and optional GIFs.
5. Stops automatically if a **flyby** occurs (within 1.01–101 × planetary radii).

---

### 🪶 3. **Adjustable Parameters**

All tunable parameters are near the top of the script.

| Parameter                 | Description                                        | Typical Range / Notes                                                             |
| ------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------- |
| `MU_SUN`                  | Solar GM constant                                  | Keep default                                                                      |
| `FD_DT`                   | Time step for numeric finite-diff of MEEs          | 1–10 s; smaller = more accurate but slower                                        |
| `EPS_A`                   | Perturbation magnitude for B-matrix FD             | ~1e-7 to 1e-6 km/s²; too small → noisy, too big → nonlinear error                 |
| `W_P, W_F, W_G, W_H, W_K` | Q-law weights (per MEE component)                  | Tune to prioritize elements; `W_P` controls semi-major axis rate (transfer speed) |
| `DELTA_P, DELTA_F, ...`   | Scaling (Δ) for normalization in Lyapunov function | Usually not changed unless element scales differ greatly                          |
| `sail_params`             | Dict in class: `{sail_C, sail_A, sail_m}`          | Physical sail model; tuning this drastically changes transfer rate                |

#### Example:

```python
self.sail_params = {
    'sail_C': 5.4026e-6,   # N/m² at 1 AU
    'sail_A': 15e3,        # m² (sail area)
    'sail_m': 500.0        # kg (spacecraft mass)
}
```

Increasing area or decreasing mass → faster acceleration.

---

### 🧮 4. **Control Logic (Q-law Overview)**

The Q-law computes sail orientation (`α`, `δ`) that minimizes the Lyapunov function
[
Q = \frac{1}{2}\sum_i \left(\frac{(x_i - x_i^*)}{w_i \Delta_i}\right)^2
]
where (x_i) are current MEEs and (x_i^*) are target MEEs.

* The gradient `∂Q/∂x` is used with numeric B-matrix to form:
  [
  \dot{Q} = g^T a_{RTN} + c
  ]
* Then, α and δ are chosen analytically (within constraints) to minimize ( \dot{Q} ).

---

### 🧰 5. **Key Subsystems**

| Subsystem                                     | Description                                                        |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `compute_mees_dot_numeric()`                  | Finite-difference propagation of MEEs for RTN acceleration input   |
| `build_B_numeric()`                           | Builds local linearized B-matrix relating RTN accel → MEE rates    |
| `compute_g_and_c()`                           | Computes Lyapunov gradient (g) and offset (c)                      |
| `find_optimal_alpha() / find_optimal_delta()` | Solve for Q-law steering angles                                    |
| `rk45_step()`                                 | Adaptive integration (1 step of dynamics)                          |
| `run_q_law()`                                 | Main loop; integrates while updating α, δ, and checking for flybys |

---

### 🛰️ 6. **Flyby Detection**

Within each integration step, spacecraft–arrival distance is checked:

```python
if (1.01 * R_arr) <= rel_dist <= (101.0 * R_arr):
    print(f"Flyby detected at {t/86400:.2f} days → distance={rel_dist/R_arr:.1f} R, v_inf={v_inf:.3f} km/s")
    break
```

* **Stopping criterion:** simulation ends at first flyby.
* **`v_inf`** is the relative velocity magnitude (km/s).
* To detect **multiple flybys**, remove the `break` and store events in a list.

---

### 🎥 7. **GIF Visualization**

Two convenience methods produce orbit animations:

| Method                                        | Description                                       | Notes                                        |
| --------------------------------------------- | ------------------------------------------------- | -------------------------------------------- |
| `make_gif(data, dep_id, arr_id, yearly=True)` | Top-down XY orbit view (yearly or monthly frames) | Shows Sun, spacecraft, dep/arr planets       |
| `make_gif_yz(data, dep_id, arr_id)`           | Side-view (YZ-plane)                              | Same setup, useful for inclination evolution |

Both automatically color-code:

* 🟢 Departure planet
* 🔴 Arrival planet
* 🟡 Sun (Altaira)
* 🔵 Spacecraft trajectory

---

### 🪜 8. **Debugging and Fine-Tuning Workflow**

| Symptom                         | Likely Cause                                                 | Fix                                                  |
| ------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| Spacecraft shoots off radially  | α/δ logic wrong or sail too strong                           | Clamp α to ≤ 85° or reduce `sail_A/m`                |
| Orbit never expands             | Sail too weak or weights penalize `p` too little             | Increase `W_P` or reduce mass                        |
| α locked near 90°               | Gradients nearly zero (no meaningful Q variation)            | Increase `FD_DT` or `EPS_A` for B-matrix sensitivity |
| Simulation stalls               | `FD_DT` too small (numerical noise)                          | Use 5–10 s                                           |
| No flyby detection              | Target orbit mismatch (Q too small)                          | Verify `arr_id` and timing (T_days)                  |
| Unstable oscillations in angles | Try lowering `W_F, W_G`, or increase `rtol` in `rk45_step()` |                                                      |

---

### 📈 9. **Typical Parameter Setups**

| Scenario                       | `sail_A` (m²) | `sail_m` (kg) | `W_P` | `FD_DT` (s) | Comment                              |
| ------------------------------ | ------------- | ------------- | ----- | ----------- | ------------------------------------ |
| **Fast transfer (aggressive)** | 20,000        | 300           | 0.5   | 5           | Converges quickly, but may oscillate |
| **Realistic demo**             | 15,000        | 500           | 0.2   | 10          | Balanced, good baseline              |
| **Gentle orbit raising**       | 10,000        | 700           | 0.1   | 10          | Smooth α evolution, slow convergence |

---

### 🧠 10. **Extending the Script**

* **Multi-planet patching:** store flyby outputs, then relaunch `run_q_law()` for next leg.
* **Optimization sweeps:** wrap parameters (sail_A/m, W_P, FD_DT) in an outer loop and compare Q reduction rate.
* **Batch runs:** automate using YAML input (planet pairs + duration).

---

### 📊 11. **Outputs and Logs**

You’ll see debug output like:

```
DEBUG: mee=[1.381198e+07, 0.0, 0.0, 0.0, 0.0], g=[-0.000, 0.015, 0.076], Q=0.004201
🚀 Flyby detected at t = 1245.3 days → distance = 30.2 R, v_inf = 2.853 km/s
```

Files generated:

* `solar_sail_xy.gif` — top-down orbital animation
* `solar_sail_yz.gif` — inclination evolution
* `plot_trajectory.png` — static 2D trajectory
* `control_angles.png` — α, δ vs time

---

Would you like me to append this as a **header docstring block** to the top of your working script (so the next user sees it right away when opening the file)?
I can format it so it appears as a neatly commented “user manual” inside the code.
