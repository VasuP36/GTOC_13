import numpy as np
import solar_sail_planner as gtoc

# 1. Load celestial body registry
bodies = gtoc.load_registry_from_csvs("GTOC_13/content/gtoc13_planets.csv")

# Look up PlanetX by name (case-insensitive)
planetx = next(b for b in bodies.values() if b.name.lower() == "planetx")

# 2. Initial state (heliocentric, km and km/s)
r0 = np.array([
    -200.0 * gtoc.AU,
     30.823783300933 * gtoc.AU,
    -10.706755180845 * gtoc.AU
])
v0 = np.array([31.381, 0.0, 0.0])  # km/s
t0 = 0.0  # seconds

# 3. Plan trajectory (using spherical-angle control)
res = gtoc.plan_solar_sail_to_body(
    r0=r0, v0=v0, t0=t0, target_body=planetx,
    tf_days_range=(3000, 20000),
    N_segments=10,
    popsize=22,
    iters=50,
    rng_seed=0
)

# 4. Results summary
print("\n=== Solar Sail Trajectory Plan ===")
print(f"Target: {res['target']['name']}  (ID {res['target']['id']})")
print(f"Miss distance: {res['miss_km']:.6e} km  | Meets 100 m tol: {res['meets_100m_tol']}")
print(f"V∞: {res['Vinf_km_s']:.6f} km/s")
print(f"Total flight time: {res['tf']/gtoc.DAY:.1f} days")

# 5. Convert controls to readable units
dts_days   = np.array(res["dts"]) / gtoc.DAY
theta0_deg = np.degrees(np.array(res["theta0s"]))
phi0_deg   = np.degrees(np.array(res["phi0s"]))
thetaf_deg = np.degrees(np.array(res["thetafs"]))
phif_deg   = np.degrees(np.array(res["phifs"]))

print("\nSegment controls (θ₀, φ₀ → θ_f, φ_f in degrees):")
print("Idx | Δt [days] | θ₀ [°] | φ₀ [°] | θ_f [°] | φ_f [°]")
for i, (dt, t0d, p0d, tfd, pfd) in enumerate(zip(dts_days, theta0_deg, phi0_deg, thetaf_deg, phif_deg), start=1):
    print(f"{i:02d}  | {dt:10.1f} | {t0d:7.2f} | {p0d:7.2f} | {tfd:7.2f} | {pfd:7.2f}")
