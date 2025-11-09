import numpy as np
import matplotlib.pyplot as plt
import solar_sail_planner as gtoc

# ===========================================
# 1. Load celestial body registry
# ===========================================
bodies = gtoc.load_registry_from_csvs("GTOC_13/content/gtoc13_planets.csv")

# Look up PlanetX by name (case-insensitive)
planetx = next(b for b in bodies.values() if b.name.lower() == "planetx")

# ===========================================
# 2. Initial state (heliocentric, km and km/s)
# ===========================================
r0 = np.array([
    -200.0 * gtoc.AU,
     30.823783300933 * gtoc.AU,
    -10.706755180845 * gtoc.AU
])
v0 = np.array([31.381, 0.0, 0.0])  # km/s
t0 = 0.0  # seconds

# ===========================================
# 3. Plan trajectory (piecewise-constant sail orientation)
# ===========================================
res = gtoc.plan_solar_sail_to_body(
    r0=r0, v0=v0, t0=t0, target_body=planetx,
    tf_days_range=(3000, 20000),
    N_segments=10,
    popsize=22,
    iters=50,
    rng_seed=0
)

# ===========================================
# 4. Results summary
# ===========================================
print("\n=== Solar Sail Trajectory Plan ===")
print(f"Target: {res['target']['name']}  (ID {res['target']['id']})")
print(f"Miss distance: {res['miss_km']:.6e} km  | Meets 100 m tol: {res['meets_100m_tol']}")
print(f"V∞: {res['Vinf_km_s']:.6f} km/s")
print(f"Total flight time: {res['tf']/gtoc.DAY:.1f} days")

# Convert to readable units
dts_days = np.array(res["dts"]) / gtoc.DAY
theta_deg = np.degrees(np.array(res["thetas"]))
phi_deg   = np.degrees(np.array(res["phis"]))

print("\nSegment controls (θ, φ in degrees):")
print("Idx | Δt [days] | θ [°] | φ [°]")
for i, (dt, tdeg, pdeg) in enumerate(zip(dts_days, theta_deg, phi_deg), start=1):
    print(f"{i:02d}  | {dt:10.1f} | {tdeg:7.2f} | {pdeg:7.2f}")

# ===========================================
# 5. Trajectory visualization
# ===========================================
times = np.array(res["times"])
states = np.array(res["states"])
r = states[:, :3]

# Convert from km to AU for plotting
r_AU = r / gtoc.AU

# Get target ephemeris path (optional)
target_ephem = gtoc.ephem_from_body(planetx)
tf = res["tf"]
t_span = np.linspace(0, tf, 500)
r_target = np.array([target_ephem(t)[0] for t in t_span]) / gtoc.AU

# === 3D trajectory plot ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(r_AU[:,0], r_AU[:,1], r_AU[:,2],
        label='Sailcraft trajectory', lw=2.5)

ax.plot(r_target[:,0], r_target[:,1], r_target[:,2],
        'r--', lw=1.5, label=f'Target: {planetx.name}')

# Start / end markers
ax.scatter(r_AU[0,0], r_AU[0,1], r_AU[0,2], color='green', s=80, label='Start')
ax.scatter(r_AU[-1,0], r_AU[-1,1], r_AU[-1,2], color='red', s=80, label='End')
ax.scatter(0, 0, 0, color='orange', s=120, marker='*', label='Star')

ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_zlabel('z [AU]')
ax.set_title(f'Solar Sail Trajectory to {planetx.name}')
ax.legend()
ax.grid(True)
ax.view_init(elev=25, azim=45)
plt.tight_layout()
plt.show()

# === 2D projection (XY-plane) ===
plt.figure(figsize=(8, 8))
plt.plot(r_AU[:,0], r_AU[:,1], lw=2, label='Sailcraft')
plt.plot(r_target[:,0], r_target[:,1], 'r--', label=f'Target: {planetx.name}')
plt.scatter(0, 0, color='orange', s=120, marker='*', label='Star')
plt.scatter(r_AU[0,0], r_AU[0,1], color='green', s=80, label='Start')
plt.scatter(r_AU[-1,0], r_AU[-1,1], color='red', s=80, label='End')
plt.axis('equal')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title(f'Trajectory to {planetx.name} (XY plane)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
