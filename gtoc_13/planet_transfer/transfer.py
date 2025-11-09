import numpy as np

import solar_sail_planner as gtoc

# 1) Load bodies (PlanetX is in this CSV)
bodies = gtoc.load_registry_from_csvs("../../content/gtoc13_planets.csv")   # your path

# Look up by name or ID
planetx = next(b for b in bodies.values() if b.name.lower()=="planetx")

# 2) spacecraft initial state (km, km/s) and start time t0 (s)
r0 = np.array([-200.0*gtoc.AU, 30.823783300933*gtoc.AU, -10.706755180845*gtoc.AU]) # km
v0 = np.array([31.381, 0.0, 0.0]) # km/s
t0 = 0.0 # s (0 <= t0 <= 200 years)

# 3) Plan: DE seed + Sims–Flanagan refine (increase pop/iters/steps for quality)
res = gtoc.plan_solar_sail_to_body(
    r0=r0, v0=v0, t0=t0, target_body=planetx,
    tf_days_range=(3000, 20000),
    N_segments=30, popsize=22, iters=50, rng_seed=0
)

print("miss (km) =", res["miss_km"], "  meets_100m_tol =", res["meets_100m_tol"])
print("Vinf (km/s)=", res["Vinf_km_s"], "  ToF (days) =", res["tf"]/gtoc.DAY)

# Convert controls to human-friendly units
dts_days  = np.array(res["dts"])/gtoc.DAY
alpha_deg = np.degrees(np.array(res["alphas"]))
sigma_deg = np.degrees(np.array(res["sigmas"]))
for i,(dt,a,s) in enumerate(zip(dts_days,alpha_deg,sigma_deg),1):
    print(f"{i:02d}: dt={dt:9.1f} d, alpha={a:7.2f}°, sigma={s:7.2f}°")
