# Solar sail planner:
# - Set arbitrary (r0, v0, t0)
# - Load PlanetX (and others) from CSV
# - Target body with piecewise-constant (alpha, sigma) control
# - DE seed + Sims–Flanagan multiple shooting refinement (SciPy-free)

import math, json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Callable, Optional, Iterable
import numpy as np
import pandas as pd

# =============================
# Constants
# =============================
AU = 149597870.691        # km
MU_STAR = 139348062043.343  # km^3/s^2 (Altaira)
DAY = 86400.0             # s
YEAR = 365.25 * DAY       # s

# Ideal sail params at 1 AU (Table values): a1 = 2*C*A/m [m/s^2] @ 1 AU.
# Convert to km/s^2 and fold into K for inverse-square model a = -(K/r^2)*(n·rhat)^2 * n.
C_1AU = 5.4026e-6         # N/m^2
SAIL_AREA = 15000.0       # m^2
SC_MASS = 500.0           # kg
A1_AU_KM = (2.0 * C_1AU * SAIL_AREA / SC_MASS) / 1000.0  # km/s^2 @ 1 AU, alpha=0
K_SAIL = A1_AU_KM * (AU**2)

# =============================
# Utilities
# =============================
def norm(v): return float(np.linalg.norm(v))
def unit(v):
    n = norm(v)
    return v if n==0 else v/n

def d2r(x): return float(np.deg2rad(x))

# =============================
# Ephemerides from Keplerian elements
# =============================
@dataclass
class Body:
    id: int
    name: str
    a: float
    e: float
    inc: float
    raan: float
    argp: float
    M0: float
    weight: float
    mu: float
    R: Optional[float] = None

def kepler_E_from_M(M, e, tol=1e-12, itmax=50):
    """Solve M = E - e*sinE (elliptic)."""
    M = (M + np.pi) % (2*np.pi) - np.pi
    if e < 1e-12:
        return M
    E = M if e < 0.8 else np.pi
    for _ in range(itmax):
        f  = E - e*np.sin(E) - M
        fp = 1.0 - e*np.cos(E)
        dE = -f/fp
        E += dE
        if abs(dE) < tol:
            break
    return E

def state_from_elements(body: Body, t: float, mu_central: float = MU_STAR):
    a, e, i, Om, w, M0 = body.a, body.e, body.inc, body.raan, body.argp, body.M0
    n = math.sqrt(mu_central / (a**3))
    M = M0 + n * t
    E = kepler_E_from_M(M, e)
    cosE, sinE = math.cos(E), math.sin(E)
    beta = math.sqrt(1 - e*e)
    cosf = (cosE - e) / (1 - e*cosE)
    sinf = (beta * sinE) / (1 - e*cosE)
    f = math.atan2(sinf, cosf)
    r = a * (1 - e*cosE)

    # Perifocal r,v
    x_p = r * math.cos(f)
    y_p = r * math.sin(f)
    vfac = math.sqrt(mu_central * a) / r
    vx_p = -vfac * math.sin(E)
    vy_p =  vfac * beta * math.cos(E)

    # Rotation PQW->IJK
    cO, sO = math.cos(Om), math.sin(Om)
    ci, si = math.cos(i), math.sin(i)
    cw, sw = math.cos(w), math.sin(w)
    R11 =  cO*cw - sO*sw*ci; R12 = -cO*sw - sO*cw*ci; R13 =  sO*si
    R21 =  sO*cw + cO*sw*ci; R22 = -sO*sw + cO*cw*ci; R23 = -cO*si
    R31 =  sw*si;             R32 =  cw*si;             R33 =  ci

    rvec = np.array([R11*x_p + R12*y_p, R21*x_p + R22*y_p, R31*x_p + R32*y_p])
    vvec = np.array([R11*vx_p + R12*vy_p, R21*vx_p + R22*vy_p, R31*vx_p + R32*vy_p])
    return rvec, vvec

def ephem_from_body(body: Body):
    return lambda t: state_from_elements(body, t)

def load_registry_from_csvs(planets_csv, asteroids_csv=None, comets_csv=None) -> Dict[int, Body]:
    bodies: Dict[int, Body] = {}
    planets = pd.read_csv(planets_csv, encoding_errors='ignore')
    for _, r in planets.iterrows():
        bodies[int(r['#Planet ID'])] = Body(
            id=int(r['#Planet ID']),
            name=str(r['Name']),
            a=float(r['Semi-Major Axis (km)']),
            e=float(r['Eccentricity ()']),
            inc=d2r(r['Inclination (deg)']),
            raan=d2r(r['Longitude of the Ascending Node (deg)']),
            argp=d2r(r['Argument of Periapsis (deg)']),
            M0=d2r(r['Mean Anomaly at t=0 (deg)']),
            weight=float(r['Weight ()']),
            mu=float(r['GM (km3/s2)']),
            R=float(r['Radius (km)']) if 'Radius (km)' in r else None
        )
    if asteroids_csv:
        ast = pd.read_csv(asteroids_csv, encoding_errors='ignore')
        for _, r in ast.iterrows():
            bodies[int(r['#Asteroid ID'])] = Body(
                id=int(r['#Asteroid ID']),
                name=str(r['Name']),
                a=float(r['Semi-Major Axis (km)']),
                e=float(r['Eccentricity ()']),
                inc=d2r(r['Inclination (deg)']),
                raan=d2r(r['Longitude of the Ascending Node (deg)']),
                argp=d2r(r['Argument of Periapsis (deg)']),
                M0=d2r(r['Mean Anomaly at t=0 (deg)']),
                weight=float(r['Weight ()']),
                mu=0.0,
                R=None
            )
    if comets_csv:
        com = pd.read_csv(comets_csv, encoding_errors='ignore')
        for _, r in com.iterrows():
            bodies[int(r['#Comet ID'])] = Body(
                id=int(r['#Comet ID']),
                name=str(r['Name']),
                a=float(r['Semi-Major Axis (km)']),
                e=float(r['Eccentricity ()']),
                inc=d2r(r['Inclination (deg)']),
                raan=d2r(r['Longitude of the Ascending Node (deg)']),
                argp=d2r(r['Argument of Periapsis (deg)']),
                M0=d2r(r['Mean Anomaly at t=0 (deg)']),
                weight=float(r['Weight ()']),
                mu=0.0,
                R=None
            )
    return bodies

# =============================
# Sail dynamics
# =============================
# TODO: check vector definitions
def rtn_frame(r, v):
    rhat = unit(r)
    h = np.cross(r, v); hhat = unit(h)
    that = unit(np.cross(hhat, rhat))
    return rhat, that, hhat

def n_from_alpha_sigma(r, v, alpha, sigma):
    rhat, that, hhat = rtn_frame(r, v)
    return (np.cos(alpha)*rhat +
            np.sin(alpha)*(np.cos(sigma)*that + np.sin(sigma)*hhat))

def sail_accel(r, v, alpha, sigma, K=K_SAIL):
    # Ideal sail: a = -(K/r^2)*(n·rhat)^2 * n (km/s^2)
    n = n_from_alpha_sigma(r, v, alpha, sigma)
    rhat = unit(r); r2 = float(np.dot(r, r))
    return -K * (1.0/r2) * (float(np.dot(n, rhat))**2) * n

def rhs_sail(t, X, alpha, sigma):
    r = X[:3]; v = X[3:]
    ar = -MU_STAR * r / (norm(r)**3)
    asail = sail_accel(r, v, alpha, sigma, K_SAIL)
    return np.hstack([v, ar + asail])

# =============================
# RK4 propagation
# =============================
# TODO: match this with base.ipynb
def rk4_step(fun, t, X, dt, *args):
    k1 = fun(t, X, *args)
    k2 = fun(t+0.5*dt, X+0.5*dt*k1, *args)
    k3 = fun(t+0.5*dt, X+0.5*dt*k2, *args)
    k4 = fun(t+dt,     X+dt*k3, *args)
    return X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def propagate_segment(X0, t0, dt, alpha, sigma, nsub=30):
    h = dt / max(1, nsub)
    X = X0.copy(); t = t0
    for _ in range(max(1, nsub)):
        X = rk4_step(rhs_sail, t, X, h, alpha, sigma)
        t += h
    return X, t

def integrate_with_profile(r0, v0, t0, dts, alphas, sigmas, nsub=40):
    X = np.hstack([r0, v0]); t = t0
    times = [t]; states = [X.copy()]
    for k in range(len(dts)):
        a = float(np.clip(alphas[k], 0.0, 0.5*np.pi))
        s = float(np.mod(sigmas[k], 2*np.pi))
        X, t = propagate_segment(X, t, max(1.0, dts[k]), a, s, nsub=nsub)
        times.append(t); states.append(X.copy())
    return np.array(times), np.array(states)

# =============================
# Differential Evolution seed
# =============================
def pack_vec(dts, alphas, sigmas): return np.concatenate([dts, alphas, sigmas])
def unpack_vec(x, N):
    dts = x[:N]; alphas = x[N:2*N]; sigmas = x[2*N:3*N]
    return dts, alphas, sigmas

def objective_miss(x, r0, v0, t0, target_ephem, N, nsub=20):
    dts, alphas, sigmas = unpack_vec(x, N)
    dts = np.maximum(dts, 1.0)
    alphas = np.clip(alphas, 0.0, 0.5*np.pi)
    sigmas = np.mod(sigmas, 2*np.pi)
    X = np.hstack([r0, v0]); t = t0
    for k in range(N):
        X, t = propagate_segment(X, t, dts[k], alphas[k], sigmas[k], nsub=nsub)
    r2, v2 = target_ephem(t)
    return norm(X[:3]-r2)

def differential_evolution(r0, v0, t0, tf_range, target_ephem,
                           N=8, popsize=16, iters=40, F=0.7, CR=0.9, seed=0):
    rng = np.random.default_rng(seed)
    T_min = max(5*DAY, tf_range[0]-t0)
    T_max = max(T_min+1.0, tf_range[1]-t0)
    pop = []
    for _ in range(popsize):
        T = rng.uniform(T_min, T_max)
        raw = rng.uniform(0.0, 1.0, size=N); dts = (raw/np.sum(raw))*T
        alphas = rng.uniform(0.0, 0.5*np.pi, size=N)
        sigmas = rng.uniform(0.0, 2*np.pi, size=N)
        pop.append(pack_vec(dts, alphas, sigmas))
    pop = np.array(pop)
    fit = np.array([objective_miss(ind, r0, v0, t0, target_ephem, N, nsub=12) for ind in pop])

    for _ in range(iters):
        for i in range(popsize):
            idxs = [j for j in range(popsize) if j!=i]
            a, b, c = pop[rng.choice(idxs, 3, replace=False)]
            mutant = a + F*(b - c)
            cross = rng.uniform(0,1,mutant.shape) < CR
            jrand = rng.integers(0, len(mutant))
            trial = np.where(cross | (np.arange(len(mutant))==jrand), mutant, pop[i])
            dts, alphas, sigmas = unpack_vec(trial, N)
            dts = np.abs(dts); dts = (dts/np.sum(dts))*rng.uniform(T_min, T_max)
            alphas = np.clip(alphas, 0.0, 0.5*np.pi)
            sigmas = np.mod(sigmas, 2*np.pi)
            trial = pack_vec(dts, alphas, sigmas)
            f = objective_miss(trial, r0, v0, t0, target_ephem, N, nsub=12)
            if f < fit[i]:
                pop[i] = trial; fit[i] = f
    j = int(np.argmin(fit))
    dts, alphas, sigmas = unpack_vec(pop[j], N)
    tf = t0 + np.sum(dts)
    return dict(dts=dts, alphas=alphas, sigmas=sigmas, tf=tf, miss=float(fit[j]))

# =============================
# Sims–Flanagan (penalty multiple shooting)
# =============================
def sims_flanagan_refine(r0, v0, t0, target_ephem, seed, nsub_prop=40, steps=220, step_scale=0.05, rng_seed=0):
    dts = seed['dts'].copy(); alphas = seed['alphas'].copy(); sigmas = seed['sigmas'].copy()
    N = len(dts)
    times, states = integrate_with_profile(r0, v0, t0, dts, alphas, sigmas, nsub=nsub_prop)
    Xnodes = states[1:-1].copy()

    def defects(dts, alphas, sigmas, Xnodes):
        cons = []
        t = t0
        nodes = [np.hstack([r0, v0])] + [Xnodes[k] for k in range(N-1)]
        for k in range(N):
            Xk = nodes[k]
            a, s = float(np.clip(alphas[k],0,0.5*np.pi)), float(np.mod(sigmas[k],2*np.pi))
            Xp, t = propagate_segment(Xk, t, max(1.0, dts[k]), a, s, nsub=nsub_prop)
            if k < N-1:
                cons.append(Xp - nodes[k+1])
        return np.concatenate(cons) if len(cons)>0 else np.zeros(0)

    def score(dts, alphas, sigmas, Xnodes):
        t = t0
        nodes = [np.hstack([r0, v0])] + [Xnodes[k] for k in range(N-1)]
        X = nodes[0].copy()
        for k in range(N):
            X = nodes[k].copy()
            a, s = float(np.clip(alphas[k],0,0.5*np.pi)), float(np.mod(sigmas[k],2*np.pi))
            X, t = propagate_segment(X, t, max(1.0, dts[k]), a, s, nsub=nsub_prop)
        rf, vf = X[:3], X[3:]
        r2, v2 = target_ephem(t)
        miss = norm(rf - r2)
        Vinf = norm(vf - v2)
        F = 0.2 + math.exp(-Vinf/13.0)/(1.0 + math.exp(-5.0*(Vinf-1.5)))
        J = miss - 1e3*F
        D = defects(dts, alphas, sigmas, Xnodes)
        J += 1e5 * np.sum(np.abs(D))
        return J, miss, Vinf, t

    best = (dts.copy(), alphas.copy(), sigmas.copy(), Xnodes.copy())
    bestJ, bestMiss, bestVinf, best_t = score(*best)
    rng = np.random.default_rng(rng_seed)
    scale_d = np.maximum(1.0, np.abs(dts))*step_scale
    scale_a = 0.1*step_scale
    scale_s = 0.2*step_scale
    for _ in range(steps):
        cand = (best[0] + rng.normal(scale=scale_d),
                best[1] + rng.normal(scale=scale_a, size=N),
                best[2] + rng.normal(scale=scale_s, size=N),
                best[3] + rng.normal(scale=0.01, size=best[3].shape))
        J, miss, Vinf, _ = score(*cand)
        if J < bestJ:
            best, bestJ, bestMiss, bestVinf = cand, J, miss, Vinf
            scale_d *= 0.99

    dts, alphas, sigmas, Xnodes = best
    times, states = integrate_with_profile(r0, v0, t0, dts, alphas, sigmas, nsub=nsub_prop)
    return dict(dts=dts, alphas=alphas, sigmas=sigmas, tf=times[-1],
                times=times, states=states, miss=bestMiss, Vinf=bestVinf)

# =============================
# Verification and high-level API
# =============================
def verify_to_target(r0, v0, t0, profile, target_ephem, nsub_hi=180):
    times, states = integrate_with_profile(r0, v0, t0, profile['dts'], profile['alphas'], profile['sigmas'], nsub=nsub_hi)
    rf, vf = states[-1,:3], states[-1,3:]
    r2, v2 = target_ephem(times[-1])
    return dict(times=times, states=states, miss=norm(rf-r2), Vinf=norm(vf-v2))

def plan_solar_sail_to_body(r0, v0, t0, target_body,
                            tf_days_range=(100.0, 2000.0),
                            N_segments: int = 10,
                            popsize: int = 22,
                            iters: int = 50,
                            rng_seed: int = 0):
    r0 = np.array(r0, dtype=float); v0 = np.array(v0, dtype=float)
    target_ephem = ephem_from_body(target_body)
    tf_range = (tf_days_range[0]*DAY + t0, tf_days_range[1]*DAY + t0)
    seed = differential_evolution(r0, v0, t0, tf_range, target_ephem,
                                  N=N_segments, popsize=popsize, iters=iters, seed=rng_seed)
    profile = sims_flanagan_refine(r0, v0, t0, target_ephem, seed,
                                   nsub_prop=40, steps=220, step_scale=0.05, rng_seed=rng_seed)
    verify = verify_to_target(r0, v0, t0, profile, target_ephem, nsub_hi=180)
    out = {
        "target": dict(id=target_body.id, name=target_body.name),
        "t0": t0,
        "dts": profile["dts"].tolist(),
        "alphas": profile["alphas"].tolist(),
        "sigmas": profile["sigmas"].tolist(),
        "tf": float(profile["tf"]),
        "miss_km": float(verify["miss"]),
        "Vinf_km_s": float(verify["Vinf"]),
        "times": verify["times"].tolist(),
        "states": verify["states"].tolist(),
        "AU_km": AU,
        "MU_star": MU_STAR,
        "A1_AU_km_s2": A1_AU_KM,
        "meets_100m_tol": float(verify["miss"]) <= 1e-4  # 100 m
    }
    return out

# -----------------------------
# example usage
# -----------------------------
# if __name__ == "__main__":
#     bodies = load_registry_from_csvs("gtoc13_planets.csv")
#     planetx = next(b for b in bodies.values() if b.name.lower()=="planetx")
#     r0 = np.array([-200.0*AU, 0.0, 0.0])
#     v0 = np.array([25.0, 0.0, 0.0])
#     res = plan_solar_sail_to_body(r0, v0, 0.0, planetx,
#                                   tf_days_range=(5000, 20000),
#                                   N_segments=10, popsize=22, iters=50, rng_seed=0)
#     print(json.dumps({k:v for k,v in res.items() if k in ("miss_km","meets_100m_tol","Vinf_km_s","tf")}, indent=2))
