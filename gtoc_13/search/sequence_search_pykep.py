"""
Beam search (global) + GA refinement (local) for long tours
    - Uses all bodies from 3 CSVs (planets, asteroids, comets)
    - Allows re-visits
    - Supports multi-revolution Lambert (PyKEP if available; proxy fallback)
    - Parallel GA refinement (SciPy DE with workers)
"""

import math, json, os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# Optional PyKEP import (to use robust Lambert). Fallback uses a fast proxy
try:
    import pykep as pk
    HAVE_PYKEP = True
    print("Have pykep!")
except Exception:
    HAVE_PYKEP = False
    print("Don't have pykep")

# Constants
DAY = 86400.0
YEAR = 365.25 * DAY
MU_STAR = 139348062043.343  # km^3/s^2

# Bodies
@dataclass
class Body:
    id: int
    name: str
    a: float         # km
    e: float
    inc: float       # rad
    raan: float      # rad
    argp: float      # rad
    M0: float        # rad at t=0
    weight: float    # scientific importance
    mu: float = 0.0  # km^3/s^2 (body GM)

def d2r(x):
    return float(x) * math.pi / 180.0

def load_registry_from_csvs(planets_csv, asteroids_csv, comets_csv) -> Dict[int, Body]:
    planets = pd.read_csv(planets_csv, encoding_errors='ignore')
    asteroids = pd.read_csv(asteroids_csv, encoding_errors='ignore')
    comets = pd.read_csv(comets_csv, encoding_errors='ignore')
    bodies: Dict[int, Body] = {}

    # Planets: weight from CSV
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
        )

    # Asteroids: weight = 1
    for _, r in asteroids.iterrows():
        bodies[int(r['#Asteroid ID'])] = Body(
            id=int(r['#Asteroid ID']),
            name=f"Ast{int(r['#Asteroid ID'])}",
            a=float(r['Semi-Major Axis (km)']),
            e=float(r['Eccentricity ()']),
            inc=d2r(r['Inclination (deg)']),
            raan=d2r(r['Longitude of the Ascending Node (deg)']),
            argp=d2r(r['Argument of Periapsis (deg)']),
            M0=d2r(r['Mean Anomaly at t=0']),
            weight=1.0,
        )

    # Comets: weight = 3
    for _, r in comets.iterrows():
        bodies[int(r['# Comet ID'])] = Body(
            id=int(r['# Comet ID']),
            name=f"Com{int(r['# Comet ID'])}",
            a=float(r['Semi-Major Axis (km)']),
            e=float(r['Eccentricity ()']),
            inc=d2r(r['Inclination (deg)']),
            raan=d2r(r['Longitude of the Ascending Node (deg)']),
            argp=d2r(r['Argument of Periapsis (deg)']),
            M0=d2r(r['Mean Anomaly at t=0 (deg)']),
            weight=3.0,
        )

    return bodies

# Kepler's equation
def kepler_E_from_M(M, e, tol=1e-12, itmax=60):
    E = M if e < 0.8 else math.pi
    for _ in range(itmax):
        f = E - e*math.sin(E) - M
        fp = 1 - e*math.cos(E)
        d = -f/fp
        E += d
        if abs(d) < tol:
            break
    return E

def state_from_kepler(body: Body, t_days: float, mu_c: float = MU_STAR) -> Tuple[np.ndarray, np.ndarray]:
    a, e = body.a, body.e
    n = math.sqrt(mu_c / (a**3))
    M = (body.M0 + n*(t_days*DAY)) % (2*math.pi)
    E = kepler_E_from_M(M, e)
    r_pf = np.array([a*(math.cos(E)-e), a*math.sqrt(1-e**2)*math.sin(E), 0.0])
    v_pf = np.array([-math.sin(E), math.sqrt(1-e**2)*math.cos(E), 0.0]) * (n*a) / (1 - e*math.cos(E))
    cO, sO = math.cos(body.raan), math.sin(body.raan)
    ci, si = math.cos(body.inc), math.sin(body.inc)
    cw, sw = math.cos(body.argp), math.sin(body.argp)
    RzO = np.array([[cO, -sO, 0],[sO, cO, 0],[0,0,1]])
    Rxi = np.array([[1,0,0],[0,ci,-si],[0,si,ci]])
    Rzw = np.array([[cw,-sw,0],[sw,cw,0],[0,0,1]])
    Q = RzO @ Rxi @ Rzw
    return Q @ r_pf, Q @ v_pf

# Lambert wrappers
def lambert_v1v2_pykep(r1_km, r2_km, dt_s, mu_km3s2=MU_STAR, max_revs=2) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Return (v1,v2) in km/s using PyKEP, scanning 0..max_revs and picking lowest |v1|+|v2|.
    """
    if not HAVE_PYKEP:
        return None
    try:
        r1 = np.array(r1_km)*1e3
        r2 = np.array(r2_km)*1e3
        mu = mu_km3s2*1e9  # km^3/s^2 -> m^3/s^2
        best = None
        best_cost = 1e300
        for m in range(max_revs+1):
            try:
                lp = pk.lambert_problem(r1, r2, dt_s, mu, m)
            except Exception:
                continue
            for branch in range(lp.get_Nmax()):
                v1 = np.array(lp.get_v1()[branch]) / 1e3  # m/s -> km/s
                v2 = np.array(lp.get_v2()[branch]) / 1e3
                cost = np.linalg.norm(v1) + np.linalg.norm(v2)
                if cost < best_cost:
                    best = (v1, v2)
                    best_cost = cost
        return best
    except Exception:
        return None

def lambert_proxy_v1v2(r1_km, r2_km, dt_s) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Very fast proxy (not true Lambert): straight-line average velocity.
    """
    if dt_s <= 0:
        return None
    v = (np.array(r2_km) - np.array(r1_km)) / dt_s
    return v, v

def lambert_solve_v1v2(r1_km, r2_km, dt_s, max_revs=2) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Try PyKEP lambert, fall back to proxy.
    """
    if dt_s <= 0:
        return None
    if HAVE_PYKEP:
        pair = lambert_v1v2_pykep(r1_km, r2_km, dt_s, MU_STAR, max_revs=max_revs)
        if pair is not None:
            return pair
    return lambert_proxy_v1v2(r1_km, r2_km, dt_s)

# Scoring
def F_vinf(vinf_kms: float) -> float:
    # GTOC13 F(V∞)
    return 0.2 + math.exp(-vinf_kms/13.0) / (1.0 + math.exp(-5.0*(vinf_kms - 1.5)))

def body_rhat(body: Body, t_days: float) -> np.ndarray:
    r, _ = state_from_kepler(body, t_days)
    n = np.linalg.norm(r)
    return r / n if n > 0 else r

def seasonal_S(rhat_i: np.ndarray, history: List[np.ndarray]) -> float:
    """
    Implements S(r_hat_ki) with degrees-based acos and the running history
    of previous science flybys of the same body.
    """
    if not history:
        # first flyby of this body
        return 1.0
    
    def acosd(x):
        return math.degrees(math.acos(max(-1.0, min(1.0, float(x)))))
    acc = 0.0
    for rhat_j in history:
        theta = acosd(np.dot(rhat_i, rhat_j))
        acc += math.exp(-(theta**2) / 50.0)
    return 0.1 + 0.9 / (1.0 + 10.0 * acc)


def leg_score(next_body: Body, vinf_arrival_kms: float) -> float:
    return next_body.weight * F_vinf(vinf_arrival_kms)

# ---------- Beam search ----------
def beam_search(
    bodies: Dict[int, Body],
    start_body_id: int,
    max_legs: int = 25,
    beam_width: int = 500,
    mission_days: float = 200*365.25,
    tof_bounds_days: Tuple[float, float] = (80.0, 4000.0),
    tof_grid: int = 24,
    max_revs: int = 2,
) -> List[Dict]:
    """Returns top partial/full paths: each is {seq, times_days, score}."""
    ids = sorted(bodies.keys())
    if start_body_id not in ids:
        raise ValueError("start_body_id not in bodies")

    # Sort candidate pool by weight (heavier first) but keep all
    candidates = [bid for bid in ids if bid != start_body_id]
    candidates.sort(key=lambda i: bodies[i].weight, reverse=True)

    # root = dict(seq=[start_body_id], times_days=[0.0], score=0.0)
    root = dict(seq=[start_body_id], times_days=[0.0], score=0.0, seasonals={})

    beam = [root]
    tgrid = np.linspace(tof_bounds_days[0], tof_bounds_days[1], max(3, tof_grid))

    for depth in range(max_legs):
        new_nodes: List[Dict] = []
        for node in beam:
            last_id = node["seq"][-1]
            t_curr = node["times_days"][-1]
            r1, v1_body = state_from_kepler(bodies[last_id], t_curr/1.0)  # days -> days
            for nb in candidates:
                # Re-visits allowed by design
                for tof_d in tgrid:
                    if t_curr + tof_d > mission_days:
                        continue
                    r2, v2_body = state_from_kepler(bodies[nb], (t_curr+tof_d))
                    pair = lambert_solve_v1v2(r1, r2, tof_d*DAY, max_revs=max_revs)
                    if pair is None:
                        continue
                    v1_lam, v2_lam = pair
                    vinf_arr = float(np.linalg.norm(v2_lam - v2_body))

                    # seasonal factor using per-body history
                    rhat_i = (r2 / np.linalg.norm(r2))
                    hist = node["seasonals"].get(nb, [])
                    S = seasonal_S(rhat_i, hist)

                    # score_inc = leg_score(bodies[nb], vinf_arr)
                    score_inc = bodies[nb].weight * S * F_vinf(vinf_arr)

                    new_hist = {k: v[:] for k, v in node["seasonals"].items()}
                    new_hist.setdefault(nb, []).append(rhat_i)

                    # new_nodes.append(dict(
                    #     seq = node["seq"] + [nb],
                    #     times_days = node["times_days"] + [t_curr + tof_d],
                    #     score = node["score"] + score_inc
                    # ))

                    new_nodes.append(dict(
                        seq = node["seq"] + [nb],
                        times_days = node["times_days"] + [t_curr + tof_d],
                        score = node["score"] + score_inc,
                        seasonals = new_hist,   # <- carry forward
                    ))

        if not new_nodes:
            break
        new_nodes.sort(key=lambda n: n["score"], reverse=True)
        beam = new_nodes[:beam_width]
    return beam

# GA refinement for top sequences
def _score_from_tofs(x, seq, bodies, mission_days, max_revs):
    n = len(seq) - 1
    t = 0.0
    total = 0.0
    # per-body list of previous unit directions for science flybys
    seasonals: Dict[int, List[np.ndarray]] = {}

    for k in range(n):
        tof = float(x[k])
        if tof <= 1.0 or (t + tof) > mission_days:
            return -1e9

        r1, v1 = state_from_kepler(bodies[seq[k]], t)
        r2, v2 = state_from_kepler(bodies[seq[k+1]], t + tof)
        pair = lambert_solve_v1v2(r1, r2, tof*DAY, max_revs=max_revs)
        if pair is None:
            return -1e8

        v1_lam, v2_lam = pair
        vinf = float(np.linalg.norm(v2_lam - v2))

        # seasonal factor for the arrival body at this leg
        nb = seq[k+1]
        rhat_i = r2 / np.linalg.norm(r2)
        S = seasonal_S(rhat_i, seasonals.get(nb, []))

        total += bodies[nb].weight * S * F_vinf(vinf)

        # log this flyby as a science flyby for seasonality
        seasonals.setdefault(nb, []).append(rhat_i)
        t += tof

    return total


def _neg_score_from_tofs(x, seq, bodies, mission_days, max_revs):
    # SciPy minimizes; we want to maximize
    return -_score_from_tofs(x, seq, bodies, mission_days, max_revs)

def refine_sequence_with_de(
    seq: List[int],
    bodies: Dict[int, Body],
    mission_days: float = 200*365.25,
    max_revs: int = 2,
    bounds_days: Tuple[float, float] = (50.0, 4000.0),
    de_popsize: int = 50,
    de_maxiter: int = 400,
    workers: int = -1,
    seed: Optional[int] = 13,
) -> Dict:
    """
    Optimize per-leg TOFs to maximize objective
    """
    n = len(seq) - 1
    if n <= 0:
        return dict(sequence=seq, best_score_proxy=0.0, best_tofs_days=[])

    lo, hi = bounds_days
    bounds = [(lo, hi)] * n

    # commenting due to multiprocessing errors
    # def score_from_tofs(x):
    #     t = 0.0
    #     total = 0.0
    #     for k in range(n):
    #         tof = float(x[k])
    #         if tof <= 1.0 or (t + tof) > mission_days:
    #             return -1e9
    #         r1, v1 = state_from_kepler(bodies[seq[k]], t)
    #         r2, v2 = state_from_kepler(bodies[seq[k+1]], t + tof)
    #         pair = lambert_solve_v1v2(r1, r2, tof*DAY, max_revs=max_revs)
    #         if pair is None:
    #             return -1e8
    #         v1_lam, v2_lam = pair
    #         vinf = float(np.linalg.norm(v2_lam - v2))
    #         total += leg_score(bodies[seq[k+1]], vinf)
    #         t += tof
    #     return total

    # older one
    # res = differential_evolution(
    #     lambda x: -score_from_tofs(x),
    #     bounds=bounds,
    #     maxiter=de_maxiter,
    #     popsize=de_popsize,
    #     tol=1e-3,
    #     polish=False,
    #     seed=seed,
    #     workers=workers  # SciPy 1.11+: parallel
    # )
    # return dict(sequence=seq, best_score_proxy=float(-res.fun), best_tofs_days=[float(v) for v in res.x])

    res = differential_evolution(
        _neg_score_from_tofs,
        args=(seq, bodies, mission_days, max_revs),
        bounds=bounds,
        maxiter=de_maxiter,
        popsize=de_popsize,
        tol=1e-3,
        polish=False,
        seed=seed,
        workers=workers,
        updating='deferred',
    )

    return dict(
        sequence=seq,
        best_score_proxy=float(-res.fun),
        best_tofs_days=[float(v) for v in res.x],
    )

def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    return obj


# Search Execution
if __name__ == "__main__":
    # 1) Load your CSVs
    PLANETS = "./content/gtoc13_planets.csv"
    ASTEROIDS = "./content/gtoc13_asteroids.csv"
    COMETS = "./content/gtoc13_comets.csv"
    bodies = load_registry_from_csvs(PLANETS, ASTEROIDS, COMETS)

    # 2) Beam search (PlanetX start, 6 legs for a quick run; scale up to 20–50)
    start_body_id = 10
    beam = beam_search(
        bodies=bodies,
        start_body_id=start_body_id,
        max_legs=6,               # try 20–25+ for higher score
        beam_width=500,            # try 1000–2000 for deeper coverage
        mission_days=200*365.25,
        tof_bounds_days=(80, 4000),
        tof_grid=24,               # try 30–40 for denser sampling
        max_revs=2,                # 0..2 multi-rev scan
    )

    # 3) Refine top-N sequences with GA (parallel DE)
    topN = 10
    refined = []
    for node in beam[:topN]:
        seq = node["seq"]
        res = refine_sequence_with_de(
            seq=seq,
            bodies=bodies,
            mission_days=200*365.25,
            max_revs=2,
            bounds_days=(50, 4000),
            de_popsize=60,    # increase for quality
            de_maxiter=400,   # increase for quality
            workers=-1,       # parallel
            seed=None
        )
        refined.append(dict(
            sequence=res["sequence"],
            best_score_proxy=res["best_score_proxy"],
            best_tofs_days=res["best_tofs_days"],
            beam_score_seed=node["score"]
        ))

    safe_beam = make_json_safe(beam)

    os.makedirs("./beam_ga_pykep_outputs", exist_ok=True)
    with open("./beam_ga_pykep_outputs/beam_top.json","w") as f:
        json.dump(safe_beam, f, indent=2)
    
    with open("./beam_ga_pykep_outputs/ga_refined.json","w") as f:
        json.dump(refined, f, indent=2)

    print("Saved outputs to ./beam_ga_pykep_outputs/")
    if HAVE_PYKEP:
        print("PyKEP detected: used robust Lambert.")
    else:
        print("PyKEP not found: used fast proxy (OK for ranking; install PyKEP for best results).")
