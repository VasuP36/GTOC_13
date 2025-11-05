import numpy as np
import pandas as pd
import math, json, os
from dataclasses import dataclass
from scipy.optimize import differential_evolution

# Load CSVs
planets = pd.read_csv("gtoc13_planets.csv")
asteroids = pd.read_csv("gtoc13_asteroids.csv")
comets = pd.read_csv("gtoc13_comets.csv")

MU_ALTAIRA = 139348062043.343  # km^3/s^2

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
    mu: float = 0.0  # body GM

def d2r(x):
    return float(x)*math.pi/180.0

# Unified registry with weights: planets=from csv, asteroids=1, comets=3
bodies = {}
for _, r in planets.iterrows():
    bodies[int(r['#Planet ID'])] = Body(
        id=int(r['#Planet ID']), name=str(r['Name']),
        a=float(r['Semi-Major Axis (km)']),
        e=float(r['Eccentricity ()']),
        inc=d2r(r['Inclination (deg)']),
        raan=d2r(r['Longitude of the Ascending Node (deg)']),
        argp=d2r(r['Argument of Periapsis (deg)']),
        M0=d2r(r['Mean Anomaly at t=0 (deg)']),
        weight=float(r['Weight ()']),
        mu=float(r['GM (km3/s2)'])
    )
for _, r in asteroids.iterrows():
    bodies[int(r['#Asteroid ID'])] = Body(
        id=int(r['#Asteroid ID']), name=f"Ast{int(r['#Asteroid ID'])}",
        a=float(r['Semi-Major Axis (km)']),
        e=float(r['Eccentricity ()']),
        inc=d2r(r['Inclination (deg)']),
        raan=d2r(r['Longitude of the Ascending Node (deg)']),
        argp=d2r(r['Argument of Periapsis (deg)']),
        M0=d2r(r['Mean Anomaly at t=0']),
        weight=1.0
    )
for _, r in comets.iterrows():
    bodies[int(r['# Comet ID'])] = Body(
        id=int(r['# Comet ID']), name=f"Com{int(r['# Comet ID'])}",
        a=float(r['Semi-Major Axis (km)']),
        e=float(r['Eccentricity ()']),
        inc=d2r(r['Inclination (deg)']),
        raan=d2r(r['Longitude of the Ascending Node (deg)']),
        argp=d2r(r['Argument of Periapsis (deg)']),
        M0=d2r(r['Mean Anomaly at t=0 (deg)']),
        weight=3.0
    )

# Kepler's equation
def kepler_E_from_M(M, e, tol=1e-12, itmax=60):
    E = M if e < 0.8 else math.pi
    for _ in range(itmax):
        f = E - e*math.sin(E) - M
        fp = 1 - e*math.cos(E)
        d = -f/fp
        E += d
        if abs(d) < tol: break
    return E

def state_from_kepler(body: Body, t_days: float, mu_c=MU_ALTAIRA):
    a, e = body.a, body.e
    n = math.sqrt(mu_c/(a**3))
    M = (body.M0 + n*(t_days*86400.0))%(2*math.pi)
    E = kepler_E_from_M(M, e)
    r_pf = np.array([a*(math.cos(E)-e), a*math.sqrt(1-e**2)*math.sin(E), 0.0])
    v_pf = np.array([-math.sin(E), math.sqrt(1-e**2)*math.cos(E), 0.0]) * (n*a)/(1 - e*math.cos(E))
    cO,sO = math.cos(body.raan), math.sin(body.raan)
    ci,si = math.cos(body.inc), math.sin(body.inc)
    cw,sw = math.cos(body.argp), math.sin(body.argp)
    RzO = np.array([[cO,-sO,0],[sO,cO,0],[0,0,1]])
    Rxi = np.array([[1,0,0],[0,ci,-si],[0,si,ci]])
    Rzw = np.array([[cw,-sw,0],[sw,cw,0],[0,0,1]])
    Q = RzO @ Rxi @ Rzw
    return Q@r_pf, Q@v_pf

def F_vinf(vinf):
    """
    F(V∞) = 0.2 + exp(-V∞/13) / (1 + exp(-5(V∞-1.5)))
    """
    return 0.2 + math.exp(-vinf/13.0)/(1.0 + math.exp(-5.0*(vinf-1.5)))

def beam_search(start_body_id=10, max_legs=6, beam_width=20, mission_days=200*365.25,
                candidates_per_step=120, tof_bounds=(150, 3000)):
    """
    Beam Search: Lambert transfer proxy for speed
    """
    ids = [bid for bid in bodies if bid != start_body_id]
    ids.sort(key=lambda i: bodies[i].weight, reverse=True)
    candidates = ids[:max(100, candidates_per_step)]  # speed guard

    root = {"seq":[start_body_id], "times":[0.0], "score":0.0}
    beam = [root]
    for _ in range(max_legs):
        new_nodes = []
        for node in beam:
            t_curr = node["times"][-1]
            for nb in candidates:
                if nb in node["seq"]:  # skip repeats for initial pass
                    continue
                # sample a small grid of TOFs
                for tof_d in np.linspace(tof_bounds[0], tof_bounds[1], 8):
                    if t_curr + tof_d > mission_days: 
                        continue
                    r1,_ = state_from_kepler(bodies[node["seq"][-1]], t_curr/86400.0)
                    r2,v2 = state_from_kepler(bodies[nb], (t_curr+tof_d)/86400.0)
                    # velocity that “connects” positions in dt -- proxy for v_infinity at arrival
                    v_est = (r2 - r1)/(tof_d*86400.0)
                    vinf_arr = np.linalg.norm(v_est - v2)
                    score_inc = bodies[nb].weight * F_vinf(vinf_arr)
                    new_nodes.append({
                        "seq": node["seq"]+[nb],
                        "times": node["times"]+[t_curr+tof_d],
                        "score": node["score"] + score_inc
                    })
        new_nodes.sort(key=lambda n: n["score"], reverse=True)
        beam = new_nodes[:beam_width] if new_nodes else beam
        if not new_nodes: break
    return beam

def refine_sequence_with_de(seq, mission_days=200*365.25):
    """
    Genetic Algorithm-based refinement of TOFs for top sequences
    """
    n = len(seq)-1
    def score_from_tofs(x):
        t = 0.0
        score = 0.0
        for k in range(n):
            tof = x[k]
            if tof <= 1 or t + tof > mission_days:
                return -1e9
            r1,_ = state_from_kepler(bodies[seq[k]], t/86400.0)
            r2,v2 = state_from_kepler(bodies[seq[k+1]], (t+tof)/86400.0)
            v_est = (r2 - r1)/((tof)*86400.0)
            vinf = np.linalg.norm(v_est - v2)
            score += bodies[seq[k+1]].weight * F_vinf(vinf)
            t += tof
        return score
    bounds = [(50, 3000)]*n
    res = differential_evolution(lambda x: -score_from_tofs(x), bounds=bounds,
                                 maxiter=60, popsize=12, tol=1e-2, polish=False, seed=3)
    return dict(sequence=seq, best_score_proxy=float(-res.fun), best_tofs_days=res.x.tolist())


# Search Execution
beam = beam_search(start_body_id=10, max_legs=6, beam_width=20, candidates_per_step=120, tof_bounds=(150, 3000))
top_sequences = [b["seq"] for b in beam[:5]]
ga_refined = [refine_sequence_with_de(seq) for seq in top_sequences]

os.makedirs("beam_ga_baseline_outputs", exist_ok=True)
with open("beam_ga_baseline_outputs/results.json","w") as f:
    json.dump({
        "beam_top": [{"sequence":b["seq"], "times_days":b["times"], "score_proxy":b["score"]} for b in beam],
        "ga_refined": ga_refined
    }, f, indent=2)

print("Saved:", "beam_ga_baseline_outputs/results.json")
