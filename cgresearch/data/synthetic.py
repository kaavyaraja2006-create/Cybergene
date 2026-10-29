"""SYNTHETIC flow generator for smoke tests and unit tests ONLY.

Nothing produced from this generator may be reported as a UNSW-NB15 / real-traffic result.
Every FlowData it returns has info['synthetic']=True and the experiment manifest records it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .unsw import FlowData

SERVICES = ["http", "dns", "ssh", "ftp", "smtp", "-"]
PROTOS = ["tcp", "udp", "icmp"]
STATES = ["fin", "con", "int", "req"]
FAMILIES = ["dos", "recon", "exploit", "backdoor"]


def make_synthetic_flows(n: int = 6000, attack_frac: float = 0.25, seed: int = 0,
                         dup_frac: float = 0.0, drift: float = 0.0, separation: float = 1.0,
                         near_dup_frac: float = 0.0, drift_offset: float = 0.0, families=None, t0: float = 0.0) -> FlowData:
    """`separation` scales how far attacks sit from benign (0 = indistinguishable).
    `drift` shifts benign traffic mean over time (concept drift). `dup_frac` injects exact
    duplicates; `near_dup_frac` injects near duplicates (same endpoints, close in time)."""
    rng = np.random.default_rng(seed)
    n_att = int(n * attack_frac)
    n_ben = n - n_att
    y = np.r_[np.zeros(n_ben, int), np.ones(n_att, int)]
    fams = list(families) if families else FAMILIES
    fam = np.r_[np.array(["Normal"] * n_ben), rng.choice(fams, n_att) if n_att else np.array([], dtype=str)]
    y_idx = rng.permutation(n)
    y, fam = y[y_idx], fam[y_idx]
    t = np.sort(rng.uniform(0, 1, n))

    def lognorm(mu, sd, size):
        return np.exp(rng.normal(mu, sd, size))

    dur = lognorm(-1.0 + drift_offset + drift * t, 1.2, n)
    spkts = np.maximum(1, lognorm(2.0 + drift_offset + drift * t, 0.9, n)).round()
    dpkts = np.maximum(0, lognorm(1.8, 0.9, n)).round()
    sbytes = spkts * lognorm(5.5, 0.5, n)
    dbytes = dpkts * lognorm(6.0, 0.7, n)
    sttl = rng.choice([62, 254, 31], n, p=[.6, .3, .1]).astype(float)
    dsport = rng.choice([80, 443, 53, 22, 25, 8080, 3306], n, p=[.3, .3, .2, .05, .05, .05, .05]).astype(float)
    sport = rng.integers(1024, 65535, n).astype(float)
    proto = rng.choice(PROTOS, n, p=[.7, .25, .05])
    service = rng.choice(SERVICES, n)
    state = rng.choice(STATES, n)

    a = y == 1
    s = separation
    # attacks: family-specific shifts
    for f in FAMILIES:
        m = a & (fam == f)
        k = int(m.sum())
        if not k:
            continue
        if f == "dos":
            spkts[m] = np.maximum(1, lognorm(2.0 + 1.2 * s, 0.5, k)).round()
            dur[m] = lognorm(-2.0 - 0.8 * s, 0.6, k)
            dsport[m] = 80
        elif f == "recon":
            dsport[m] = rng.integers(1, 10000, k)
            sbytes[m] = spkts[m] * lognorm(4.5 - 0.6 * s, 0.4, k)
            proto[m] = "tcp"
        elif f == "exploit":
            dbytes[m] = dpkts[m] * lognorm(6.0 + 0.9 * s, 0.6, k)
            dsport[m] = rng.choice([80, 445, 21], k)
        else:
            dur[m] = lognorm(1.0 + 0.8 * s, 0.8, k)
            sttl[m] = 254
            dsport[m] = rng.choice([4444, 31337, 6667], k)
    rate = (spkts + dpkts) / np.maximum(dur, 1e-3)
    srcip = np.array([f"10.0.{rng.integers(0, 8)}.{i}" for i in rng.integers(1, 120, n)])
    dstip = np.array([f"10.1.{rng.integers(0, 4)}.{i}" for i in rng.integers(1, 40, n)])
    df = pd.DataFrame(dict(sport=sport, dsport=dsport, proto=proto, state=state, service=service,
                           dur=dur, sbytes=sbytes, dbytes=dbytes, sttl=sttl, spkts=spkts, dpkts=dpkts,
                           rate=rate, sload=sbytes * 8 / np.maximum(dur, 1e-3),
                           dload=dbytes * 8 / np.maximum(dur, 1e-3)))
    meta = pd.DataFrame(dict(srcip=srcip, dstip=dstip, sport=sport, dsport=dsport, stime=(t0 + t) * 86400.0))
    if dup_frac > 0:
        k = int(n * dup_frac)
        src = rng.integers(0, n, k)
        df = pd.concat([df, df.iloc[src]], ignore_index=True)
        meta = pd.concat([meta, meta.iloc[src]], ignore_index=True)
        y, fam = np.r_[y, y[src]], np.r_[fam, fam[src]]
    if near_dup_frac > 0:
        k = int(n * near_dup_frac)
        src = rng.integers(0, n, k)
        nd = df.iloc[src].copy()
        for c in ("dur", "sbytes", "dbytes", "rate", "sload", "dload"):
            nd[c] = nd[c] * rng.uniform(0.995, 1.005, k)
        nm = meta.iloc[src].copy()
        nm["stime"] = nm["stime"] + rng.uniform(0, 2.0, k)
        df = pd.concat([df, nd], ignore_index=True)
        meta = pd.concat([meta, nm], ignore_index=True)
        y, fam = np.r_[y, y[src]], np.r_[fam, fam[src]]
    order = np.argsort(meta["stime"].to_numpy(), kind="stable")
    return FlowData(df.iloc[order].reset_index(drop=True), y[order], meta.iloc[order].reset_index(drop=True),
                    fam[order],
                    {"synthetic": True, "sha256": None, "path": None, "version": "synthetic-v1",
                     "generator_args": dict(n=n, attack_frac=attack_frac, seed=seed, dup_frac=dup_frac,
                                            drift=drift, separation=separation, near_dup_frac=near_dup_frac)})


def make_shift_flows(n_days: int = 5, per_day: int = 3000, drift_per_day: float = 0.35, attack_schedule=None,
                     seed: int = 0, separation: float = 1.0, dup_frac: float = 0.0) -> FlowData:
    """SYNTHETIC multi-day traffic with BENIGN drift (+drift_per_day on log-duration and log-packets per day) and
    attack families that appear on different days -- the temporal/LOFO stress test, for smoke tests ONLY.
    Day 0 is benign-only (like CIC-IDS2017's Monday). attack_schedule: {day: (attack_frac, [families])}."""
    sched = attack_schedule or {0: (0.0, []), 1: (0.25, ["dos", "recon"]), 2: (0.25, ["recon", "exploit"]),
                                3: (0.2, ["exploit", "backdoor"]), 4: (0.3, ["dos", "backdoor"])}
    parts = []
    for d in range(n_days):
        frac, fams = sched.get(d, (0.2, FAMILIES))
        fd = make_synthetic_flows(per_day, frac, seed * 1000 + d, dup_frac=dup_frac, drift=0.0, separation=separation,
                                  drift_offset=drift_per_day * d, families=fams or None, t0=float(d))
        fd.meta["day"] = f"day{d}"
        parts.append(fd)
    X = pd.concat([p.X for p in parts], ignore_index=True)
    meta = pd.concat([p.meta for p in parts], ignore_index=True)
    y = np.concatenate([p.y for p in parts]); fam = np.concatenate([p.attack_cat for p in parts])
    return FlowData(X, y, meta, fam, {"synthetic": True, "sha256": None, "path": None, "version": "synthetic-shift-v1",
                                       "generator_args": dict(n_days=n_days, per_day=per_day, drift_per_day=drift_per_day, seed=seed,
                                                              separation=separation, dup_frac=dup_frac)})
