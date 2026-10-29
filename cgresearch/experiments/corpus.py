"""Synthetic Attack-DNA corpus with explicit ground truth for retrieval experiments.

Ground truth is derived from the generator labels (family = attack_type, template = which
action template). This is CIRCULAR w.r.t. the vector construction for behavioural genomes and
must be replaced/complemented by expert-labelled incidents before journal claims are made
(see roadmap). Relevance: 2 = same template, 1 = same family, 0 otherwise.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .._backend import BACKEND  # noqa: F401
from dna import ACTION_VOCAB, AttackDNA, BehavioralGenome, ContextGenome, NetworkGenome, ResourceGenome  # type: ignore
from simulator import ATTACK_TEMPLATES  # type: ignore

FAMILIES = list(ATTACK_TEMPLATES)
TEMPLATES = [(f, i) for f in FAMILIES for i in range(len(ATTACK_TEMPLATES[f]))]
FAMILY_ACTION = {"privilege_escalation": "rotate_credentials", "credential_abuse": "rotate_credentials",
                 "data_exfiltration": "isolate_pod", "lateral_movement": "apply_network_policy"}
SERVICES = ["k8s-pod", "lambda", "database", "iam", "web", "api"]
PORTS = [443, 8080, 22, 3306, 6443]


def make_dna(rng: np.random.Generator, tpl_idx: int, noise: float = 1.0) -> AttackDNA:
    fam, ti = TEMPLATES[tpl_idx]
    seq = list(ATTACK_TEMPLATES[fam][ti])
    if rng.random() < 0.3 * noise:  # sequence variation: insert benign-looking noise action
        seq.insert(int(rng.integers(0, len(seq) + 1)), str(rng.choice(["search", "view_profile", "logout"])))
    svc = str(rng.choice(SERVICES))
    n = len(seq)
    return AttackDNA.new(
        fam,
        NetworkGenome(ports=sorted(rng.choice(PORTS, int(rng.integers(1, 3)), replace=False).tolist()),
                      protocol=str(rng.choice(["tcp", "udp"], p=[.9, .1])),
                      request_rate=float(rng.uniform(20, 200)), unique_destinations=int(rng.integers(1, 15))),
        BehavioralGenome(api_sequence=seq, action_frequency=float(rng.uniform(20, 200)),
                         failed_logins=int(rng.integers(0, 6)) + (3 if fam == "credential_abuse" else 0),
                         unique_resources=int(rng.integers(1, 15)), timing_regularity=float(rng.uniform(0.2, 0.95))),
        ResourceGenome(cpu_pct=float(rng.uniform(40, 95)), memory_pct=float(rng.uniform(30, 90)),
                       network_egress_kbps=float(rng.uniform(500, 4500) * (1.5 if fam == "data_exfiltration" else 1.0)),
                       process_spawns=int(rng.integers(0, 4))),
        ContextGenome(service=svc, service_criticality="high" if svc in ("database", "iam") else "medium"))


def make_corpus(n: int, seed: int) -> Tuple[List[AttackDNA], np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
    tpl = rng.integers(0, len(TEMPLATES), n)
    return [make_dna(rng, int(t)) for t in tpl], tpl


def relevance(q_tpl: np.ndarray, c_tpl: np.ndarray) -> np.ndarray:
    """(nq, nc) graded relevance from template ids."""
    fam_of = np.array([FAMILIES.index(f) for f, _ in TEMPLATES])
    same_tpl = q_tpl[:, None] == c_tpl[None, :]
    same_fam = fam_of[q_tpl][:, None] == fam_of[c_tpl][None, :]
    return np.where(same_tpl, 2, np.where(same_fam, 1, 0))
