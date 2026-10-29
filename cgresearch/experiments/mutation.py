"""Redesigned mutation experiment (experiment 7).

IMPORTANT SCOPE NOTE: this is a SIMULATION study. Outcomes come from a hidden ground-truth
environment (`GroundTruthEnv`) that is independent of the Response Engine ranking model and of the
Digital Twin. Conclusions describe mechanism behaviour under the stated assumptions (esp.
`direction_prior`, `reason_noise`), not real-world containment efficacy. A sensitivity sweep over
those assumptions is part of the experiment.

Gene space (units = one targeted-mutation step):
  u0 timing_regularity/0.15   u1 #distinct actions   u2 cpu_pct/10   u3 unique_destinations
Default targeted directions: (-1, +1, -1, -1) for reasons
  low_response_success, novel_action_seen, high_business_impact, dependency_failure.
Each episode hides a target t = u_start + sign * m * e_d (m=3 steps) in the dimension d that the
failure reason implicates (random d, random sign for repeated_identical_failure).
sign equals the default direction with probability `direction_prior`.
P(success | u) = p_max * exp(-||u - t||^2 / 2); initial P ~= 0.01 (retry fails w.h.p.);
repeated_identical_failure: P = 0 at the unchanged initial parameters (deterministic repeat).
Common random numbers: all policies see identical per-episode uniforms/noise.
"""
from __future__ import annotations

import zlib
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .._backend import BACKEND  # noqa: F401  (puts backend on sys.path)
from dna import ACTION_VOCAB, AttackDNA, BehavioralGenome, ContextGenome, NetworkGenome, ResourceGenome  # type: ignore
from evolution import EvolutionEngine  # type: ignore
from response_engine import ResponseEngine  # type: ignore
from digital_twin import DigitalTwin  # type: ignore

from ..runlog import ExperimentRun
from ..stats import paired_comparisons, summarize

REASONS = ["low_response_success", "novel_action_seen", "high_business_impact",
           "dependency_failure", "repeated_identical_failure"]
REASON_DIM = {"low_response_success": 0, "novel_action_seen": 1, "high_business_impact": 2, "dependency_failure": 3}
DEFAULT_DIR = np.array([-1.0, 1.0, -1.0, -1.0])
POLICIES = ["retry", "random", "generic", "targeted", "fixed_rule", "history_only", "detector_only"]
U_START = np.array([6.0, 5.0, 8.0, 8.0])
ATTACK_FOR_REASON = {"low_response_success": "credential_abuse", "novel_action_seen": "privilege_escalation",
                     "high_business_impact": "data_exfiltration", "dependency_failure": "lateral_movement",
                     "repeated_identical_failure": "credential_abuse"}


def _dna_from_u(u, base: AttackDNA | None = None) -> AttackDNA:
    k = int(np.clip(round(u[1]), 1, len(ACTION_VOCAB)))
    d = AttackDNA.new(
        "credential_abuse", NetworkGenome(ports=[443], unique_destinations=int(max(0, round(u[3])))),
        BehavioralGenome(api_sequence=list(ACTION_VOCAB[:k]), timing_regularity=round(max(0.0, u[0] * EvolutionEngine.STEP_TIMING), 2)),
        ResourceGenome(cpu_pct=round(u[2] * EvolutionEngine.STEP_CPU, 1)), ContextGenome())
    return d


def _u_from_dna(d: AttackDNA) -> np.ndarray:
    return np.array([d.behavior.timing_regularity / EvolutionEngine.STEP_TIMING, float(len(d.behavior.api_sequence)),
                     d.resource.cpu_pct / EvolutionEngine.STEP_CPU, float(d.network.unique_destinations)])


class Episode:
    """Scenario + all pre-drawn randomness (common random numbers across policies)."""
    def __init__(self, seed: int, idx: int, reason: str, cfg: Dict[str, Any]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, idx, 7]))
        G = cfg["max_generations"]
        self.seed, self.idx, self.reason = seed, idx, reason
        self.scenario_id = f"s{seed}-e{idx}-{reason}"
        d = REASON_DIM.get(reason, int(rng.integers(0, 4)))
        informative_default = DEFAULT_DIR[d] if reason in REASON_DIM else float(rng.choice([-1, 1]))
        agree = rng.random() < cfg["direction_prior"] if reason in REASON_DIM else True
        sign = informative_default if agree else -informative_default
        self.target = U_START.copy()
        self.target[d] += sign * cfg["initial_offset_steps"]
        self.deterministic_fail_at_start = reason == "repeated_identical_failure"
        self.u_succ = rng.random(G); self.u_fs = rng.random(G)
        self.resid_noise = rng.normal(0, cfg["residual_obs_sd"], G)
        self.det_est = self.target + rng.normal(0, cfg["detector_estimate_sd"], 4)
        self.reported_reason = reason
        if rng.random() < cfg["reason_noise"]:
            self.reported_reason = str(rng.choice([r for r in REASONS if r != reason]))


class GroundTruthEnv:
    p_max = 0.95

    def p_success(self, ep: Episode, u: np.ndarray) -> float:
        if ep.deterministic_fail_at_start and np.allclose(u, U_START):
            return 0.0
        return float(self.p_max * np.exp(-0.5 * np.sum((u - ep.target) ** 2)))


def _clip_u(u):
    return np.array([max(0.0, u[0]), np.clip(u[1], 1, len(ACTION_VOCAB)),
                     max(0.0, u[2]), max(0.0, u[3])])


class PolicyState:
    pass


def propose(policy: str, ep: Episode, u: np.ndarray, g: int, rng: np.random.Generator, st: PolicyState,
            engine: EvolutionEngine, last_resid: float | None):
    """Return (new_u, mutation_label). Policies see only what they are allowed to see."""
    if policy == "retry":
        return u.copy(), "none"
    if policy == "random":
        d = int(rng.integers(0, 4))
        v = u.copy(); v[d] += rng.normal(0, 1.0)
        return v, f"random_dim{d}"
    if policy == "generic":
        return u * rng.uniform(0.9, 1.1, 4), "multiplicative_jitter_0.9_1.1"  # mirrors legacy 'resource_jitter'
    if policy == "targeted":
        dna = _dna_from_u(u)
        child = engine.mutate(dna, ep.reported_reason)
        return _u_from_dna(child), "+".join(child.evolution.mutations)
    if policy == "fixed_rule":
        d = (g - 1) % 4  # static playbook ladder: timing, repertoire, intensity, scope
        v = u.copy(); v[d] += DEFAULT_DIR[d]
        return v, f"ladder_dim{d}"
    if policy == "history_only":
        # greedy coordinate hill-climb on the (noisy) residual; ignores the failure reason
        if st.__dict__.get("prev") is not None and last_resid is not None and last_resid > st.best_resid + 1e-9:
            u = st.best_u.copy()  # revert the worsening step
        if last_resid is not None and (st.__dict__.get("best_resid") is None or last_resid < st.best_resid):
            st.best_resid, st.best_u = last_resid, u.copy()
        st.prev = u.copy()
        d = int(rng.integers(0, 4)); s = float(rng.choice([-1, 1]))
        v = u.copy(); v[d] += s
        return v, f"hillclimb_dim{d}{'+' if s > 0 else '-'}"
    if policy == "detector_only":
        # telemetry-derived estimate of the true parameters, noise sd = detector_estimate_sd
        return u + 0.5 * (ep.det_est - u), "move_halfway_to_detector_estimate"
    raise ValueError(policy)


def run_episode(policy: str, ep: Episode, cfg: Dict[str, Any], env: GroundTruthEnv, resp: ResponseEngine, seed: int):
    G = cfg["max_generations"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, ep.idx, zlib.crc32(policy.encode()), 11]))
    engine = EvolutionEngine(seed=int(rng.integers(0, 2 ** 31)), birth_penalty=cfg["birth_penalty"])
    action = resp.recommend(ATTACK_FOR_REASON[ep.reason], "high", {}).action  # logged only; NOT used for outcomes
    u = U_START.copy(); st = PolicyState(); last_resid = None
    fit_pen = fit_nopen = 0.5
    rows = []; contained = False; observed_stop = False
    cost = 0.0
    for g in range(1, G + 1):
        p = env.p_success(ep, u)
        true_ok = bool(ep.u_succ[g - 1] < p)
        obs_ok = true_ok or bool(ep.u_fs[g - 1] < cfg["false_success_rate"])
        resid = float(np.linalg.norm(u - ep.target) + ep.resid_noise[g - 1])
        cost += cfg["attempt_cost"] + cfg["intensity_cost"] * u[2]
        fit_pen = round(max(0.0, min(1.0, fit_pen + (0.08 if true_ok else -0.12))), 2)
        fit_nopen = round(max(0.0, min(1.0, fit_nopen + (0.08 if true_ok else -0.12))), 2)
        row = {"strategy": policy, "seed": seed, "scenario": ep.scenario_id, "episode": ep.idx, "true_reason": ep.reason,
               "reported_reason": ep.reported_reason, "action": action, "generation": g,
               "params_before": u.round(3).tolist(), "p_success_true": round(p, 5),
               "simulated_outcome_true": true_ok, "observed_success": obs_ok,
               "mutation_applied": None, "params_after": None, "failure_reason_used": None}
        if obs_ok:
            observed_stop = True; contained = true_ok
            row.update({"containment_status": "contained" if true_ok else "FALSE_CONTAINMENT",
                        "fitness_with_birth_penalty": fit_pen, "fitness_without_birth_penalty": fit_nopen,
                        "params_after": u.round(3).tolist(), "mutation_applied": "none (stopped)"})
            rows.append(row); break
        if g == G:
            row.update({"containment_status": "failed_cap_reached", "fitness_with_birth_penalty": fit_pen,
                        "fitness_without_birth_penalty": fit_nopen, "params_after": u.round(3).tolist(),
                        "mutation_applied": "none (cap)"})
            rows.append(row); break
        v, label = propose(policy, ep, u, g, rng, st, engine, resid)
        v = _clip_u(v)
        if policy != "retry":
            fit_pen = round(max(0.1, fit_pen - cfg["birth_penalty_report"]), 2)  # prototype's fixed 0.05 handicap
        row.update({"containment_status": "failed_retrying", "failure_reason_used": ep.reported_reason,
                    "mutation_applied": label, "params_after": v.round(3).tolist(),
                    "fitness_with_birth_penalty": fit_pen, "fitness_without_birth_penalty": fit_nopen})
        rows.append(row)
        u, last_resid = v, resid
    n_gen = rows[-1]["generation"]
    ep_row = {"strategy": policy, "seed": seed, "episode": ep.idx, "reason": ep.reason, "contained": contained,
              "false_containment": bool(observed_stop and not contained), "failed": not contained and not observed_stop,
              "generations": n_gen, "generations_censored": n_gen if (contained) else G + 1,
              "cost": cost + (cfg["fail_penalty"] if not contained and not observed_stop else 0.0)
                           + (cfg["false_containment_penalty"] if observed_stop and not contained else 0.0),
              "fitness_final_with_birth_penalty": rows[-1]["fitness_with_birth_penalty"],
              "fitness_final_without_birth_penalty": rows[-1]["fitness_without_birth_penalty"]}
    return rows, ep_row


DEFAULTS = dict(max_generations=8, episodes_per_reason=20, direction_prior=0.9, initial_offset_steps=3.0,
                residual_obs_sd=0.3, detector_estimate_sd=1.0, reason_noise=0.0, false_success_rate=0.05,
                attempt_cost=1.0, intensity_cost=0.05, fail_penalty=10.0, false_containment_penalty=20.0,
                birth_penalty=0.05, birth_penalty_report=0.05, policies=POLICIES)

METRICS = ["success_rate", "false_containment_rate", "failed_rate", "generations_censored", "cost",
           "fitness_final_with_birth_penalty", "fitness_final_without_birth_penalty"]


def run_mutation(cfg: Dict[str, Any], out_root="results", argv=None, log_generations=True) -> ExperimentRun:
    cfg = {**DEFAULTS, **cfg}
    run = ExperimentRun(cfg.get("name", "mutation"), cfg, out_root, argv, {"synthetic": True, "note": "simulated episodes"}, cfg["seeds"])
    env = GroundTruthEnv()
    resp = ResponseEngine(DigitalTwin(seed=12345))  # separate model: used for action naming only
    priors = cfg.get("direction_priors", [cfg["direction_prior"]])
    per_seed_by_prior = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for prior in priors:
        c = {**cfg, "direction_prior": prior}
        for seed in cfg["seeds"]:
            eps = [Episode(seed, i * len(REASONS) + j, r, c) for i in range(c["episodes_per_reason"]) for j, r in enumerate(REASONS)]
            for pol in c["policies"]:
                ep_rows = []
                for ep in eps:
                    g_rows, e_row = run_episode(pol, ep, c, env, resp, seed)
                    e_row["direction_prior"] = prior
                    ep_rows.append(e_row)
                    if log_generations:
                        for r in g_rows:
                            r["direction_prior"] = prior
                        run.add_rows("generation_log", g_rows)
                run.add_rows("episodes", ep_rows)
                agg = {"success_rate": np.mean([r["contained"] for r in ep_rows]),
                       "false_containment_rate": np.mean([r["false_containment"] for r in ep_rows]),
                       "failed_rate": np.mean([r["failed"] for r in ep_rows]),
                       "generations_censored": np.mean([r["generations_censored"] for r in ep_rows]),
                       "cost": np.mean([r["cost"] for r in ep_rows]),
                       "fitness_final_with_birth_penalty": np.mean([r["fitness_final_with_birth_penalty"] for r in ep_rows]),
                       "fitness_final_without_birth_penalty": np.mean([r["fitness_final_without_birth_penalty"] for r in ep_rows])}
                cont = [r["generations"] for r in ep_rows if r["contained"]]
                run.add_row("per_seed_policy", {"direction_prior": prior, "seed": seed, "strategy": pol, **agg,
                                                "generations_to_containment_given_success": float(np.mean(cont)) if cont else None,
                                                "n_episodes": len(ep_rows), "failed_episodes": int(sum(r["failed"] for r in ep_rows))})
                for m, v in agg.items():
                    per_seed_by_prior[prior][m][pol][seed] = float(v)
        # per-reason breakdown
    ep = run.tables["episodes"]
    br = []
    for prior in priors:
        for pol in cfg["policies"]:
            for reason in REASONS:
                sub = [r for r in ep if r["strategy"] == pol and r["reason"] == reason and r["direction_prior"] == prior]
                br.append({"direction_prior": prior, "strategy": pol, "reason": reason, "episodes": len(sub),
                           "success_rate": float(np.mean([r["contained"] for r in sub])),
                           "mean_generations_censored": float(np.mean([r["generations_censored"] for r in sub]))})
    run.save_table("by_reason", br)
    summ, comps = [], []
    for prior in priors:
        for m in METRICS:
            for pol in cfg["policies"]:
                vals = [per_seed_by_prior[prior][m][pol][s] for s in sorted(per_seed_by_prior[prior][m][pol])]
                summ.append({"direction_prior": prior, "metric": m, "strategy": pol, **summarize(vals, cfg.get("n_boot", 10_000)), "per_seed": vals})
            base = "retry" if "retry" in cfg["policies"] else cfg["policies"][0]
            pairs = [(p, base) for p in cfg["policies"] if p != base]
            for r in paired_comparisons(per_seed_by_prior[prior][m], m, pairs):
                r["direction_prior"] = prior
                comps.append(r)
    run.save_table("summary", summ)
    run.save_table("comparisons_vs_retry", comps)
    retry_fail = [r for r in ep if r["strategy"] == "retry" and r["direction_prior"] == priors[0]]
    run.manifest["retry_baseline_check"] = {
        "retry_success_rate": float(np.mean([r["contained"] for r in retry_fail])) if retry_fail else None,
        "requirement": "unmodified retry should fail within the cap in (nearly) all scenarios",
    }
    run.finish()
    return run
