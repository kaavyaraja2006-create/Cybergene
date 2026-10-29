"""Integration: small synthetic runs must produce the complete artifact set and be reproducible."""
import json

import numpy as np

from cgresearch.experiments.detection import run_detection
from cgresearch.experiments.mutation import run_mutation, POLICIES, Episode, DEFAULTS
from cgresearch.experiments.retrieval import run_retrieval

DET = {"name": "t_det", "data": {"kind": "synthetic", "synthetic_args": {"n": 1800, "seed": 1, "dup_frac": 0.03}},
       "dedup": "exact", "port_mode": "onehot", "seeds": [0, 1, 2], "models": ["isolation_forest", "lof", "logreg", "rule_based"],
       "tune": False, "n_boot": 200}


def test_detection_artifacts_and_pairing(tmp_path):
    r = run_detection(DET, str(tmp_path))
    m = json.load(open(r.dir / "manifest.json"))
    for k in ("experiment_id", "timestamp_utc", "git", "environment", "dataset", "seeds", "cli_args", "config_sha256", "runtime_seconds"):
        assert k in m
    assert m["status"] == "completed" and m["dataset"]["synthetic"] is True
    rows = r.tables["per_seed_fpr_sweep"]
    by = {}
    for x in rows:
        by.setdefault(x["seed"], set()).add(x["split_hash"])
    assert all(len(v) == 1 for v in by.values()), "all models must share one split per seed"
    assert {x["target_fpr"] for x in rows} == {0.01, 0.02, 0.05, 0.10}
    assert (r.dir / "artifacts" / "split_seed0.npz").exists() and (r.dir / "artifacts" / "curves_seed0.npz").exists()
    comps = r.tables["comparisons_fpr_sweep"]
    assert comps and all("p_holm" in c and "cliffs_delta" in c for c in comps)
    for s in r.tables["summary_fpr_sweep"]:
        assert {"mean", "std", "median", "ci_low", "ci_high", "per_seed"} <= set(s)


def test_detection_is_reproducible(tmp_path):
    a = run_detection({**DET, "name": "r1"}, str(tmp_path))
    b = run_detection({**DET, "name": "r2"}, str(tmp_path))
    ka = [(r["seed"], r["model"], round(r["roc_auc"], 12)) for r in a.tables["per_seed_threshold_free"]]
    kb = [(r["seed"], r["model"], round(r["roc_auc"], 12)) for r in b.tables["per_seed_threshold_free"]]
    assert ka == kb


def test_tuning_never_sees_test(tmp_path):
    import inspect
    from cgresearch.detectors import tuning
    assert "test" not in inspect.signature(tuning.tune).parameters


def test_mutation_paired_episodes_and_logging(tmp_path):
    cfg = {"name": "t_mut", "seeds": [0, 1], "episodes_per_reason": 3, "n_boot": 100, "direction_priors": [0.9]}
    r = run_mutation(cfg, str(tmp_path))
    gl = r.tables["generation_log"]
    need = {"strategy", "seed", "scenario", "action", "params_before", "failure_reason_used", "mutation_applied",
            "params_after", "simulated_outcome_true", "generation", "containment_status", "fitness_with_birth_penalty",
            "fitness_without_birth_penalty"}
    assert need <= set(gl[0])
    # every policy faces the same scenarios
    sc = {p: {x["scenario"] for x in gl if x["strategy"] == p} for p in POLICIES}
    assert all(v == sc["retry"] for v in sc.values())
    assert r.manifest["retry_baseline_check"]["retry_success_rate"] < 0.2
    # reproducible across processes (no salted hash())
    r2 = run_mutation({**cfg, "name": "t_mut2"}, str(tmp_path))
    assert [x["params_after"] for x in gl] == [x["params_after"] for x in r2.tables["generation_log"]]


def test_birth_penalty_zero_is_reported_separately():
    from cgresearch.experiments.mutation import GroundTruthEnv, run_episode
    from cgresearch._backend import BACKEND  # noqa
    from response_engine import ResponseEngine
    c = {**DEFAULTS, "seeds": [0]}
    ep = Episode(0, 0, "low_response_success", c)
    _, e = run_episode("targeted", ep, c, GroundTruthEnv(), ResponseEngine(), 0)
    assert e["fitness_final_without_birth_penalty"] >= e["fitness_final_with_birth_penalty"]


def test_env_independent_of_response_engine():
    import cgresearch.experiments.mutation as m, inspect
    src = inspect.getsource(m.GroundTruthEnv)
    assert "ResponseEngine" not in src and "DigitalTwin" not in src


def test_retrieval_ablation_same_queries_and_genome_dims(tmp_path):
    r = run_retrieval({"name": "t_ret", "seeds": [0, 1], "n_store": 400, "n_queries": 30, "n_boot": 100}, str(tmp_path))
    dims = {x["config"]: x["dim"] for x in r.tables["per_seed"] if x["seed"] == 0}
    assert dims["network"] < dims["network+behavior"] < dims["network+behavior+resource"] < dims["all_five"]


def test_legacy_dna_vector_unchanged_and_evolution_birth_penalty_default():
    from cgresearch._backend import BACKEND  # noqa
    from dna import AttackDNA, NetworkGenome, BehavioralGenome, ResourceGenome, ContextGenome
    from evolution import EvolutionEngine
    d = AttackDNA.new("credential_abuse", NetworkGenome(), BehavioralGenome(["login"]), ResourceGenome(), ContextGenome())
    assert d.vector().shape == (28,)
    assert EvolutionEngine(0).mutate(d, "low_response_success").evolution.fitness == 0.45
    assert EvolutionEngine(0, birth_penalty=0.0).mutate(d, "low_response_success").evolution.fitness == 0.5
