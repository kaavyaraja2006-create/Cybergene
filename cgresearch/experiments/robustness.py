"""Robustness experiments (experiment 10): drift, memory poisoning, adversarial telemetry.
All paired by seed; all raw per-seed rows saved. Synthetic data unless a UNSW path is configured."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import numpy as np

from ..data.unsw import FlowData
from ..data.splits import make_split
from ..detectors import REGISTRY
from ..metrics import calibrate_threshold, threshold_free, threshold_metrics
from ..preprocess import FlowPreprocessor
from ..runlog import ExperimentRun
from ..stats import paired_comparisons, summarize
from .common import prepare
from .corpus import FAMILIES, FAMILY_ACTION, TEMPLATES, make_corpus, relevance
from .retrieval import l2norm


def _fit_det(name, X, y, seed, pre):
    kw = {"feature_names": pre.num_cols} if name == "rule_based" else {}
    d = REGISTRY[name](seed=seed, **kw)
    d.fit(X, y if d.supervised else None)
    return d


# ---------------------------------------------------------------- drift
def run_drift(cfg, run: ExperimentRun, data: FlowData):
    n = len(data); W = cfg.get("windows", 5)
    order = np.argsort(data.meta["stime"].to_numpy(), kind="stable") if "stime" in data.meta else np.arange(n)
    wins = np.array_split(order, W)
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 31]))
        parts = []
        for w in wins:
            b = w[data.y[w] == 0]; b = b[rng.permutation(len(b))]
            a = w[data.y[w] == 1]
            f, c = int(0.4 * len(b)), int(0.6 * len(b))
            parts.append((b[:f], b[f:c], np.r_[b[c:], a]))
        for mname in cfg.get("models", ["isolation_forest", "lof", "autoencoder"]):
            for policy in ("static", "retrain_previous_window"):
                for wi in range(W):
                    src = 0 if policy == "static" else max(wi - 1, 0)
                    fit_i, cal_i, _ = parts[src]
                    test_i = parts[wi][2]
                    pre = FlowPreprocessor(cfg.get("port_mode", "onehot")).fit(data, fit_i)
                    det = _fit_det(mname, pre.transform(data, fit_i), None, seed, pre)
                    thr = calibrate_threshold(det.score(pre.transform(data, cal_i)), 0.05)
                    s = det.score(pre.transform(data, test_i)); y = data.y[test_i]
                    run.add_row("drift", {"seed": seed, "model": mname, "policy": policy, "window": wi,
                                          **threshold_free(y, s), **{k: v for k, v in threshold_metrics(y, s, thr).items() if k in ("recall", "measured_fpr", "f1", "mcc")}})


# ---------------------------------------------------------------- poisoning
def _vote(sim_row, idx, actions, k, weights=True):
    top = idx[:k]
    score = defaultdict(float)
    for j in top:
        score[actions[j]] += (sim_row[j] if weights else 1.0)
    return max(score, key=score.get) if score else None


def run_poisoning(cfg, run: ExperimentRun):
    n_store, n_q, k = cfg.get("n_store", 2000), cfg.get("n_queries", 200), cfg.get("k", 5)
    rates = cfg.get("poison_rates", [0.0, 0.01, 0.05, 0.10, 0.20])
    all_actions = sorted(set(FAMILY_ACTION.values()) | {"block_ip", "rate_limit", "disable_function"})
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 41]))
        store, st = make_corpus(n_store, seed); qs, qt = make_corpus(n_q, seed + 10_000)
        M = l2norm(np.vstack([d.vector() for d in store])); Q = l2norm(np.vstack([d.vector() for d in qs]))
        fam_idx = {f: np.nonzero([TEMPLATES[t][0] == f for t in st])[0] for f in FAMILIES}
        clean_actions = np.array([FAMILY_ACTION[TEMPLATES[t][0]] for t in st], dtype=object)
        q_fam = [TEMPLATES[t][0] for t in qt]
        for ptype in cfg.get("poison_types", ["mimic_wrong_action", "random_false"]):
            for rate in rates:
                n_p = int(rate * n_store)
                tf = rng.choice(FAMILIES, n_p)
                if ptype == "mimic_wrong_action":
                    src = np.array([rng.choice(fam_idx[f]) for f in tf]) if n_p else np.array([], int)
                    PV = M[src] + rng.normal(0, 0.005, (n_p, M.shape[1])) if n_p else np.zeros((0, M.shape[1]))
                    pa = np.array([rng.choice([a for a in all_actions if a != FAMILY_ACTION[f]]) for f in tf], dtype=object)
                else:
                    PV = rng.random((n_p, M.shape[1])) ** 2
                    pa = np.array([rng.choice(all_actions) for _ in range(n_p)], dtype=object)
                PV = l2norm(PV) if n_p else PV
                for mit in ("none", "knn_vetting", "duplicate_cap"):
                    keep = np.ones(n_p, bool)
                    if mit == "knn_vetting" and n_p:
                        # accept a new record only if its action agrees with the majority action of its
                        # k nearest TRUSTED (pre-existing) records
                        S = PV @ M.T
                        for i in range(n_p):
                            nn = np.argpartition(-S[i], 10)[:10]
                            maj = max(set(clean_actions[nn]), key=list(clean_actions[nn]).count)
                            keep[i] = pa[i] == maj
                    MM = np.vstack([M, PV[keep]]) if n_p else M
                    AA = np.r_[clean_actions, pa[keep]] if n_p else clean_actions
                    is_p = np.r_[np.zeros(len(M), bool), np.ones(int(keep.sum()), bool)] if n_p else np.zeros(len(M), bool)
                    S = Q @ MM.T
                    pol, wrong, p5 = [], [], []
                    for i in range(n_q):
                        order = np.argsort(-S[i], kind="stable")
                        if mit == "duplicate_cap":  # at most 2 results per near-identical cluster (cos>0.999 to a kept result)
                            kept = []
                            for j in order[:200]:
                                if sum(MM[j] @ MM[kk] > 0.999 for kk in kept) < 2:
                                    kept.append(j)
                                if len(kept) == k:
                                    break
                            top = np.array(kept)
                        else:
                            top = order[:k]
                        pol.append(float(is_p[top].mean()))
                        rec = _vote(S[i], top, AA, k)
                        wrong.append(float(rec != FAMILY_ACTION[q_fam[i]]))
                    run.add_row("poisoning", {"seed": seed, "poison_type": ptype, "poison_rate": rate, "mitigation": mit,
                                              "retrieval_pollution_at_k": float(np.mean(pol)),
                                              "wrong_response_rate": float(np.mean(wrong)),
                                              "poison_records_accepted": int(keep.sum()), "poison_records_attempted": n_p, "k": k})


# ---------------------------------------------------------------- adversarial telemetry
def _perturb(frame, factors):
    f = frame.copy()
    for c, i in (("dur", 0), ("sbytes", 1), ("dbytes", 2), ("spkts", 3), ("dpkts", 4)):
        if c in f:
            f[c] = f[c].to_numpy(float) * factors[:, i]
    dur = np.maximum(f["dur"].to_numpy(float), 1e-3) if "dur" in f else None
    if dur is not None:
        if "rate" in f: f["rate"] = (f["spkts"] + f["dpkts"]) / dur
        if "sload" in f: f["sload"] = f["sbytes"] * 8 / dur
        if "dload" in f: f["dload"] = f["dbytes"] * 8 / dur
    return f


def run_adversarial(cfg, run: ExperimentRun, data: FlowData):
    eps_list = cfg.get("epsilons", [0.0, 0.1, 0.25, 0.5]); Q = cfg.get("query_budget", 30)
    names = cfg.get("models", ["isolation_forest", "lof", "autoencoder"])
    for seed in cfg["seeds"]:
        sp = make_split(data.y, seed)
        pre = FlowPreprocessor(cfg.get("port_mode", "onehot")).fit(data, sp["benign_train"])
        dets = {m: _fit_det(m, pre.transform(data, sp["benign_train"]), None, seed, pre) for m in names}
        bv = pre.transform(data, sp["benign_val"])
        cdf = {m: np.sort(d.score(bv)) for m, d in dets.items()}
        def ens(X):  # rank-average ensemble: each score mapped through its benign-val empirical CDF
            return np.mean([np.searchsorted(cdf[m], d.score(X), side="right") / len(cdf[m]) for m, d in dets.items()], axis=0)
        scorers = {m: d.score for m, d in dets.items()}; scorers["ensemble_rank_avg"] = ens
        att_idx = sp["attack_test"]; frame0 = data.X.iloc[att_idx].reset_index(drop=True)
        ben_te = pre.transform(data, sp["benign_test"])
        def tx(frame):
            return pre.transform(FlowData(frame, np.ones(len(frame), int), data.meta.iloc[att_idx].reset_index(drop=True), data.attack_cat[att_idx]))
        for sname, fn in scorers.items():
            thr = calibrate_threshold(fn(bv), 0.05)
            s0 = fn(tx(frame0)); det0 = s0 > thr
            for eps in eps_list:
                rng = np.random.default_rng(np.random.SeedSequence([seed, 53]))  # same proposals for every scorer (paired)
                best_f = np.ones((len(frame0), 5)); best_s = s0.copy()
                if eps > 0:
                    for _ in range(Q):
                        cand = rng.uniform(1 - eps, 1 + eps, (len(frame0), 5))
                        s = fn(tx(_perturb(frame0, cand)))
                        better = s < best_s
                        best_f[better] = cand[better]; best_s[better] = s[better]
                adv = best_s
                y = np.r_[np.zeros(len(ben_te)), np.ones(len(adv))]
                allscore = np.r_[fn(ben_te), adv]
                tm = threshold_metrics(y, allscore, thr)
                run.add_row("adversarial", {"seed": seed, "scorer": sname, "epsilon": eps, "query_budget": Q if eps > 0 else 0,
                                            "recall_at_5pct_fpr": tm["recall"], "measured_fpr": tm["measured_fpr"],
                                            "roc_auc": threshold_free(y, allscore)["roc_auc"],
                                            "evasion_rate_among_detected": float(((adv <= thr) & det0).sum() / max(det0.sum(), 1)),
                                            "n_attack": int(len(adv))})


def run_robustness(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    which = cfg.get("which", ["drift", "poisoning", "adversarial"])
    need_data = [w for w in which if w in ("drift", "adversarial")]
    data = dsinfo = None
    if need_data:
        data, dsinfo, dd = prepare(cfg)
    run = ExperimentRun(cfg.get("name", "robustness"), cfg, out_root, argv, dsinfo or {"synthetic": True}, cfg["seeds"])
    if "drift" in which: run_drift(cfg.get("drift", {**cfg}), run, data)
    if "poisoning" in which: run_poisoning(cfg.get("poisoning", {**cfg}), run)
    if "adversarial" in which: run_adversarial(cfg.get("adversarial", {**cfg}), run, data)
    nb = cfg.get("n_boot", 10_000)
    def agg(table, keys, metrics):
        groups = defaultdict(lambda: defaultdict(dict))
        for r in run.tables.get(table, []):
            g = tuple(r[k] for k in keys)
            for m in metrics: groups[g][m][r["seed"]] = r[m]
        return [{**dict(zip(keys, g)), "metric": m, **summarize([v[s] for s in sorted(v)], nb), "per_seed": [v[s] for s in sorted(v)]}
                for g, ms in groups.items() for m, v in ms.items()]
    if "drift" in which:
        run.save_table("summary_drift", agg("drift", ("model", "policy", "window"), ["roc_auc", "recall", "measured_fpr"]))
    if "poisoning" in which:
        run.save_table("summary_poisoning", agg("poisoning", ("poison_type", "poison_rate", "mitigation"), ["retrieval_pollution_at_k", "wrong_response_rate"]))
        by = defaultdict(lambda: defaultdict(dict))
        for r in run.tables["poisoning"]:
            by[(r["poison_type"], r["poison_rate"])][r["mitigation"]][r["seed"]] = r["wrong_response_rate"]
        comps = []
        for (pt, pr), d in by.items():
            if pr > 0:
                for c in paired_comparisons(d, "wrong_response_rate", [("none", "knn_vetting"), ("none", "duplicate_cap")]):
                    c.update({"poison_type": pt, "poison_rate": pr}); comps.append(c)
        run.save_table("comparisons_poisoning", comps)
    if "adversarial" in which:
        run.save_table("summary_adversarial", agg("adversarial", ("scorer", "epsilon"), ["recall_at_5pct_fpr", "evasion_rate_among_detected", "roc_auc"]))
    run.finish()
    return run
