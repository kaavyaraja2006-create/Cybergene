"""Non-circular retrieval evaluation on REAL flow records (weak labels).

Records = attack flows from a real dataset (after deduplication). Ground truth ("related incident")
= same dataset attack category (UNSW `attack_cat`, CIC `Label`). This is a WEAK label: it is assigned
by the dataset authors, independent of how our vectors are built, but (a) it is a family label, not
an expert judgement that two incidents are operationally related, and (b) two flows of one category
may be legitimately dissimilar. Both limits must be stated in the paper.

Genome mapping (explicit, auditable; saved in the manifest): real flow columns are assigned to the
Attack-DNA genomes by GENOME_COLUMNS / regexes below. The context genome can only use `service`
(public datasets have no cloud context) and the evolution genome has NO public-data analogue, so the
'all_five' configuration here is 'network+behaviour+resource+context' and is labelled as such;
the true five-genome test needs real CYBERGENE incidents.

Same stored records, same queries, same cosine metric across configurations; queries are disjoint
from the store. Use dedup: exact (preferably near) so copies of a query are not in the store.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from ..preprocess import FlowPreprocessor
from ..runlog import ExperimentRun
from ..stats import paired_comparisons, summarize
from .common import prepare
from .retrieval import l2norm, rank_metrics

GENOME_REGEX = {
    "network": [r"^(proto|state|sport|dsport|sttl|dttl|swin|dwin|init_win.*|.*flag.*|fin_.*|syn_.*|rst_.*|psh_.*|ack_.*|urg_.*|cwe_.*|ece_.*)$"],
    "behavior": [r"^(spkts|dpkts|sloss|dloss|sintpkt|dintpkt|sjit|djit|trans_depth|is_.*|ct_.*|tcprtt|synack|ackdat|rate)$",
                 r".*iat.*", r"^(fwd|bwd)_.*(packets|pkts).*", r".*packets_s$", r".*pkts_s$", r"^down_up_ratio$"],
    "resource": [r"^(dur|sbytes|dbytes|sload|dload|smeansz|dmeansz|res_bdy_len|stcpb|dtcpb)$",
                 r".*(length|len|bytes).*", r"^(active|idle)_.*", r".*packet_size.*", r".*segment_size.*", r".*bulk.*"],
    "context": [r"^service$"],
}
CONFIGS = {
    "network": ("network",),
    "network+behavior": ("network", "behavior"),
    "network+behavior+resource": ("network", "behavior", "resource"),
    "network+behavior+resource+context": ("network", "behavior", "resource", "context"),
}


def assign_genomes(sources: List[str]) -> Dict[str, np.ndarray]:
    """Each source column goes to the FIRST genome (in dict order) whose regex matches; unmatched
    columns are reported, not silently used."""
    out = {g: [] for g in GENOME_REGEX}; unmatched = []
    for i, c in enumerate(sources):
        for g, pats in GENOME_REGEX.items():
            if any(re.match(p, c) for p in pats):
                out[g].append(i); break
        else:
            unmatched.append(c)
    return {g: np.array(v, int) for g, v in out.items()}, sorted(set(unmatched))


def run_retrieval_real(cfg: Dict[str, Any], out_root="results", argv=None) -> ExperimentRun:
    data, dsinfo, dd = prepare(cfg)
    run = ExperimentRun(cfg.get("name", "retrieval_real"), cfg, out_root, argv, dsinfo, cfg["seeds"])
    run.manifest["dedup"] = dd
    n_store, n_q, min_support = cfg.get("n_store", 3000), cfg.get("n_queries", 200), cfg.get("min_category_support", 50)
    att = np.nonzero(data.y == 1)[0]
    cats = data.attack_cat[att]
    vc = {c: int((cats == c).sum()) for c in np.unique(cats)}
    keep_cats = [c for c, n in vc.items() if n >= min_support]
    run.manifest["categories"] = {"support": vc, "used": keep_cats, "min_support": min_support,
                                  "label_type": "WEAK: dataset attack category"}
    att = att[np.isin(cats, keep_cats)]
    per = defaultdict(lambda: defaultdict(dict))
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 77]))
        perm = att[rng.permutation(len(att))]
        nq = min(n_q, len(perm) // 5); ns = min(n_store, len(perm) - nq)
        q_idx, s_idx = perm[:nq], perm[nq:nq + ns]
        pre = FlowPreprocessor(cfg.get("port_mode", "onehot")).fit(data, s_idx)
        Xs, Xq = pre.transform(data, s_idx), pre.transform(data, q_idx)
        groups, unmatched = assign_genomes(pre.feature_sources)
        if seed == cfg["seeds"][0]:
            run.manifest["genome_columns"] = {g: sorted({pre.feature_sources[i] for i in idx}) for g, idx in groups.items()}
            run.manifest["unmatched_columns_excluded"] = unmatched
        cs, cq = data.attack_cat[s_idx], data.attack_cat[q_idx]
        rel = (cq[:, None] == cs[None, :]).astype(int)
        for cname, gs in CONFIGS.items():
            cols = np.concatenate([groups[g] for g in gs if len(groups[g])]) if any(len(groups[g]) for g in gs) else np.array([], int)
            if not len(cols):
                continue
            M = l2norm(Xs[:, cols]); Q = l2norm(Xq[:, cols])
            agg = defaultdict(list); lat = []
            for i in range(nq):
                t0 = time.perf_counter(); sims = M @ Q[i]; top = np.argpartition(-sims, 10)[:10]; top = top[np.argsort(-sims[top])]
                lat.append(time.perf_counter() - t0)
                m = rank_metrics(rel[i], np.argsort(-sims, kind="stable"), int(rel[i].sum()))
                for k, v in m.items():
                    agg[k].append(v)
            row = {"seed": seed, "config": cname, "dim": int(len(cols)), "n_store": ns, "n_queries": nq,
                   **{k: float(np.mean(v)) for k, v in agg.items()},
                   "latency_ms_median": 1e3 * float(np.median(lat)), "index_memory_bytes": int(M.nbytes)}
            # chance-level reference: expected P@k of random retrieval = category frequency in store
            row["P@10_random_reference"] = float(np.mean([rel[i].mean() for i in range(nq)]))
            run.add_row("per_seed", row)
            for k in ("P@1", "P@5", "P@10", "R@5", "R@10", "MRR", "nDCG@10"):
                per[k][cname][seed] = row[k]
    nb = cfg.get("n_boot", 10_000)
    summ, comps = [], []
    for k, byc in per.items():
        for c, d in byc.items():
            summ.append({"metric": k, "config": c, **summarize([d[s] for s in sorted(d)], nb), "per_seed": [d[s] for s in sorted(d)]})
        names = list(byc)
        comps += paired_comparisons(byc, k, [(names[i + 1], names[i]) for i in range(len(names) - 1)])
    run.save_table("summary", summ); run.save_table("comparisons_adjacent", comps)
    run.finish()
    return run
