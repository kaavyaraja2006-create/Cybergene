"""
Orchestrator -- wires the six agents into the complete CYBERGENE loop:

  TELEMETRY -> SENTINEL -> FORENSIC -> IMMUNE MEMORY SEARCH
      -> RESPONSE ENGINE -> DIGITAL TWIN -> (auto-execute | approval)
      -> OUTCOME -> FITNESS UPDATE -> EVOLUTION (on failure)

This is the single place that turns "a bunch of agent classes" into
the behavior described in the design doc's workflow diagram.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from simulator import CloudSimulator, Event
from sentinel_agent import SentinelAgent
from forensic_agent import ForensicAgent
from memory import ImmuneMemory, dna_from_dict
from evolution import EvolutionEngine
from response_engine import ResponseEngine
from dna import AttackDNA
from attack_lab import build_scenario_events, SCENARIOS

SIMILARITY_MATCH_THRESHOLD = 0.55
AUTO_EXECUTE_RISK_LEVELS = {"low"}


class Orchestrator:
    def __init__(self, db_path: str = "cybergene.db", seed: Optional[int] = 42):
        self.sim = CloudSimulator(seed=seed)
        self.sentinel = SentinelAgent()
        self.forensic = ForensicAgent()
        self.memory = ImmuneMemory(db_path)
        self.evolution = EvolutionEngine(seed=seed)
        self.response_engine = ResponseEngine()
        self.incidents: Dict[str, dict] = {}
        self._warm_start()

    def _warm_start(self):
        # seed the sentinel with a baseline of normal behavior
        normal = self.sim.emit_normal(60)
        self.sentinel.warm_up(normal)

    # -------------------------------------------------------------------
    def ingest_normal(self, n: int = 20) -> List[Event]:
        events = self.sim.emit_normal(n)
        self.sentinel.warm_up(events)
        return events

    def ingest_attack(self, attack_type: Optional[str] = None) -> dict:
        """Emit a synthetic attack burst and run it through the full loop.
        Returns a dict describing everything that happened, for the API/UI."""
        events = self.sim.emit_attack(attack_type)
        scores = self.sentinel.score_session(events)
        sid = events[0].session_id
        result = scores[sid]

        pipeline_log = [f"Sentinel: anomaly_score={result['anomaly_score']} is_anomaly={result['is_anomaly']}"]

        if not result["is_anomaly"]:
            return {
                "events": [e.to_dict() for e in events],
                "sentinel": result,
                "incident": None,
                "log": pipeline_log + ["No incident raised (below threshold)."],
            }

        # --- Forensic: build DNA -----------------------------------------
        dna = self.forensic.build_dna(events)
        pipeline_log.append(f"Forensic: built {dna.dna_id} ({dna.attack_type}), fingerprint={dna.fingerprint()}")

        # --- Immune memory: similarity search ------------------------------
        matches = self.memory.similarity_search(dna, top_k=5)
        matches = [(mid, sim) for mid, sim in matches if mid != dna.dna_id]
        best_match_id, best_sim = (matches[0] if matches else (None, 0.0))

        history = {}
        if best_match_id and best_sim >= SIMILARITY_MATCH_THRESHOLD:
            family_ids = [m for m, s in matches if s >= SIMILARITY_MATCH_THRESHOLD]
            history = self.memory.response_stats_for_family(family_ids)
            pipeline_log.append(
                f"Memory: matched {best_match_id} at {round(best_sim*100)}% similarity -- "
                f"reusing knowledge from {len(family_ids)} related threat(s)."
            )
        else:
            pipeline_log.append("Memory: no strong match found -- treating as a new threat family.")

        self.memory.store_dna(dna)

        # --- Response engine + digital twin ---------------------------------
        recommendation = self.response_engine.recommend(dna.attack_type, dna.context.service_criticality, history)
        pipeline_log.append(
            f"Response: recommend '{recommendation.action}' (risk={recommendation.risk_level}, "
            f"score={recommendation.risk_score}) -- {recommendation.twin.notes}"
        )

        auto = recommendation.risk_level in AUTO_EXECUTE_RISK_LEVELS
        status = "auto_executed" if auto else "pending"

        incident_id = "INC-" + uuid.uuid4().hex[:8].upper()
        incident = {
            "incident_id": incident_id,
            "dna_id": dna.dna_id,
            "attack_type": dna.attack_type,
            "anomaly_score": result["anomaly_score"],
            "matched_dna_id": best_match_id if best_sim >= SIMILARITY_MATCH_THRESHOLD else None,
            "similarity": round(best_sim, 3),
            "recommended_action": recommendation.action,
            "risk_level": recommendation.risk_level,
            "risk_score": recommendation.risk_score,
            "status": status,
            "twin": recommendation.twin.__dict__,
            "candidates": [
                {"action": r.action, "risk_score": r.risk_score, "risk_level": r.risk_level,
                 "historical_success_rate": r.historical_success_rate}
                for r in self.response_engine.score_all(dna.attack_type, dna.context.service_criticality, history)
            ],
        }
        self.memory.store_incident(incident)
        self.incidents[incident_id] = incident

        if auto:
            self._execute(incident_id, dna, recommendation.action, recommendation.twin.predicted_success)
            pipeline_log.append(f"Executed automatically (low risk). Outcome success={recommendation.twin.predicted_success}")
        else:
            pipeline_log.append("Held for human approval (medium/high risk).")

        return {
            "events": [e.to_dict() for e in events],
            "sentinel": result,
            "dna": dna.to_dict(),
            "incident": incident,
            "log": pipeline_log,
        }

    # -------------------------------------------------------------------
    def approve_incident(self, incident_id: str) -> dict:
        incident = self.incidents.get(incident_id) or self.memory.get_incident(incident_id)
        if not incident:
            raise KeyError(incident_id)
        dna_dict = self.memory.get_dna_raw(incident["dna_id"])
        dna = dna_from_dict(dna_dict)
        # re-run the twin once more at decision time as the "final check"
        rec_action = incident["recommended_action"]
        twin_result = self.response_engine.twin.simulate(rec_action, dna.context.service_criticality)
        self._execute(incident_id, dna, rec_action, twin_result.predicted_success)
        self.memory.update_incident_status(incident_id, "approved")
        return {"incident_id": incident_id, "status": "approved", "outcome_success": twin_result.predicted_success}

    def reject_incident(self, incident_id: str) -> dict:
        self.memory.update_incident_status(incident_id, "rejected")
        if incident_id in self.incidents:
            self.incidents[incident_id]["status"] = "rejected"
        return {"incident_id": incident_id, "status": "rejected"}

    def _execute(self, incident_id: str, dna: AttackDNA, action: str, success: bool):
        twin = self.response_engine.twin.simulate(action, dna.context.service_criticality)
        self.memory.record_response(dna.dna_id, action, success, twin.security_benefit, twin.business_impact)
        new_fitness = self.evolution.update_fitness(dna, success)
        self.memory.store_dna(dna)  # persist updated fitness

        if not success:
            # targeted mutation in response to the failure
            child = self.evolution.mutate(dna, reason="low_response_success")
            self.memory.store_dna(child)

        if incident_id in self.incidents:
            self.incidents[incident_id]["status"] = "auto_executed"
            self.incidents[incident_id]["outcome_success"] = success
            self.incidents[incident_id]["fitness_after"] = new_fitness
            self.memory.update_incident_status(incident_id, "auto_executed")

    # ------------------------------------------------------------------
    # ATTACK LAB -- safe, predefined, deterministic demo scenarios.
    # See attack_lab.py for the safety rules (fixed scenario IDs only,
    # no shell execution, no real exploitation, no arbitrary targets).
    # ------------------------------------------------------------------
    def ingest_lab_attack(self, scenario_id: str) -> dict:
        """Run one Attack Lab scenario through the full pipeline. Unlike
        ingest_attack(), Attack Lab incidents auto-defend regardless of
        risk level -- this is a controlled local sandbox, per the demo's
        'no approval gate' requirement -- but only ever through the same
        allowlisted response catalog the rest of CYBERGENE uses."""
        if scenario_id not in SCENARIOS:
            raise KeyError(f"Unknown scenario: {scenario_id}")
        scenario = SCENARIOS[scenario_id]

        events = build_scenario_events(scenario_id)
        return self._run_incident_pipeline(
            events, header=f"Attack Lab: '{scenario.label}' launched against the demo sandbox app.",
            source="attack_lab", scenario_id=scenario_id, scenario_label=scenario.label,
        )

    def ingest_live_event_burst(self, events: list, category: str, header: str) -> dict:
        """Used by the live demo bank app (backend/demo_bank.py): the events
        here come from *real* HTTP requests a person made against the demo
        site (a real login form submitted 6 times, a real search box, a
        real admin-route hit without the right role) -- not a scripted
        scenario. Otherwise runs through the identical pipeline."""
        for e in events:
            e.attack_type = category
        return self._run_incident_pipeline(events, header=header, source="demo_bank_live")

    def _run_incident_pipeline(self, events, header: str, source: str,
                                scenario_id=None, scenario_label=None) -> dict:
        self.sim.events.extend(events)
        sid = events[0].session_id

        scored = self.sentinel.score_session(events)
        result = dict(scored[sid])
        # Attack Lab / live-demo events are known-suspicious by construction
        # (either a fixed scenario, or a real threshold like "6 failed
        # logins" was already crossed before this was called), so they're
        # always treated as anomalous -- the model's own score is still
        # reported for transparency.
        result["is_anomaly"] = True
        result["note"] = "Treated as anomalous by design (Attack Lab / live demo threshold already crossed)."

        pipeline_log = [
            header,
            f"Sentinel: anomaly_score={result['anomaly_score']} (forced anomalous for reliable demo)",
        ]

        dna = self.forensic.build_dna(events)
        pipeline_log.append(f"Forensic: built {dna.dna_id} ({dna.attack_type}), fingerprint={dna.fingerprint()}")

        matches = self.memory.similarity_search(dna, top_k=5)
        matches = [(mid, sim) for mid, sim in matches if mid != dna.dna_id]
        best_match_id, best_sim = (matches[0] if matches else (None, 0.0))
        history = {}
        if best_match_id and best_sim >= SIMILARITY_MATCH_THRESHOLD:
            family_ids = [m for m, s in matches if s >= SIMILARITY_MATCH_THRESHOLD]
            history = self.memory.response_stats_for_family(family_ids)
            pipeline_log.append(
                f"Memory: matched {best_match_id} at {round(best_sim*100)}% similarity -- "
                f"reusing knowledge from {len(family_ids)} related threat(s)."
            )
        else:
            pipeline_log.append("Memory: no strong match -- recorded as a new attack family.")
        self.memory.store_dna(dna)

        recommendation = self.response_engine.recommend(dna.attack_type, dna.context.service_criticality, history)
        pipeline_log.append(
            f"Response Engine: recommend '{recommendation.action}' (score={recommendation.risk_score}) -- "
            f"{recommendation.twin.notes}"
        )
        pipeline_log.append(
            f"Digital Twin: predicted containment={'likely' if recommendation.twin.predicted_success else 'uncertain'}, "
            f"security_benefit={recommendation.twin.security_benefit}"
        )

        incident_id = "INC-" + uuid.uuid4().hex[:8].upper()
        incident = {
            "incident_id": incident_id,
            "dna_id": dna.dna_id,
            "attack_type": dna.attack_type,
            "anomaly_score": result["anomaly_score"],
            "matched_dna_id": best_match_id if best_sim >= SIMILARITY_MATCH_THRESHOLD else None,
            "similarity": round(best_sim, 3),
            "recommended_action": recommendation.action,
            "risk_level": recommendation.risk_level,
            "risk_score": recommendation.risk_score,
            "status": "pending",
            "source": source,
            "scenario_id": scenario_id,
            "scenario_label": scenario_label,
            "twin": recommendation.twin.__dict__,
            "candidates": [
                {"action": r.action, "risk_score": r.risk_score, "risk_level": r.risk_level,
                 "historical_success_rate": r.historical_success_rate}
                for r in self.response_engine.score_all(dna.attack_type, dna.context.service_criticality, history)
            ],
        }
        self.memory.store_incident(incident)
        self.incidents[incident_id] = incident

        # Attack Lab / live demo: auto-defend always (safe sandbox, no approval gate).
        self._execute(incident_id, dna, recommendation.action, recommendation.twin.predicted_success)
        outcome = self.incidents[incident_id]
        contained = outcome.get("outcome_success", False)
        pipeline_log.append(
            f"Defense executed automatically: {recommendation.action} -> "
            f"{'ATTACK CONTAINED' if contained else 'ATTACK CONTINUED (response failed)'}"
        )
        if not contained:
            pipeline_log.append("Evolution Agent: mutating DNA in response to the failure.")

        return {
            "events": [e.to_dict() for e in events],
            "sentinel": result,
            "dna": dna.to_dict(),
            "incident": outcome,
            "log": pipeline_log,
        }

    def run_evolution_demo(self, scenario_id: str) -> dict:
        """Presentation-safe walkthrough of response-guided evolution: run
        the same lab scenario twice. Round 1's response is recorded as a
        failure on purpose so the demo reliably shows a mutation step;
        round 2 lets the (now failure-informed) response engine pick
        again naturally -- it is not hand-picked, it falls out of the
        same scoring formula seeing one prior failed try for round 1's
        action. Clearly a guided demo, not a live containment statistic."""
        if scenario_id not in SCENARIOS:
            raise KeyError(f"Unknown scenario: {scenario_id}")
        scenario = SCENARIOS[scenario_id]
        steps = []

        # --- Round 1: forced failure -----------------------------------
        events1 = build_scenario_events(scenario_id)
        self.sim.events.extend(events1)
        dna1 = self.forensic.build_dna(events1)
        self.memory.store_dna(dna1)
        rec1 = self.response_engine.recommend(dna1.attack_type, dna1.context.service_criticality, {})
        fitness_before1 = dna1.evolution.fitness
        self.memory.record_response(dna1.dna_id, rec1.action, False, rec1.twin.security_benefit, rec1.twin.business_impact)
        fitness_after1 = self.evolution.update_fitness(dna1, False)
        self.memory.store_dna(dna1)
        child = self.evolution.mutate(dna1, reason="low_response_success")
        self.memory.store_dna(child)
        steps.append({
            "generation": dna1.evolution.generation, "dna_id": dna1.dna_id,
            "response": rec1.action, "result": "FAILED",
            "fitness_before": fitness_before1, "fitness_after": fitness_after1,
        })

        # --- Round 2: evolved DNA, response engine reconsults history ----
        history = self.memory.response_stats(dna1.dna_id)
        rec2 = self.response_engine.recommend(child.attack_type, child.context.service_criticality, history)
        fitness_before2 = child.evolution.fitness
        self.memory.record_response(child.dna_id, rec2.action, True, rec2.twin.security_benefit, rec2.twin.business_impact)
        fitness_after2 = self.evolution.update_fitness(child, True)
        self.memory.store_dna(child)
        steps.append({
            "generation": child.evolution.generation, "dna_id": child.dna_id,
            "response": rec2.action, "result": "SUCCESS",
            "fitness_before": fitness_before2, "fitness_after": fitness_after2,
            "mutations": child.evolution.mutations, "parent": dna1.dna_id,
        })

        return {"scenario": scenario.label, "steps": steps}

    # ------------------------------------------------------------------
    # ATTACK INVESTIGATION -- causal timeline + root-cause hypothesis.
    # ------------------------------------------------------------------
    def root_cause(self, incident_id: str) -> dict:
        incident = self.incidents.get(incident_id) or self.memory.get_incident(incident_id)
        if not incident:
            raise KeyError(incident_id)
        dna = self.memory.get_dna_raw(incident["dna_id"])
        actions = dna["behavior"]["api_sequence"]
        timeline = [
            {"step": i + 1, "action": a, "offset_seconds": round(i * 0.6, 1)}
            for i, a in enumerate(actions)
        ]
        root_cause_text, confidence = self._infer_root_cause(dna)
        return {
            "incident_id": incident_id,
            "what": f"{dna['attack_type'].replace('_', ' ').title()} detected against {dna['context']['service']}",
            "where": f"{dna['context']['service']} ({dna['context']['environment']}, {dna['context']['network_zone']} zone)",
            "how": " -> ".join(actions) if actions else "n/a",
            "why": root_cause_text,
            "affected": f"{dna['behavior']['unique_resources']} distinct resource(s)",
            "response": f"{incident.get('recommended_action', 'n/a')} "
                        f"({'contained' if incident.get('outcome_success') else 'pending or unsuccessful'})",
            "confidence": confidence,
            "timeline": timeline,
        }

    @staticmethod
    def _infer_root_cause(dna: dict):
        b = dna["behavior"]
        seq = b["api_sequence"]
        if b["failed_logins"] >= 3:
            return "Suspected root cause: abnormal authentication behavior -- repeated failed logins preceding a breakthrough.", 0.87
        if "exfiltrate" in seq or seq.count("download") >= 3:
            return "Suspected root cause: abnormal data access volume relative to this session's normal pattern.", 0.8
        if "access_admin" in seq or "permission_change" in seq or "privilege_change" in seq:
            return "Suspected root cause: privilege-sensitive endpoint accessed outside the user's typical behavior.", 0.75
        return "Suspected root cause: anomalous request pattern relative to the learned normal-traffic baseline.", 0.6

    # -------------------------------------------------------------------
    def stats(self) -> dict:
        all_dna = self.memory.all_dna()
        incidents = self.memory.all_incidents()
        by_status = {}
        for inc in incidents:
            by_status[inc["status"]] = by_status.get(inc["status"], 0) + 1
        avg_fitness = (
            round(sum(d["evolution"]["fitness"] for d in all_dna) / len(all_dna), 3) if all_dna else 0.0
        )
        max_generation = max((d["evolution"]["generation"] for d in all_dna), default=1)
        return {
            "total_events": len(self.sim.events),
            "total_dna": len(all_dna),
            "total_incidents": len(incidents),
            "incidents_by_status": by_status,
            "avg_fitness": avg_fitness,
            "max_generation": max_generation,
        }
