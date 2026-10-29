"""
Agent 2 -- Forensic Agent.

Given a session flagged by the Sentinel Agent, reconstructs the event
chain and extracts a full AttackDNA (network / behavioral / resource /
context genes) from it.
"""
from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np

from dna import AttackDNA, NetworkGenome, BehavioralGenome, ResourceGenome, ContextGenome
from simulator import Event


def _guess_attack_type(actions: List[str]) -> str:
    joined = " ".join(actions)
    if "exfiltrate" in joined:
        return "data_exfiltration"
    if "lateral_move" in joined:
        return "lateral_movement"
    if actions.count("login") >= 3 or actions.count("token_access") >= 2:
        return "credential_abuse"
    if "access_admin" in joined or "permission_change" in joined or "privilege_change" in joined:
        return "privilege_escalation"
    return "anomalous_behavior"


class ForensicAgent:
    def build_dna(self, session_events: List[Event]) -> AttackDNA:
        actions = [e.action for e in session_events]
        ports = sorted({e.src_port for e in session_events})
        span_seconds = max(
            0.001, (session_events[-1].ts - session_events[0].ts) if len(session_events) > 1 else 1.0
        )

        network = NetworkGenome(
            ports=ports,
            protocol="tcp",
            request_rate=round(len(session_events) / (span_seconds / 60.0 + 0.01), 1),
            unique_destinations=len({e.resource for e in session_events}),
        )

        # timing regularity: how evenly spaced the events are (1 = very
        # regular/bot-like, 0 = bursty/irregular)
        gaps = np.diff([e.ts for e in session_events]) if len(session_events) > 1 else np.array([0.0])
        regularity = 1.0 - min(1.0, float(np.std(gaps)) / (float(np.mean(gaps)) + 1e-6)) if len(gaps) else 0.0
        regularity = max(0.0, min(1.0, regularity))

        behavior = BehavioralGenome(
            api_sequence=actions,
            action_frequency=round(len(session_events) / (span_seconds / 60.0 + 0.01), 1),
            failed_logins=sum(1 for e in session_events if e.status == "failed"),
            unique_resources=len({e.resource for e in session_events}),
            timing_regularity=round(regularity, 2),
        )

        resource = ResourceGenome(
            cpu_pct=round(float(np.mean([e.cpu_pct for e in session_events])), 1),
            memory_pct=round(float(np.mean([e.memory_pct for e in session_events])), 1),
            network_egress_kbps=round(float(np.mean([e.network_egress_kbps for e in session_events])), 1),
            process_spawns=Counter(actions).get("lateral_move", 0),
        )

        service = Counter(e.service for e in session_events).most_common(1)[0][0]
        context = ContextGenome(
            cloud_provider="aws",
            environment="production",
            service=service,
            service_criticality="high" if service in ("database", "iam") else "medium",
            network_zone="internal",
            user_role="standard",
        )

        attack_type = _guess_attack_type(actions)
        return AttackDNA.new(attack_type, network, behavior, resource, context)
