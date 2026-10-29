"""
Response Agent + Response Risk Engine.

Generates candidate remediation actions for an attack type, scores each
one (security benefit + historical success + confidence - business
impact - dependency risk), and classifies the overall risk so that
low-risk actions can auto-execute while medium/high-risk ones require
human approval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from digital_twin import DigitalTwin, RESPONSE_PROFILES, TwinResult

CANDIDATE_ACTIONS = {
    "privilege_escalation": ["rotate_credentials", "isolate_pod", "block_ip"],
    "credential_abuse": ["rotate_credentials", "rate_limit", "block_ip"],
    "data_exfiltration": ["isolate_pod", "rotate_credentials", "disable_function"],
    "lateral_movement": ["apply_network_policy", "isolate_pod", "rollback_deployment"],
    "anomalous_behavior": ["rate_limit", "block_ip"],
}

RISK_LOW = {"rate_limit", "block_ip"}
RISK_MEDIUM = {"rotate_credentials", "isolate_pod", "apply_network_policy"}
RISK_HIGH = {"rollback_deployment", "disable_function", "shutdown_service"}


def risk_level(action: str) -> str:
    if action in RISK_HIGH:
        return "high"
    if action in RISK_MEDIUM:
        return "medium"
    return "low"


@dataclass
class ScoredResponse:
    action: str
    risk_score: float
    risk_level: str
    twin: TwinResult
    historical_success_rate: Optional[float]


class ResponseEngine:
    def __init__(self, twin: Optional[DigitalTwin] = None):
        self.twin = twin or DigitalTwin()

    def candidates_for(self, attack_type: str) -> List[str]:
        base = attack_type.split("+")[0]
        return CANDIDATE_ACTIONS.get(base, ["rate_limit", "block_ip"])

    def score_all(self, attack_type: str, service_criticality: str,
                  history: Dict[str, dict]) -> List[ScoredResponse]:
        scored = []
        for action in self.candidates_for(attack_type):
            hist = history.get(action)
            success_rate = hist["success_rate"] if hist else None
            twin_result = self.twin.simulate(action, service_criticality, success_rate)

            confidence = 0.9 if (hist and hist["tries"] >= 3) else (0.6 if hist else 0.4)
            dep_risk = RESPONSE_PROFILES.get(action, {}).get("dependency_risk", 0.3)

            score = (
                0.4 * twin_result.security_benefit
                + 0.3 * (success_rate if success_rate is not None else 0.5)
                + 0.15 * confidence
                - 0.1 * twin_result.business_impact
                - 0.05 * dep_risk
            )
            scored.append(ScoredResponse(
                action=action,
                risk_score=round(score, 3),
                risk_level=risk_level(action),
                twin=twin_result,
                historical_success_rate=success_rate,
            ))
        scored.sort(key=lambda s: s.risk_score, reverse=True)
        return scored

    def recommend(self, attack_type: str, service_criticality: str,
                   history: Dict[str, dict]) -> ScoredResponse:
        return self.score_all(attack_type, service_criticality, history)[0]
