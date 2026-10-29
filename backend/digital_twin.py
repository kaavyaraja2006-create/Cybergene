"""
Digital Twin -- validates a candidate response in simulation before it
ever touches the (simulated) real environment.

For the prototype this is a probabilistic model rather than an actual
container/K8s sandbox: each response action has a base security-benefit
and business-impact distribution, adjusted by the target service's
criticality and by how well that action has historically performed
against this DNA / DNA family. This keeps the "don't execute blindly"
principle intact without requiring a live cluster.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


# Base characteristics of each candidate response, before context adjustment.
RESPONSE_PROFILES = {
    "rate_limit":          {"security_benefit": 0.45, "business_impact": 0.05, "dependency_risk": 0.05},
    "block_ip":            {"security_benefit": 0.60, "business_impact": 0.10, "dependency_risk": 0.10},
    "rotate_credentials":  {"security_benefit": 0.80, "business_impact": 0.20, "dependency_risk": 0.25},
    "isolate_pod":         {"security_benefit": 0.90, "business_impact": 0.30, "dependency_risk": 0.30},
    "apply_network_policy": {"security_benefit": 0.75, "business_impact": 0.15, "dependency_risk": 0.20},
    "rollback_deployment": {"security_benefit": 0.85, "business_impact": 0.70, "dependency_risk": 0.65},
    "disable_function":    {"security_benefit": 0.88, "business_impact": 0.40, "dependency_risk": 0.35},
    "shutdown_service":    {"security_benefit": 0.97, "business_impact": 0.90, "dependency_risk": 0.85},
}

CRITICALITY_MULTIPLIER = {"low": 0.8, "medium": 1.0, "high": 1.25}


@dataclass
class TwinResult:
    action: str
    security_benefit: float
    business_impact: float
    predicted_success: bool
    downtime_seconds: float
    notes: str


class DigitalTwin:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def simulate(self, action: str, service_criticality: str,
                 historical_success_rate: Optional[float] = None) -> TwinResult:
        profile = RESPONSE_PROFILES.get(action, {"security_benefit": 0.5, "business_impact": 0.3, "dependency_risk": 0.3})
        mult = CRITICALITY_MULTIPLIER.get(service_criticality, 1.0)

        benefit = min(1.0, profile["security_benefit"] * self.rng.uniform(0.92, 1.08))
        impact = min(1.0, profile["business_impact"] * mult * self.rng.uniform(0.9, 1.15))

        if historical_success_rate is not None:
            # blend the model's own view with what has actually worked before
            benefit = round(0.6 * benefit + 0.4 * historical_success_rate, 2)

        predicted_success = self.rng.random() < benefit
        downtime = round(impact * self.rng.uniform(0, 45), 1)  # seconds, toy model

        notes = (
            f"Simulated against a {service_criticality}-criticality service. "
            f"Predicted containment {'likely' if predicted_success else 'uncertain'}, "
            f"est. {downtime}s service disruption."
        )
        return TwinResult(
            action=action,
            security_benefit=round(benefit, 2),
            business_impact=round(impact, 2),
            predicted_success=predicted_success,
            downtime_seconds=downtime,
            notes=notes,
        )
