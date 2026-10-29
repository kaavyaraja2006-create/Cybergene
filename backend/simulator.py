"""
Simulated cloud environment.

Generates synthetic telemetry events for a small fake cloud (web, api,
database, k8s pod) -- never touches a real system. Produces both normal
background traffic and labeled synthetic attack sequences so the rest
of the pipeline (Sentinel -> Forensic -> Memory -> Evolution ->
Response -> Digital Twin) has something to work with.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

SERVICES = ["web", "api", "database", "k8s-pod", "lambda", "iam"]
NORMAL_ACTIONS = ["login", "logout", "view_profile", "search", "upload", "download"]

# Named attack "families": different surface sequences that a naive
# signature match would treat as unrelated, but that share an
# underlying behavioral shape (recon -> privilege step -> impact).
ATTACK_TEMPLATES = {
    "privilege_escalation": [
        ["login", "enumerate_users", "access_admin"],
        ["login", "search", "permission_change"],
        ["token_access", "enumerate_roles", "privilege_change"],
    ],
    "credential_abuse": [
        ["login", "login", "login", "login", "access_admin"],
        ["token_access", "token_access", "read_secret"],
    ],
    "data_exfiltration": [
        ["login", "enumerate_users", "download", "exfiltrate"],
        ["read_secret", "download", "exfiltrate"],
    ],
    "lateral_movement": [
        ["login", "lateral_move", "enumerate_roles", "lateral_move", "access_admin"],
    ],
}


@dataclass
class Event:
    event_id: str
    ts: float
    user: str
    service: str
    action: str
    resource: str
    status: str
    cpu_pct: float
    memory_pct: float
    network_egress_kbps: float
    src_port: int
    label: str = "normal"          # normal | attack
    attack_type: Optional[str] = None
    session_id: str = ""

    def to_dict(self):
        return self.__dict__


class CloudSimulator:
    """Keeps a rolling buffer of events and can emit normal or attack bursts."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.events: List[Event] = []

    def _make_event(self, action, service, label, cpu, mem, egress,
                     session_id=None, attack_type=None) -> Event:
        return Event(
            event_id=uuid.uuid4().hex[:12],
            ts=time.time(),
            user=f"user_{self.rng.randint(1, 80)}",
            service=service,
            action=action,
            resource=f"{service}:{self.rng.randint(1, 30)}",
            status="success" if self.rng.random() > 0.05 else "failed",
            cpu_pct=round(cpu, 1),
            memory_pct=round(mem, 1),
            network_egress_kbps=round(egress, 1),
            src_port=self.rng.choice([443, 8080, 22, 3306, 6443]),
            label=label,
            attack_type=attack_type,
            session_id=session_id or uuid.uuid4().hex[:8],
        )

    def emit_normal(self, n: int = 20) -> List[Event]:
        batch = []
        for _ in range(n):
            batch.append(self._make_event(
                action=self.rng.choice(NORMAL_ACTIONS),
                service=self.rng.choice(SERVICES),
                label="normal",
                cpu=self.rng.uniform(5, 35),
                mem=self.rng.uniform(10, 40),
                egress=self.rng.uniform(1, 200),
            ))
        self.events.extend(batch)
        return batch

    def emit_attack(self, attack_type: Optional[str] = None,
                     environment_service: Optional[str] = None) -> List[Event]:
        if attack_type is None:
            attack_type = self.rng.choice(list(ATTACK_TEMPLATES.keys()))
        template = self.rng.choice(ATTACK_TEMPLATES[attack_type])
        service = environment_service or self.rng.choice(["k8s-pod", "lambda", "database", "iam"])
        session_id = uuid.uuid4().hex[:8]
        batch = []
        for action in template:
            batch.append(self._make_event(
                action=action,
                service=service,
                label="attack",
                attack_type=attack_type,
                cpu=self.rng.uniform(40, 95),
                mem=self.rng.uniform(30, 90),
                egress=self.rng.uniform(500, 4500),
                session_id=session_id,
            ))
        self.events.extend(batch)
        return batch

    def recent(self, n: int = 200) -> List[Event]:
        return self.events[-n:]
