"""
CYBERGENE Attack Lab.

A small set of *predefined, safe, deterministic* attack simulations
used to demonstrate the full detect -> DNA -> memory -> respond ->
digital twin -> defend -> outcome -> evolve loop end to end.

Hard safety rules (see project docs/ATTACK_LAB.md for the full writeup):
  - No arbitrary target. Every scenario is a fixed ID from SCENARIOS below;
    the API layer rejects anything else.
  - No shell execution, no real exploitation. Each scenario only ever
    produces synthetic telemetry `Event` objects (same shape simulator.py
    already produces) describing what a real instance of that attack
    would look like on the wire. Nothing here parses or executes
    attacker-supplied payloads against anything.
  - "Defense" actions are the same allowlisted, reversible action names
    already used by response_engine.py / digital_twin.py (rate_limit,
    isolate_pod, rotate_credentials, ...) -- simulated the same way the
    rest of CYBERGENE simulates them. No new execution surface is added.
  - The Attack Lab only ever talks to CYBERGENE's own in-memory
    orchestrator. It never opens a socket, never accepts a target host,
    and never runs on anything but 127.0.0.1.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from simulator import Event

DEMO_SERVICE = "demo-web"  # the fictional "NEXUS BANK DEMO" sandbox app


def _mk_event(rng_user, action, resource, status, cpu, mem, egress, session_id, src_port=443) -> Event:
    return Event(
        event_id=uuid.uuid4().hex[:12],
        ts=time.time(),
        user=rng_user,
        service=DEMO_SERVICE,
        action=action,
        resource=resource,
        status=status,
        cpu_pct=round(cpu, 1),
        memory_pct=round(mem, 1),
        network_egress_kbps=round(egress, 1),
        src_port=src_port,
        label="attack",
        attack_type=None,  # filled in by caller
        session_id=session_id,
    )


def _scenario_credential_abuse(session_id: str) -> List[Event]:
    """Repeated failed logins followed by a successful one -> brute force / credential abuse."""
    evs = []
    for i in range(6):
        evs.append(_mk_event("demo_user_01", "login", "auth:login", "failed", 15, 20, 40, session_id))
    evs.append(_mk_event("demo_user_01", "login", "auth:login", "success", 22, 28, 60, session_id))
    evs.append(_mk_event("demo_user_01", "access_admin", "admin:dashboard", "success", 55, 40, 900, session_id))
    return evs


def _scenario_injection(session_id: str) -> List[Event]:
    """A search request carrying a SQL-injection-shaped pattern -- string is never executed,
    only its *shape* (recognized here, not interpreted) is turned into a security event."""
    evs = [
        _mk_event("anon", "login", "auth:login", "success", 10, 15, 30, session_id),
        _mk_event("anon", "search", "demo:search?event=injection_attempt", "failed", 30, 25, 80, session_id),
        _mk_event("anon", "search", "demo:search?event=injection_attempt", "failed", 32, 26, 85, session_id),
        _mk_event("anon", "enumerate_users", "demo:search", "failed", 35, 27, 90, session_id),
    ]
    return evs


def _scenario_xss(session_id: str) -> List[Event]:
    evs = [
        _mk_event("demo_user_02", "login", "auth:login", "success", 12, 18, 35, session_id),
        _mk_event("demo_user_02", "upload", "demo:comment?event=xss_attempt", "failed", 20, 22, 45, session_id),
        _mk_event("demo_user_02", "upload", "demo:comment?event=xss_attempt", "failed", 21, 23, 46, session_id),
    ]
    return evs


def _scenario_path_traversal(session_id: str) -> List[Event]:
    evs = [
        _mk_event("demo_user_03", "login", "auth:login", "success", 10, 15, 30, session_id),
        _mk_event("demo_user_03", "download", "demo:files?event=path_traversal_attempt", "failed", 25, 20, 60, session_id),
        _mk_event("demo_user_03", "download", "demo:files?event=path_traversal_attempt", "failed", 26, 21, 62, session_id),
    ]
    return evs


def _scenario_suspicious_api(session_id: str) -> List[Event]:
    """login -> profile -> admin endpoint -> audit endpoint -> sensitive endpoint"""
    seq = [
        ("login", "auth:login", "success"),
        ("view_profile", "user:profile", "success"),
        ("access_admin", "admin:dashboard", "success"),
        ("enumerate_roles", "admin:audit-logs", "success"),
        ("read_secret", "admin:sensitive", "success"),
    ]
    return [
        _mk_event("demo_user_04", a, r, s, 45 + i * 5, 30 + i * 4, 300 + i * 100, session_id)
        for i, (a, r, s) in enumerate(seq)
    ]


def _scenario_privilege_escalation(session_id: str) -> List[Event]:
    evs = [
        _mk_event("demo_user_05", "login", "auth:login", "success", 15, 20, 40, session_id),
        _mk_event("demo_user_05", "access_admin", "admin:users?event=privilege_escalation_attempt", "failed", 40, 30, 300, session_id),
        _mk_event("demo_user_05", "permission_change", "admin:users?event=privilege_escalation_attempt", "failed", 42, 32, 320, session_id),
    ]
    return evs


def _scenario_data_exfiltration(session_id: str) -> List[Event]:
    evs = [_mk_event("demo_user_06", "login", "auth:login", "success", 15, 20, 40, session_id)]
    for i in range(6):
        evs.append(_mk_event("demo_user_06", "download", f"demo:customers/DEMO-{i:03d}", "success",
                              60 + i, 45 + i, 700 + i * 150, session_id))
    evs.append(_mk_event("demo_user_06", "exfiltrate", "demo:export", "success", 80, 60, 4200, session_id))
    return evs


def _scenario_api_abuse(session_id: str) -> List[Event]:
    return [
        _mk_event("demo_user_07", "search", "demo:api/products", "success", 30 + i, 25, 200 + i * 50, session_id)
        for i in range(10)
    ]


def _scenario_credential_stuffing(session_id: str) -> List[Event]:
    users = ["demo_user_08", "demo_user_09", "demo_user_10", "demo_user_11"]
    evs = []
    for u in users:
        evs.append(_mk_event(u, "login", "auth:login", "failed", 18, 22, 50, session_id))
    evs.append(_mk_event("demo_user_08", "login", "auth:login", "success", 20, 24, 55, session_id))
    return evs


@dataclass
class Scenario:
    id: str
    label: str
    category: str          # maps to a CYBERGENE attack_type for DNA classification
    description: str
    builder: Callable[[str], List[Event]]
    severity_hint: str = "high"


SCENARIOS: Dict[str, Scenario] = {
    "credential_abuse_lab": Scenario(
        "credential_abuse_lab", "Login Abuse / Brute Force", "credential_abuse",
        "Repeated failed logins against the demo bank's auth endpoint, followed by a breakthrough.",
        _scenario_credential_abuse,
    ),
    "injection_simulation": Scenario(
        "injection_simulation", "Injection Simulation", "privilege_escalation",
        "A search request carrying an injection-shaped pattern against /demo/search (never executed).",
        _scenario_injection,
    ),
    "xss_simulation": Scenario(
        "xss_simulation", "XSS Simulation", "anomalous_behavior",
        "A comment field submission carrying a script-injection-shaped pattern (sanitized, never rendered).",
        _scenario_xss, severity_hint="medium",
    ),
    "path_traversal_simulation": Scenario(
        "path_traversal_simulation", "Path Traversal Simulation", "anomalous_behavior",
        "A file-viewer request with a directory-traversal-shaped path against an in-memory demo file table.",
        _scenario_path_traversal, severity_hint="medium",
    ),
    "suspicious_api_access": Scenario(
        "suspicious_api_access", "Suspicious API Access", "privilege_escalation",
        "login -> profile -> admin endpoint -> audit endpoint -> sensitive endpoint, in one session.",
        _scenario_suspicious_api,
    ),
    "privilege_escalation_simulation": Scenario(
        "privilege_escalation_simulation", "Privilege Escalation Simulation", "privilege_escalation",
        "A standard demo session attempting an admin-only operation it isn't entitled to.",
        _scenario_privilege_escalation,
    ),
    "data_exfiltration_simulation": Scenario(
        "data_exfiltration_simulation", "Data Exfiltration Simulation", "data_exfiltration",
        "A burst of demo-customer record downloads followed by a bulk export, against synthetic data only.",
        _scenario_data_exfiltration,
    ),
    "api_abuse": Scenario(
        "api_abuse", "API Abuse", "anomalous_behavior",
        "High-frequency repeated requests against a single demo API endpoint.",
        _scenario_api_abuse, severity_hint="low",
    ),
    "credential_stuffing": Scenario(
        "credential_stuffing", "Credential Stuffing Simulation", "credential_abuse",
        "Failed logins across several synthetic demo usernames in quick succession.",
        _scenario_credential_stuffing,
    ),
}


def build_scenario_events(scenario_id: str, session_id: Optional[str] = None) -> List[Event]:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"Unknown or disallowed scenario id: {scenario_id!r}")
    sid = session_id or uuid.uuid4().hex[:8]
    scenario = SCENARIOS[scenario_id]
    events = scenario.builder(sid)
    for e in events:
        e.attack_type = scenario.category
    return events


def list_scenarios() -> List[dict]:
    return [
        {"id": s.id, "label": s.label, "category": s.category,
         "description": s.description, "severity_hint": s.severity_hint}
        for s in SCENARIOS.values()
    ]
