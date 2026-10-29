"""
Attack DNA — the core data structure of CYBERGENE.

Instead of storing "this IP attacked us", we store a structured,
multi-gene representation of *behavior* so that different-looking
attacks that share an underlying pattern can be recognized as the
same threat family.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

# Fixed vocabulary for the toy action-sequence space. In a production
# system this would be learned / open-vocabulary (e.g. an embedding
# model over raw API call names); a fixed table keeps the prototype
# fast, deterministic and easy to inspect.
ACTION_VOCAB = [
    "login", "logout", "view_profile", "search", "upload", "download",
    "enumerate_users", "enumerate_roles", "access_admin", "permission_change",
    "role_change", "privilege_change", "token_access", "delete_record",
    "read_secret", "write_secret", "lateral_move", "exfiltrate",
]
ACTION_INDEX = {a: i for i, a in enumerate(ACTION_VOCAB)}
PORT_VOCAB = [443, 8080, 22, 3306, 6443]
PROTO_VOCAB = ["tcp", "udp", "icmp"]
SERVICE_VOCAB = ["web", "api", "database", "k8s-pod", "lambda", "iam"]


@dataclass
class NetworkGenome:
    ports: List[int] = field(default_factory=list)
    protocol: str = "tcp"
    request_rate: float = 0.0          # requests / minute
    unique_destinations: int = 0


@dataclass
class BehavioralGenome:
    api_sequence: List[str] = field(default_factory=list)
    action_frequency: float = 0.0      # actions / minute
    failed_logins: int = 0
    unique_resources: int = 0
    timing_regularity: float = 0.0     # 0 = bursty/irregular, 1 = perfectly periodic


@dataclass
class ResourceGenome:
    cpu_pct: float = 0.0
    memory_pct: float = 0.0
    network_egress_kbps: float = 0.0
    process_spawns: int = 0


@dataclass
class ContextGenome:
    cloud_provider: str = "aws"
    environment: str = "production"
    service: str = "web"
    service_criticality: str = "medium"    # low | medium | high
    network_zone: str = "internal"
    user_role: str = "standard"


@dataclass
class EvolutionMeta:
    generation: int = 1
    parent_ids: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)
    fitness: float = 0.5
    created_at: float = field(default_factory=time.time)


@dataclass
class AttackDNA:
    """A complete, structured representation of one observed threat."""
    dna_id: str
    attack_type: str
    network: NetworkGenome
    behavior: BehavioralGenome
    resource: ResourceGenome
    context: ContextGenome
    evolution: EvolutionMeta

    # --- construction -----------------------------------------------
    @staticmethod
    def new(attack_type: str, network: NetworkGenome, behavior: BehavioralGenome,
            resource: ResourceGenome, context: ContextGenome,
            parent_ids: Optional[List[str]] = None, generation: int = 1) -> "AttackDNA":
        dna_id = "DNA-" + uuid.uuid4().hex[:10].upper()
        return AttackDNA(
            dna_id=dna_id,
            attack_type=attack_type,
            network=network,
            behavior=behavior,
            resource=resource,
            context=context,
            evolution=EvolutionMeta(generation=generation, parent_ids=parent_ids or []),
        )

    # --- vectorization -------------------------------------------------
    def vector(self, genomes=None) -> np.ndarray:
        """
        Turn the DNA into a fixed-length numeric vector so it can be
        compared with cosine similarity ("immune memory" lookup).
        Bag-of-actions for the behavioral gene + normalized scalars for
        the rest.
        """
        if genomes is not None:  # genome-ablation vector; default stays the legacy vector
            return self.vector_genomes(genomes)
        bag = np.zeros(len(ACTION_VOCAB), dtype=float)
        for a in self.behavior.api_sequence:
            idx = ACTION_INDEX.get(a)
            if idx is not None:
                bag[idx] += 1.0
        if bag.sum() > 0:
            bag = bag / bag.sum()

        scalars = np.array([
            min(self.network.request_rate / 200.0, 1.0),
            min(self.network.unique_destinations / 20.0, 1.0),
            min(self.behavior.action_frequency / 200.0, 1.0),
            min(self.behavior.failed_logins / 10.0, 1.0),
            min(self.behavior.unique_resources / 20.0, 1.0),
            self.behavior.timing_regularity,
            min(self.resource.cpu_pct / 100.0, 1.0),
            min(self.resource.memory_pct / 100.0, 1.0),
            min(self.resource.network_egress_kbps / 5000.0, 1.0),
            min(self.resource.process_spawns / 10.0, 1.0),
        ])
        return np.concatenate([bag, scalars])

    def vector_genomes(self, genomes) -> np.ndarray:
        """Per-genome blocks in canonical order, for the retrieval ablation.
        network: [rate, unique_dest] + port multi-hot + protocol one-hot
        behavior: bag-of-actions + 4 scalars; resource: 4 scalars
        context: service / criticality / zone one-hots; evolution: [generation/10, fitness]."""
        g = set(genomes)
        parts = []
        if "network" in g:
            ports = [1.0 if p in self.network.ports else 0.0 for p in PORT_VOCAB]
            proto = [1.0 if self.network.protocol == p else 0.0 for p in PROTO_VOCAB]
            parts.append(np.array([min(self.network.request_rate / 200.0, 1.0),
                                   min(self.network.unique_destinations / 20.0, 1.0)] + ports + proto))
        if "behavior" in g:
            bag = np.zeros(len(ACTION_VOCAB))
            for a in self.behavior.api_sequence:
                if a in ACTION_INDEX:
                    bag[ACTION_INDEX[a]] += 1.0
            if bag.sum() > 0:
                bag /= bag.sum()
            parts.append(np.concatenate([bag, [min(self.behavior.action_frequency / 200.0, 1.0),
                                               min(self.behavior.failed_logins / 10.0, 1.0),
                                               min(self.behavior.unique_resources / 20.0, 1.0),
                                               self.behavior.timing_regularity]]))
        if "resource" in g:
            parts.append(np.array([min(self.resource.cpu_pct / 100.0, 1.0), min(self.resource.memory_pct / 100.0, 1.0),
                                   min(self.resource.network_egress_kbps / 5000.0, 1.0),
                                   min(self.resource.process_spawns / 10.0, 1.0)]))
        if "context" in g:
            c = self.context
            parts.append(np.array([1.0 if c.service == s else 0.0 for s in SERVICE_VOCAB]
                                  + [1.0 if c.service_criticality == s else 0.0 for s in ("low", "medium", "high")]
                                  + [1.0 if c.network_zone == s else 0.0 for s in ("internal", "dmz", "external")]))
        if "evolution" in g:
            parts.append(np.array([min(self.evolution.generation / 10.0, 1.0), self.evolution.fitness]))
        return np.concatenate(parts) if parts else np.zeros(1)

    def fingerprint(self) -> str:
        """Short stable hash of the behavioral shape, for display only."""
        raw = ",".join(self.behavior.api_sequence)
        return hashlib.sha1(raw.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
