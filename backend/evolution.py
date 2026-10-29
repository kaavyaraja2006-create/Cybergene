"""
Agent 4 -- Evolution Agent.

Not "failure -> random mutation". Instead: analyze *why* detection or
response under-performed, identify the weak gene, generate a targeted
mutation candidate, and (optionally) recombine two related DNA into a
hybrid that represents a compound behavior.
"""
from __future__ import annotations

import copy
import random
from typing import List, Optional

from dna import AttackDNA, EvolutionMeta


class EvolutionEngine:
    # Step sizes of the targeted mutations (one "step" per failure).
    STEP_TIMING = 0.15
    STEP_CPU = 10.0

    def __init__(self, seed: Optional[int] = None, birth_penalty: float = 0.05):
        # birth_penalty: fitness deducted from a child at birth (prototype default 0.05).
        # Set to 0.0 to compare fitness without the built-in handicap.
        self.rng = random.Random(seed)
        self.birth_penalty = birth_penalty

    # ---- mutation -----------------------------------------------------
    def mutate(self, parent: AttackDNA, reason: str) -> AttackDNA:
        """Produce a new-generation DNA that targets the weakness named
        in `reason` (e.g. 'low_response_success', 'novel_action_seen')."""
        child = copy.deepcopy(parent)
        mutations = []

        if reason == "low_response_success":
            # widen the behavioral signature slightly -- the response that
            # worked for the narrow pattern didn't generalize, so treat
            # timing as less reliable and bump failed-login weight
            child.behavior.timing_regularity = round(max(0.0, child.behavior.timing_regularity - 0.15), 2)
            mutations.append("reduced_timing_confidence")
        elif reason == "novel_action_seen":
            # extend the repertoire with the next vocabulary action not yet in the sequence
            # (previously this branch only recorded a label and left the vector unchanged)
            from dna import ACTION_VOCAB
            novel = next((a for a in ACTION_VOCAB if a not in child.behavior.api_sequence), None)
            if novel:
                child.behavior.api_sequence = child.behavior.api_sequence + [novel]
            mutations.append("extended_action_repertoire")
        elif reason == "high_business_impact":
            child.resource.cpu_pct = round(max(0.0, child.resource.cpu_pct - self.STEP_CPU), 1)
            mutations.append("reduced_response_intensity")
        elif reason == "dependency_failure":
            child.network.unique_destinations = max(0, child.network.unique_destinations - 1)
            mutations.append("narrowed_dependency_scope")
        elif reason == "repeated_identical_failure":
            # no directional information: perturb ONE randomly chosen gene by one step
            from dna import ACTION_VOCAB
            which = self.rng.choice(["timing", "repertoire", "cpu", "dest"])
            sign = self.rng.choice([-1, 1])
            if which == "timing":
                child.behavior.timing_regularity = round(max(0.0, child.behavior.timing_regularity + sign * self.STEP_TIMING), 2)
            elif which == "repertoire":
                seq = child.behavior.api_sequence
                if sign > 0:
                    novel = next((a for a in ACTION_VOCAB if a not in seq), None)
                    if novel:
                        child.behavior.api_sequence = seq + [novel]
                elif len(seq) > 1:
                    child.behavior.api_sequence = seq[:-1]
            elif which == "cpu":
                child.resource.cpu_pct = round(min(100.0, max(0.0, child.resource.cpu_pct + sign * self.STEP_CPU)), 1)
            else:
                child.network.unique_destinations = max(0, child.network.unique_destinations + sign)
            mutations.append("random_gene_step:" + which + ("+" if sign > 0 else "-"))
        else:
            # generic small perturbation
            child.resource.cpu_pct = round(min(100, child.resource.cpu_pct * self.rng.uniform(0.9, 1.1)), 1)
            mutations.append("resource_jitter")

        child.dna_id = self._new_id()
        child.evolution = EvolutionMeta(
            generation=parent.evolution.generation + 1,
            parent_ids=[parent.dna_id],
            mutations=mutations,
            fitness=max(0.1, parent.evolution.fitness - self.birth_penalty),
        )
        return child

    # ---- recombination --------------------------------------------------
    def recombine(self, dna_a: AttackDNA, dna_b: AttackDNA) -> AttackDNA:
        """Combine two related threat profiles into a hybrid DNA --
        e.g. 'API enumeration' (A) + 'privilege escalation' (B) seen
        together in one incident."""
        child = copy.deepcopy(dna_a)
        child.behavior.api_sequence = dna_a.behavior.api_sequence + [
            a for a in dna_b.behavior.api_sequence if a not in dna_a.behavior.api_sequence
        ]
        child.behavior.failed_logins = max(dna_a.behavior.failed_logins, dna_b.behavior.failed_logins)
        child.attack_type = f"{dna_a.attack_type}+{dna_b.attack_type}"
        child.dna_id = self._new_id()
        child.evolution = EvolutionMeta(
            generation=max(dna_a.evolution.generation, dna_b.evolution.generation) + 1,
            parent_ids=[dna_a.dna_id, dna_b.dna_id],
            mutations=["meiotic_recombination"],
            fitness=(dna_a.evolution.fitness + dna_b.evolution.fitness) / 2,
        )
        return child

    # ---- fitness ----------------------------------------------------------
    @staticmethod
    def update_fitness(dna: AttackDNA, response_success: bool) -> float:
        delta = 0.08 if response_success else -0.12
        dna.evolution.fitness = round(max(0.0, min(1.0, dna.evolution.fitness + delta)), 2)
        return dna.evolution.fitness

    @staticmethod
    def _new_id() -> str:
        import uuid
        return "DNA-" + uuid.uuid4().hex[:10].upper()
