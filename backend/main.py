"""
CYBERGENE API server.

Run with:
    uvicorn main:app --reload --port 8000

Serves the JSON API the dashboard (frontend/index.html) polls, and
also serves the dashboard itself as static files so the whole demo
runs from a single command.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from orchestrator import Orchestrator

app = FastAPI(title="CYBERGENE API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator(db_path=os.environ.get("CYBERGENE_DB", "cybergene.db"))

from demo_bank import router as demo_bank_router  # noqa: E402
app.include_router(demo_bank_router, prefix="/demo", tags=["demo-bank"])


class AttackRequest(BaseModel):
    attack_type: Optional[str] = None


@app.get("/api/status")
def status():
    return orchestrator.stats()


@app.post("/api/simulate/normal")
def simulate_normal(n: int = 20):
    events = orchestrator.ingest_normal(n)
    return {"emitted": len(events), "events": [e.to_dict() for e in events]}


@app.post("/api/simulate/attack")
def simulate_attack(req: AttackRequest):
    return orchestrator.ingest_attack(req.attack_type)


@app.get("/api/attack-types")
def attack_types():
    from simulator import ATTACK_TEMPLATES
    return {"attack_types": list(ATTACK_TEMPLATES.keys())}


@app.get("/api/events")
def events(n: int = 100):
    return {"events": [e.to_dict() for e in orchestrator.sim.recent(n)]}


@app.get("/api/incidents")
def incidents():
    return {"incidents": orchestrator.memory.all_incidents()}


@app.get("/api/incidents/{incident_id}")
def incident_detail(incident_id: str):
    inc = orchestrator.incidents.get(incident_id) or orchestrator.memory.get_incident(incident_id)
    if not inc:
        raise HTTPException(404, "incident not found")
    return inc


@app.post("/api/incidents/{incident_id}/approve")
def approve(incident_id: str):
    try:
        return orchestrator.approve_incident(incident_id)
    except KeyError:
        raise HTTPException(404, "incident not found")


@app.post("/api/incidents/{incident_id}/reject")
def reject(incident_id: str):
    return orchestrator.reject_incident(incident_id)


@app.get("/api/demo/scenarios")
def demo_scenarios():
    from attack_lab import list_scenarios
    return {"scenarios": list_scenarios()}


class LabAttackRequest(BaseModel):
    scenario: str


@app.post("/api/demo/attack")
def demo_attack(req: LabAttackRequest):
    try:
        return orchestrator.ingest_lab_attack(req.scenario)
    except KeyError:
        raise HTTPException(400, f"Unknown or disallowed scenario id: {req.scenario!r}")


@app.post("/api/demo/evolution-demo")
def demo_evolution(req: LabAttackRequest):
    try:
        return orchestrator.run_evolution_demo(req.scenario)
    except KeyError:
        raise HTTPException(400, f"Unknown or disallowed scenario id: {req.scenario!r}")


@app.get("/api/incidents/{incident_id}/root-cause")
def incident_root_cause(incident_id: str):
    try:
        return orchestrator.root_cause(incident_id)
    except KeyError:
        raise HTTPException(404, "incident not found")


@app.get("/api/dna")
def dna_list():
    return {"dna": orchestrator.memory.all_dna()}


@app.get("/api/dna/{dna_id}")
def dna_detail(dna_id: str):
    d = orchestrator.memory.get_dna_raw(dna_id)
    if not d:
        raise HTTPException(404, "dna not found")
    return d


@app.get("/api/dna/{dna_id}/lineage")
def dna_lineage(dna_id: str):
    return orchestrator.memory.lineage(dna_id)


@app.get("/api/dna/{dna_id}/memory")
def dna_memory(dna_id: str):
    return orchestrator.memory.response_stats(dna_id)


@app.post("/api/reset")
def reset():
    global orchestrator
    db_path = os.environ.get("CYBERGENE_DB", "cybergene.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    orchestrator = Orchestrator(db_path=db_path)
    return {"reset": True}


# --- serve the dashboard -------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def dashboard():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
