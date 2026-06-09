"""
Persona Engine I2I Integration — persona-decomposed voices as fleet agents.

Personas are discoverable fleet agents that speak the I2I bottle protocol.
Any fleet agent can:
- Request persona decomposition: "Decompose this audio → persona vector"
- Request persona composition: "Render this content through persona X"
- Query available personas: "List all decomposed personas with X capability"
- Compare personas: "How similar is persona A to persona B?"

Bottle types:
    PERSONA_DECOMPOSE  → audio path → Persona
    PERSONA_COMPOSE     → content + persona → audio
    PERSONA_QUERY       → search → list of PersonaManifests
    PERSONA_COMPARE     → two persona IDs → similarity score
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from persona_engine.schemas.persona import Persona, PersonaManifest, CharacterParameters

logger = logging.getLogger(__name__)

# Persona storage directory
PERSONA_STORE = Path(os.environ.get("PERSONA_STORE", str(Path.cwd() / "memory")))
PERSONA_STORE.mkdir(parents=True, exist_ok=True)


def _persona_path(persona_id: str) -> Path:
    return PERSONA_STORE / f"{persona_id}.json"


# ------------------------------------------------------------------
# I2I Bottle Handlers
# ------------------------------------------------------------------

BOTTLE_TYPE_DECOMPOSE = "PERSONA_DECOMPOSE"
BOTTLE_TYPE_COMPOSE = "PERSONA_COMPOSE"
BOTTLE_TYPE_QUERY = "PERSONA_QUERY"
BOTTLE_TYPE_COMPARE = "PERSONA_COMPARE"
BOTTLE_TYPE_CREATE_CHARACTER = "PERSONA_CREATE_CHARACTER"


async def handle_decompose(bottle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle PERSONA_DECOMPOSE bottle.

    Payload:
        audio_path: str — path to audio file
        speaker: str — speaker name (optional)
        store: bool — whether to store the result (optional, default True)

    Returns:
        Persona manifest and persona_id
    """
    audio_path = bottle.get("payload", {}).get("audio_path", "")
    speaker = bottle.get("payload", {}).get("speaker", "unknown")
    store_result = bottle.get("payload", {}).get("store", True)

    if not audio_path or not os.path.exists(audio_path):
        return {
            "status": "error",
            "error": f"Audio file not found: {audio_path}",
        }

    from persona_engine.decompose.pipeline import DecompositionPipeline

    pipeline = DecompositionPipeline()
    import asyncio
    persona = await pipeline.decompose(audio_path, speaker=speaker)

    persona_id = persona.id
    if store_result:
        pipeline.store(persona)

    manifest = PersonaManifest(
        id=persona.id,
        name=persona.name,
        tags=persona.tags,
        capabilities=["persona_compose", "persona_query"],
        vector_dim=len(persona.persona_vector) if persona.persona_vector else 0,
        confidence=persona.confidence,
    )

    return {
        "status": "ok",
        "persona_id": persona_id,
        "manifest": manifest.model_dump(mode="json"),
        "summary": {
            "name": persona.name,
            "confidence": f"{persona.confidence:.0%}",
            "speaking_rate": f"{persona.cadence.mean_wpm:.0f} wpm",
            "groove_bpm": f"{persona.groove.conversational_bpm:.0f} bpm",
        },
    }


async def handle_compose(bottle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle PERSONA_COMPOSE bottle.

    Payload:
        content: str — text to render
        persona_id: str — ID of persona to use
        persona_name: str — name of persona (alt to ID)
        output_path: str — where to save audio (optional)

    Returns:
        Audio metadata or SSML if no TTS available
    """
    content = bottle.get("payload", {}).get("content", "")
    persona_id = bottle.get("payload", {}).get("persona_id", "")
    persona_name = bottle.get("payload", {}).get("persona_name", "")
    output_path = bottle.get("payload", {}).get("output_path", "composed.wav")

    if not content:
        return {"status": "error", "error": "No content provided"}

    # Resolve persona
    persona = _resolve_persona(persona_id, persona_name)
    if persona is None:
        return {"status": "error", "error": f"Persona not found: {persona_id or persona_name}"}

    from persona_engine.compose.engine import CompositionEngine

    engine = CompositionEngine()
    result = engine.compose(content, persona, output_path=output_path)

    return {
        "status": "ok",
        "audio_path": result["audio_path"],
        "duration": result["duration"],
        "persona_name": persona.name,
        "ssml": result.get("ssml", ""),
    }


async def handle_query(bottle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle PERSONA_QUERY bottle.

    Payload:
        query: str — search query (optional)
        tag_filter: str or list — filter by tags (optional)
        limit: int — max results (optional, default 10)

    Returns:
        List of PersonaManifests
    """
    query = bottle.get("payload", {}).get("query", "")
    tag_filter = bottle.get("payload", {}).get("tag_filter", None)
    limit = bottle.get("payload", {}).get("limit", 10)

    results = _list_personas(tag_filter=tag_filter, limit=limit)
    return {"status": "ok", "personas": results, "count": len(results)}


async def handle_compare(bottle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle PERSONA_COMPARE bottle.

    Payload:
        persona_a_id: str
        persona_b_id: str

    Returns:
        Similarity score and comparison breakdown
    """
    a_id = bottle.get("payload", {}).get("persona_a_id", "")
    b_id = bottle.get("payload", {}).get("persona_b_id", "")

    persona_a = _resolve_persona(a_id) if a_id else None
    persona_b = _resolve_persona(b_id) if b_id else None

    if not persona_a or not persona_b:
        return {"status": "error", "error": "One or both personas not found"}

    similarity = _compute_similarity(persona_a, persona_b)

    return {
        "status": "ok",
        "persona_a": {"id": a_id, "name": persona_a.name},
        "persona_b": {"id": b_id, "name": persona_b.name},
        "similarity": similarity,
        "breakdown": {
            "cadence_similarity": _field_similarity(
                persona_a.cadence.mean_wpm, persona_b.cadence.mean_wpm,
                50.0
            ),
            "prosody_similarity": _field_similarity(
                persona_a.prosody.mean_f0, persona_b.prosody.mean_f0,
                60.0
            ),
            "groove_similarity": _field_similarity(
                persona_a.groove.conversational_bpm,
                persona_b.groove.conversational_bpm,
                30.0
            ),
        },
    }


async def handle_create_character(bottle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle PERSONA_CREATE_CHARACTER bottle.

    Payload:
        character_params: dict — CharacterParameters dict
        store: bool — whether to store (optional, default True)

    Returns:
        Persona created from character parameters
    """
    params_data = bottle.get("payload", {}).get("character_params", {})
    store_result = bottle.get("payload", {}).get("store", True)

    if not params_data:
        return {"status": "error", "error": "No character params provided"}

    params = CharacterParameters(**params_data)
    persona = params.to_persona()
    persona.smp_seed = _derive_seed_from_params(params)

    if store_result:
        path = PERSONA_STORE / f"{persona.id}.json"
        path.write_text(persona.model_dump_json(indent=2))

    manifest = PersonaManifest(
        id=persona.id,
        name=persona.name,
        tags=persona.tags,
        capabilities=["persona_compose", "persona_query"],
        confidence=0.8,
    )

    return {
        "status": "ok",
        "persona_id": persona.id,
        "manifest": manifest.model_dump(mode="json"),
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _resolve_persona(
    persona_id: str = "",
    persona_name: str = ""
) -> Optional[Persona]:
    """Find a persona by ID or name."""
    # By ID
    if persona_id:
        p = _persona_path(persona_id)
        if p.exists():
            return Persona(**json.loads(p.read_text()))

    # By name (scan all)
    for f in PERSONA_STORE.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if data.get("name", "").lower() == persona_name.lower():
                return Persona(**data)
        except (json.JSONDecodeError, KeyError):
            continue

    return None


def _list_personas(
    tag_filter: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """List all stored personas with optional tag filter."""
    results = []
    for f in sorted(PERSONA_STORE.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            persona = Persona(**data)
            if tag_filter:
                tags = persona.tags
                if isinstance(tag_filter, str):
                    if tag_filter not in tags:
                        continue
                elif isinstance(tag_filter, list):
                    if not any(t in tags for t in tag_filter):
                        continue

            manifest = PersonaManifest(
                id=persona.id,
                name=persona.name,
                tags=persona.tags,
                capabilities=["persona_compose"],
                vector_dim=len(persona.persona_vector) if persona.persona_vector else 0,
                confidence=persona.confidence,
            )
            results.append(manifest.model_dump(mode="json"))
            if len(results) >= limit:
                break
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def _compute_similarity(a: Persona, b: Persona) -> float:
    """Compute overall similarity between two personas (0-1)."""
    if a.persona_vector and b.persona_vector:
        import numpy as np
        va = np.array(a.persona_vector)
        vb = np.array(b.persona_vector)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        if norm == 0:
            return 0.0
        return float(np.dot(va, vb) / norm)
    # Fallback: compare key fields
    scores = [
        _field_similarity(a.cadence.mean_wpm, b.cadence.mean_wpm, 50.0),
        _field_similarity(a.cadence.mean_pause_duration, b.cadence.mean_pause_duration, 0.5),
        _field_similarity(a.prosody.mean_f0, b.prosody.mean_f0, 60.0),
        _field_similarity(a.prosody.f0_std, b.prosody.f0_std, 30.0),
        _field_similarity(a.groove.conversational_bpm, b.groove.conversational_bpm, 30.0),
        _field_similarity(a.groove.swing_factor, b.groove.swing_factor, 0.2),
    ]
    return sum(scores) / len(scores)


def _field_similarity(val_a: float, val_b: float, tolerance: float) -> float:
    """How similar two numeric fields are (0-1)."""
    if tolerance == 0:
        return 1.0 if val_a == val_b else 0.0
    diff = abs(val_a - val_b)
    return max(0.0, 1.0 - diff / tolerance / 2)


def _derive_seed_from_params(params: CharacterParameters) -> str:
    """Derive SMP seed from character parameters."""
    import hashlib
    raw = (
        f"{params.name}:{params.speed}:{params.swing}:{params.enthusiasm}:"
        f"{params.formality}:{params.technical_density}:{params.art_style}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
