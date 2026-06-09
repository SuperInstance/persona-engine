"""
Persona Engine — decompose, compose, and vibe-code personalities.

A voice is not just what someone says. It's *how* they say it.
This engine captures the how.

System:
    ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
    │  Decompose   │ ──► │  Persona     │ ──► │  Compose         │
    │  (audio →    │     │  Vector DB   │     │  (persona +      │
    │   persona)   │     │  (SurrealDB) │     │   content → TTS) │
    └─────────────┘     └──────────────┘     └─────────────────┘
                                 │
                          ┌──────┴──────┐
                          │  Character   │
                          │  Builder    │
                          │ (vibe-code) │
                          └─────────────┘

Key insight (Casey, 2026):
    "Stories are mostly characters in places with a plot. The art is rendered
     through those elements the way a jazz group plays with the elements of
     a lead sheet using their versions of a good sound."

    The character IS the content. Like Mickey Mouse is more important than
    Jack and the Beanstalk. The persona vector IS the lead sheet for a voice.
"""

__version__ = "0.1.0"
