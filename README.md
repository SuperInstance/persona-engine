# Persona Engine — Multi-Agent Identity Management

**Persona Engine** manages agent identities — the configuration that defines how an agent presents itself, what decisions it can make autonomously, and how it interacts with other agents and humans. Each persona is a serialized profile loaded at agent startup that shapes behavior without changing the underlying agent code.

## Why It Matters

The same build agent code can serve different roles with different personas: a "Forgemaster" persona is cautious and methodical (prefers thorough testing, longer timeouts), while a "Heat-Treater" persona is aggressive and fast (shorter timeouts, retry immediately, optimize for throughput). The code is identical — only the persona differs.

This separation of code and configuration is the same insight behind Kubernetes ConfigMaps, environment-specific settings (dev/staging/prod), and the Strategy pattern. But for agents, it goes further: a persona defines not just parameters but *decision policies* — what the agent does when it encounters ambiguity, failure, or conflicting priorities.

Persona management matters more as fleets grow. With 4 agents, you configure each manually. With 155+, you need a system that creates, validates, version-controls, and distributes persona configurations across the fleet.

## How It Works

### Persona Model
```
Persona {
    name: "forgemaster",
    version: "1.2.0",
    decision_policy: DecisionPolicy {
        timeout_secs: 600,
        retry_count: 2,
        risk_tolerance: RiskTolerance::Low,
        escalation: EscalateOn::ThirdFailure,
    },
    capabilities: ["build", "test", "package"],
    communication_style: CommunicationStyle {
        verbosity: Verbosity::Concise,
        format: Format::Structured,
        channel_preference: Channel::Async,
    },
    memory_policy: MemoryPolicy {
        remember_duration: Duration::hours(24),
        share_with_peers: true,
        persist_locally: true,
    },
}
```

### Validation
Before deployment, each persona is validated:
- All required fields present
- Capabilities are a subset of the agent code's declared capabilities
- Decision policies are internally consistent (e.g., retry_count > 0 implies timeout_secs > 0)
- Version is semver-compatible with previous deployment

### Distribution
Personas are stored as TOML files in a central registry. On startup, an agent fetches its assigned persona, validates the signature (preventing tampering), and loads it into memory. Persona updates are hot-reloadable — the agent receives a new persona without restarting.

### Current State
Scaffolded. The persona model, validation rules, and TOML serialization format are designed. Implementation pending.

## Quick Start

```rust
// Intended API
use persona_engine::{Persona, DecisionPolicy};

let persona = Persona::builder("forgemaster", "1.0.0")
    .capability("build")
    .capability("test")
    .decision_policy(DecisionPolicy::cautious())
    .build();

persona.validate()?; // Returns Err if invalid
agent.load_persona(persona);
```

## API

- `Persona::builder(name, version)` — Start building a persona
- `persona.validate()` — Check all invariants
- `Persona::from_toml(toml_str)` — Deserialize from TOML config
- `persona.to_toml()` — Serialize for storage/distribution
- `DecisionPolicy::cautious()` / `::aggressive()` / `::balanced()` — Preset policies
- `MemoryPolicy` — What to remember, how long, whether to share

## Architecture Notes

Part of the [SuperInstance](https://github.com/SuperInstance) ecosystem. Personas implement the hermit crab analogy: the agent code is the crab (persistent, alive), the persona is the shell (swappable, environment-appropriate). A deep-sea crab needs a different shell than a tide-pool crab — same animal, different operating parameters. The conservation law γ + η = C is parameterized per persona: a "cautious" persona spends more γ (thorough testing) to ensure higher η (reliable output), while an "aggressive" persona accepts lower η per task but completes more tasks.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Gamma, E. et al. (1994). *Design Patterns.* — Strategy pattern for swappable behavior
- Kubernetes SIG: "ConfigMap and Secret Design." — configuration distribution patterns
- Wooldridge, M. (2009). *An Introduction to MultiAgent Systems*, Ch. 4. — agent architectures and profiles

## License

MIT
