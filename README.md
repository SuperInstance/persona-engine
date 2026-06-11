# persona-engine

Persona configuration engine for multi-agent identity management.

**Status:** Early stage — scaffolded, building, tests passing.

## What it does

Manages persona definitions — name, tone, capabilities, constraints — for
agents that need to switch or compose identities. Keeps persona config
separate from agent logic so the same agent can adopt different roles.

## Building

```sh
cargo build
cargo test
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
