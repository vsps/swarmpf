# swarmpf

Read `PLAN.md` first: concept, design decisions, layout, status and next steps.

- `cargo test` runs the `sim` tests (the client has none). Keep `sim` dependency-free and deterministic.
- Match existing style: short doc comments explaining why, `cargo fmt` before committing.
- Networking is on hold unless the user says otherwise.
- Verify rendering changes with the headless `shot` binary and look at the PNGs.
