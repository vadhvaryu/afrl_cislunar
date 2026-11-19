# Benchmark work

This directory holds repeatable noise, geometry, cadence, and observation-update
studies.

| Study | What it answers |
|---|---|
| [`collocation_iod/`](collocation_iod/RESULTS.md) | Does the collocation IOD work in the stack, and how does it compare with range search, the two-body methods and the Part B methods across arc lengths and noise levels? Follow-up: [IMPROVEMENTS.md](collocation_iod/IMPROVEMENTS.md), changes made from those results (polished range search first, ambiguity flags, honest short-arc uncertainty, light time) and their before/after numbers. | Existing numerical tests live beside their packages; the
| [`maneuver_bls/`](maneuver_bls/RESULTS.md) | Batch least squares with burn parameters: can collocation's impulsive mode plus least squares that fits each burn recover a maneuvering orbit, and how accurately are the burns found? |
full-stack verification entry point is:

```text
python packages/collocation-iod/examples/verify_stack.py --run-tests
```

Its reports go to `outputs/stack`. Earlier saved runs are preserved in
`outputs/collocation-original`.
