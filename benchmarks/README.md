# Benchmark work

This directory reserves a separate home for future repeatable noise, geometry,
cadence, and observation-update studies. No new benchmark coverage is implied by
the reorganization. Existing numerical tests live beside their packages; the
full-stack verification entry point is:

```text
python packages/collocation-iod/examples/verify_stack.py --run-tests
```

Its reports go to `outputs/stack`. Earlier saved runs are preserved in
`outputs/collocation-original`.
