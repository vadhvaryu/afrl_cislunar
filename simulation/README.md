# Synthetic simulation tools

The product now generates observations directly from
[`scenarios/plug_and_play/config.yaml`](../scenarios/plug_and_play/config.yaml)
through `od_stack.synthetic`. Use `cislunar-stack --config PATH`; all generation
choices are YAML-controlled. The script described below is retained as the
original historical example and is not used by the product container.

The original observation-generator code is preserved in `obs_generator.py`.
Install `packages/cislunar-stack` first, then run from the repository root:

```text
python simulation/obs_generator.py
```

It writes observations, truth, and a plot under `outputs/synthetic_generation`.
Its original numerical settings remain at the top of the script. It is still an
executable script with module-level work, not a side-effect-free library API.
For headless execution, set `MPLBACKEND=Agg`.

Truth-based evaluation remains optional in `od_stack.results` and
`od_stack.sources`. Product outputs no longer require grading truth.
