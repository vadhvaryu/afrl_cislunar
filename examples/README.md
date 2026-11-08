# Examples and exploratory cases

Install `./packages/cislunar-stack[examples]` from the repository root first.

| Case | Command from repository root | Output |
|---|---|---|
| Weber CR3BP | `python examples/weber_cr3bp/weber_cr3bp_example.py` | Interactive plot |
| STM sensitivity | `python examples/stm_sensitivity/stm_check.py` | `outputs/stm_sensitivity` and plot |
| Artemis comparison | `python examples/artemis_comparison/artemis2_vs_cr3bp.py` | `outputs/artemis_comparison` and plot |
| Orekit tutorials | See [tutorial README](orekit_tutorials/README.md) | Separate Java/Docker demonstrations |

Scripts resolve input/output paths independently of the calling directory.
Set `MPLBACKEND=Agg` for headless execution. The original Artemis OEM file is
preserved under `artemis_comparison/data`; the tutorial output text files remain
beside their tutorial Dockerfile. Earlier extensionless scripts and dated
meeting figures are in `archive/early_scripts` and `archive/meeting_figures`.

Standalone collocation CSV/verification examples remain with that package under
`packages/collocation-iod/examples`, so the standalone distribution retains its
example assets.
