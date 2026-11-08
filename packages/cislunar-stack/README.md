# Cislunar simulation stack

Install both packages from the repository root:

```text
python -m pip install -e "./packages/cislunar-stack[orekit,test,examples]"
python -m pip install -e "./packages/collocation-iod[test]"
cislunar-stack --config scenarios/plug_and_play/config.yaml
```

Simulation choices come only from YAML. `collocation-iod-stack` is a compatibility
alias and no longer forces collocation. Legacy stage-oriented config files remain
supported. Use `--show-config` to inspect resolved defaults.

The product supports CSV input or deterministic synthetic generation. It exports
states and trajectories even without grading truth. Native Orekit needs Java and
`OREKIT_DATA`; set estimator method `none` and prediction backend `scipy` in the
versioned YAML for a Java-free run. Docker includes the runtime dependencies.

See [configuration and local file mounting](../../docs/configuration.md),
[the example YAML](../../scenarios/plug_and_play/config.yaml), and the
[detailed numerical pipeline notes](src/od_stack/README.md).
