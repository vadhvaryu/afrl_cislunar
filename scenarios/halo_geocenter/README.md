# Original halo geocenter baseline

This preserves the original stage-oriented configuration and both CSV fixtures.
Run `cislunar-stack --config scenarios/halo_geocenter/config.yaml` from the repo
root after installation and Java/Orekit setup. The YAML selects range-search IOD.
Edit the YAML to change methods; the compatibility CLI no longer selects collocation.

For the new versioned YAML interface and configurable synthetic generation, use
[the plug-and-play scenario](../plug_and_play/config.yaml) and
[configuration guide](../../docs/configuration.md). Reports no longer require
truth; this original fixture retains truth for baseline grading.
