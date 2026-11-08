"""Behavioral coverage for the YAML-only product and local data boundary."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from od_stack.config import ConfigError, load_config, prepare
from od_stack.frames import EarthMoonSystem, rotating_to_emi
from od_stack.run import main, run
from od_stack.sources import load_synthetic_obs
from od_stack.synthetic import generate

ROOT = Path(__file__).resolve().parents[3]


def scenario(tmp_path):
    return {"schema_version": 1,
            "determination": {"estimator": {"method": "none"}},
            "prediction": {"backend": "scipy", "end_days": 8, "points": 30},
            "output": {"directory": str(tmp_path / "results"), "plots": False}}


def test_synthetic_reproduces_original_fixture_and_seed(tmp_path):
    cfg = prepare(scenario(tmp_path))
    obs, truth = generate(cfg["simulation"], EarthMoonSystem())
    original = load_synthetic_obs(ROOT / "scenarios/halo_geocenter/data/synthetic_obs.csv", EarthMoonSystem())
    np.testing.assert_allclose(obs.t, original.t, atol=1e-8)
    np.testing.assert_allclose(obs.los, original.los, atol=2e-10, rtol=0)
    repeated, _ = generate(cfg["simulation"], EarthMoonSystem())
    np.testing.assert_array_equal(obs.los, repeated.los)
    cfg["simulation"]["seed"] += 1
    other, _ = generate(cfg["simulation"], EarthMoonSystem())
    assert not np.array_equal(obs.los, other.los)
    expected, _ = rotating_to_emi(0, np.array(cfg["simulation"]["truth"]["initial_state"]["values"]), EarthMoonSystem())
    np.testing.assert_allclose(truth.at(np.array([0.0])).r[0], expected)


def test_times_state_and_observer_come_from_yaml(tmp_path):
    c = scenario(tmp_path)
    c["simulation"] = {"truth": {"epoch_days": 2, "initial_state": {"values": [0.9, 0.1, 0.1, 0, 0.2, 0]}},
                       "observer": {"type": "fixed_emi", "position_km": [1000, 2000, 3000]},
                       "observations": {"times_days": [2, 2.1, 2.4, 2.8], "noise": {"sigma_arcsec": 0}}}
    cfg = prepare(c)
    obs, truth = generate(cfg["simulation"], EarthMoonSystem())
    np.testing.assert_allclose(obs.t / 86400, [2, 2.1, 2.4, 2.8])
    np.testing.assert_array_equal(obs.observer, np.tile([1e6, 2e6, 3e6], (4, 1)))
    expected = truth.at(obs.t).r - obs.observer
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    np.testing.assert_allclose(obs.los, expected, atol=1e-15)


@pytest.mark.parametrize("change", [
    {"simulation": {"observations": {"times_days": [0, 2, 1]}}},
    {"determination": {"iod": {"options": {"nodse": 12}}}},
    {"simulation": {"truth": {"maneuvers": [1]}}},
    {"system": {"model": "not_implemented"}},
    {"prediction": {"backend": "not_implemented"}},
    {"input": {"source": "csv"}},
    {"output": {"plots": "false"}},
])
def test_bad_or_unsupported_yaml_rejected(tmp_path, change):
    c = scenario(tmp_path)
    c.update(change)
    with pytest.raises(ConfigError):
        prepare(c)


def test_cli_rejects_simulation_overrides():
    for flag in ("--iod", "--estimation", "--propagator", "--set"):
        with pytest.raises(SystemExit) as exc:
            main([flag, "anything"])
        assert exc.value.code == 2


def test_compatibility_executable_does_not_select_iod(tmp_path, capsys):
    from collocation_iod.adapters import stack_main
    path = tmp_path / "config.yaml"
    path.write_text("schema_version: 1\ndetermination:\n  iod:\n    method: range_search\n")
    assert stack_main(["--config", str(path), "--show-config"]) == 0
    assert yaml.safe_load(capsys.readouterr().out)["iod"]["method"] == "range_search"


def test_synthetic_then_csv_without_truth_exports_and_replays(tmp_path, monkeypatch):
    monkeypatch.delenv("CISLUNAR_OUTPUT_ROOT", raising=False)
    cfg = scenario(tmp_path)
    result = run(cfg)
    assert result["evaluation_available"]
    out = tmp_path / "results"
    assert (out / "state.json").is_file()
    assert (out / "trajectory.csv").is_file()
    assert (out / "observations.csv").is_file()
    assert json.loads((out / "state.json").read_text())["frame"] == "EMI"
    # Different working directory, no truth, same measured observations.
    cfg["input"] = {"source": "csv", "observations": str(out / "observations.csv")}
    cfg["output"] = {"directory": str(tmp_path / "csv-results"), "plots": True}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(ROOT / "examples")
    reread = load_config(config_path)
    second = run(reread)
    assert not second["evaluation_available"]
    output = tmp_path / "csv-results"
    for name in ("state.json", "trajectory.csv", "summary.json", "config_used.yaml", "results.png"):
        assert (output / name).is_file()
    np.testing.assert_allclose(json.loads((out / "state.json").read_text())["position_m"],
                               json.loads((output / "state.json").read_text())["position_m"], atol=10, rtol=0)
    replay = load_config(output / "config_used.yaml")
    assert replay["iod"]["method"] == "collocation"
    assert replay["data"]["truth"] is None


def test_bad_observer_combination_fails_before_java(tmp_path, monkeypatch):
    import od_stack.run as runner
    monkeypatch.setattr(runner, "build_stages", lambda *args: pytest.fail("Must fail before constructing Orekit"))
    cfg = scenario(tmp_path)
    cfg["simulation"] = {"observer": {"type": "fixed_emi", "position_km": [10, 0, 0]}}
    cfg["determination"]["estimator"]["method"] = "orekit_bls"
    with pytest.raises(ConfigError, match="geocenter"):
        run(cfg)


def test_output_cannot_escape_bind_mount(tmp_path, monkeypatch):
    monkeypatch.setenv("CISLUNAR_OUTPUT_ROOT", str(tmp_path / "mounted"))
    with pytest.raises(ConfigError, match="mounted output"):
        run(scenario(tmp_path))


def launcher():
    spec = importlib.util.spec_from_file_location("container_launcher", ROOT / "scripts/run_container.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bind_mounts_preserve_spaces_and_commas_and_use_readonly_inputs(tmp_path):
    config = tmp_path / "my config.yaml"
    config.write_text("schema_version: 1")
    inputs = tmp_path / "input, with spaces"
    inputs.mkdir()
    outputs = tmp_path / "local output"
    cmd = launcher().command(config, inputs, outputs, "cislunar-stack:local")
    mounts = [next(csv.reader([cmd[i+1]])) for i, v in enumerate(cmd) if v == "--mount"]
    assert f"source={inputs.resolve()}" in mounts[1]
    assert "readonly" in mounts[0] and "readonly" in mounts[1]
    assert "readonly" not in mounts[2]
    assert f"source={outputs.resolve()}" in mounts[2]
    assert cmd[-2:] == ["--config", "/workspace/config.yaml"]
    assert "--iod" not in cmd and "--set" not in cmd
    assert not outputs.exists()


def test_launcher_dry_run_has_no_filesystem_side_effects(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1")
    output = tmp_path / "new-output"
    assert launcher().main(["--config", str(config), "--output-dir", str(output), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)[0][0] == "docker"
    assert not output.exists()
