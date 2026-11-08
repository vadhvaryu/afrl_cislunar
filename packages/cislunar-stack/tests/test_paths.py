"""Scenario loading must work after packaging, from unrelated directories."""
from pathlib import Path

from od_stack.config import load_config, prepare, set_value


def test_config_paths_are_relative_to_config_not_working_directory(tmp_path, monkeypatch):
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    config = scenario / "config.yaml"
    config.write_text("data:\n  observations: measurements.csv\n  truth: null\nresults:\n  dir: reports\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = prepare(load_config(config))
    assert Path(cfg["data"]["observations"]) == scenario / "measurements.csv"
    assert cfg["data"]["truth"] is None
    assert Path(cfg["results"]["dir"]) == scenario / "reports"
    set_value(cfg, "results.dir=cli-output")
    assert cfg["results"]["dir"] == "cli-output"


def test_bundled_scenario_paths_exist_from_any_directory(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(tmp_path)
    cfg = load_config(root / "scenarios" / "halo_geocenter" / "config.yaml")
    for key in ("observations", "truth"):
        assert Path(cfg["data"][key]).is_file()
