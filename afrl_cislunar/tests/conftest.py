"""Shared fixtures."""

import builtins
import os

import pytest


@pytest.fixture
def no_truth(monkeypatch):
    """Any attempt to read the truth file raises. Part B (real IOD) methods must pass under it.

    Patches open() for any path containing 'synthetic_truth', and the two readers in
    od_stack.sources, so both direct reads and the helper functions are caught."""
    from od_stack import sources

    real_open = builtins.open

    def guarded_open(file, *a, **kw):
        if isinstance(file, (str, bytes, os.PathLike)) and "synthetic_truth" in os.fsdecode(file):
            raise PermissionError(f"IOD method tried to read the truth file: {file}")
        return real_open(file, *a, **kw)

    def refuse(*a, **kw):
        raise PermissionError("IOD method tried to load the truth trajectory")

    real_read = sources._read_csv

    def guarded_read(path, *a, **kw):
        if "synthetic_truth" in str(path):
            raise PermissionError(f"IOD method tried to read the truth file: {path}")
        return real_read(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(sources, "load_synthetic_truth", refuse)
    monkeypatch.setattr(sources, "_read_csv", guarded_read)
    yield
