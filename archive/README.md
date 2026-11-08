# Preserved project history

No project files were intentionally deleted during the layout migration.

- `early_scripts/`: original extensionless root scripts, unchanged.
- `early-docs/`: the earlier simulation README, unchanged.
- `meeting_figures/`: original dated figures, unchanged.
- `build-artifacts/`: original collocation build/dist files, preserved locally.
- `legacy-layout/`: remaining old-directory metadata, caches, and crash log.
- `logs/`: original root JVM crash log.
- `reorganization/original-files.zip`: exact pre-edit copies of all 129 original
  project files, including ignored artifacts; excludes `.git` and `.venv`.
- `reorganization/original-sha256.json`: SHA-256 hashes of those originals.
- `reorganization/moves.json`: ordered source/destination mapping.

Check archive integrity and the existence of every relocated file:

```text
python archive/reorganization/verify_preservation.py
```

This archive retains original bytes even where working copies needed import,
path, documentation, or packaging changes. Old documentation and build artifacts
are historical, not the supported execution path.
