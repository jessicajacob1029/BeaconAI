# Changelog

## Phase 0: Repo setup (2026-10-21)

- Moved the old Tkinter/Flask app into `legacy/` (binaries gitignored).
- Scaffolded the `src/beaconai/` package layout, `tests/`, `config/`, and support folders.
- Added uv (Python 3.11), ruff, pytest, pre-commit, Makefile, and GitHub Actions CI.
- Added a validated config loader (`src/beaconai/config.py`) with tests.
- Metrics: none yet (no ML work in this phase).
