# marin-experiments

Standalone experiments using the [Marin](https://github.com/marin-community/marin) framework.

Each experiment lives in its own folder with an isolated `pyproject.toml` and virtual environment.

## Structure

```
hackable-speedrun/    # Example experiment
  pyproject.toml      # Depends on marin from git
  src/
  ...
```

## Getting Started

```bash
cd hackable-speedrun
uv sync
uv run python ...
```
