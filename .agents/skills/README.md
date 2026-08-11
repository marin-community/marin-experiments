# Shared skills for marin consumer repos

Skills in this directory are written for repos that consume marin as PyPI
wheels — [marin-dna](https://github.com/Open-Athena/marin-dna),
[MarinFold](https://github.com/Open-Athena/MarinFold),
[MarinMat](https://github.com/Open-Athena/MarinMat), and anything bootstrapped
from the templates in this repo — rather than for the marin repo itself.

## Why this exists

marin's own skill tree
([`marin-community/marin` → `.agents/skills/`](https://github.com/marin-community/marin/tree/main/.agents/skills))
assumes marin's repo layout: vendored iris configs under `lib/iris/`, in-repo
tooling (`marin-mcp-babysitter`, ferry/canary scripts), Grug models. Downstream
repos have been porting those skills by hand and re-adapting the same specifics
each time (MarinFold's `babysit-job`/`babysit-zephyr`, marin-dna's
`agent-research`). This directory is the shared home for consumer-shaped
skills, so downstream repos copy instead of each re-adapting from marin.

## Layout convention

Same as marin, marin-dna, and MarinFold:

- `.agents/skills/<name>/SKILL.md` — YAML frontmatter (`name` + `description`)
  followed by the workflow.
- `.agents/skills/<name>/scripts/` — optional helper scripts, invoked with
  `uv run` and written to be run from the repo root.
- `.claude/skills` is a symlink to `../.agents/skills` so Claude Code discovers
  the skills; other harnesses read `.agents/skills/` directly or on demand.

## Consuming from a downstream repo

There is no automated distribution — the nightly automation in this repo only
repins lockfiles, and the wheels don't carry skills. Copy the skill directory,
the same way you'd copy a template:

```
git clone --depth 1 git@github.com:marin-community/marin-experiments.git /tmp/mx
cp -r /tmp/mx/.agents/skills/<name> <your-repo>/.agents/skills/
```

Skill-internal paths are relative to the repo root
(`.agents/skills/<name>/...`), so an unmodified copy keeps working. Keep the
Provenance section of the copied `SKILL.md` intact and append a line noting
your copy (repo + commit), so drift stays traceable in both directions.

## Contributing a skill

- **Write for the wheel-consumer shape.** No references to marin's repo layout
  or marin-only tooling. Iris configs come from the user or the `marin-iris`
  wheel's examples, not `lib/iris/config/`.
- **Carry provenance.** Every `SKILL.md` gets a Provenance section. When
  porting from marin, link a **commit-pinned** permalink, never `blob/main/` —
  upstream renames skills (`agent-research` became `run-research` in
  [marin#5858](https://github.com/marin-community/marin/pull/5858)) and `main`
  links rot.
- **Mark the knobs.** Values that legitimately differ per repo (cluster config
  paths, `--extra` names, example experiment dirs) go in a clearly labeled
  "adapt these" block near the top, so consumers change one place.

## Current skills

| Skill | Purpose | Origin |
|---|---|---|
| [gh-upload-asset](gh-upload-asset/SKILL.md) | Upload a local file to a per-user GitHub gist and get a stable raw URL for PR/issue markdown | shared from marin-dna, already repo-agnostic |

## Wanted — candidates for promotion

- `babysit-job` / `babysit-zephyr` — MarinFold's ports are already
  consumer-shaped (wheel-shipped iris configs, no `marin-mcp-babysitter`);
  promoting them here would de-duplicate the next port.
- `agent-research` — marin-dna's trimmed port; the upstream original was
  renamed to `run-research`, so this directory would become its canonical
  consumer-side home.
- Experiment packaging — marin-dna's `marin-experiment` skill (per-experiment
  `pyproject.toml` with marin packages in base deps), generalized beyond DNA.
