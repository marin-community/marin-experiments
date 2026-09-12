---
name: gh-upload-asset
description: Upload a local file (screenshot, plot, data file) to a user-owned GitHub gist and get back a stable raw URL that renders inline in PR/issue comments and markdown documents. Use whenever you need to share an image or artifact in a GitHub comment, an analysis issue, a README, or any markdown doc and the file is NOT already committed to a repo. Common triggers include "attach this plot to the PR", "include the screenshot in the issue", "embed this figure in the analysis", or "I need a link for this image".
---

# gh-upload-asset

Upload local files to a per-user GitHub gist and emit URLs that GitHub renders inline.

## Provenance

Authored in Open-Athena/marin-dna
([#124](https://github.com/Open-Athena/marin-dna/pull/124); commit-pinned source:
[SKILL.md](https://github.com/Open-Athena/marin-dna/blob/19e3e2de3b633fc1febe5c7fcb394bb27f6530c9/.agents/skills/gh-upload-asset/SKILL.md),
script as of [54466a1](https://github.com/Open-Athena/marin-dna/blob/54466a11349e7b0f9affe7270fde1cdb0319222c/.agents/skills/gh-upload-asset/scripts/upload_asset.py)),
where the script was adapted from
[ryan-williams/git-helpers](https://github.com/ryan-williams/git-helpers).
Shared here unmodified as the canonical copy for marin consumer repos.

## When to use this skill

Use it when the user wants to reference a local file from GitHub-rendered markdown — a PR or issue comment, an analysis issue body, a gist, a README draft — and the file is not already tracked in a repository.

Do NOT use it for files that are already committed to a GitHub repo. Link to the file in the repo instead, so the reference stays in sync with the code.

## How it works

The script uses a single "assets" gist per user, stored in git config under `assets.gist` (falling back to the `ghpr`-compatible `pr.gist`). First invocation creates a secret gist and records the ID. Subsequent uploads push to the `assets` branch and return raw URLs of the form `https://gist.githubusercontent.com/<user>/<gist-id>/raw/<sha>/<file>` — pinned to a specific commit, so later uploads overwriting the same filename don't change what a reader sees.

## Invocation

From the repo root:

```
uv run .agents/skills/gh-upload-asset/scripts/upload_asset.py <file> [<file> ...]
```

Useful flags:

- `-a ALT` — alt text for the `markdown`/`img` output formats
- `-f {url,markdown,img,auto}` — output format. `auto` (default) picks `markdown` for images (by extension or MIME type) and `url` for everything else.
- `-g GIST_ID` — force a specific gist ID instead of reading from git config
- `-b BRANCH` — gist branch to push to (default `assets`)

The script prints one formatted URL per uploaded file to stdout. Paste that directly into the target markdown.

## Prerequisites

- `gh` CLI authenticated — confirm with `gh auth status` before first use.
- Git available on `PATH`. The script clones and pushes the gist over HTTPS using `gh auth git-credential`, so no SSH key setup is required.

## Example

User asks: "I just saved a ZRS attention plot to `/tmp/zrs_attn.png`, give me a link I can paste into the PR description."

Run:

```
uv run .agents/skills/gh-upload-asset/scripts/upload_asset.py /tmp/zrs_attn.png
```

Stdout will contain a single line like `![zrs_attn.png](https://gist.githubusercontent.com/.../raw/<sha>/zrs_attn.png)` that the user can paste directly into a PR or issue body.
