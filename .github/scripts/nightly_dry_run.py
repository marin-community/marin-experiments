# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightly dry-run smoke test for the marin-experiments templates.

The templates here depend on marin (and its sibling libraries: levanter, fray,
haliax, iris, ...) as *library* wheels pulled from rolling GitHub releases.
When upstream reorganizes a module, the wheel still installs but the templates'
``import`` lines and config construction break — exactly the breakage that
landed marin#6007's "Fix imports broken by ... module reorganization" commit.

``repin-lockfiles.yml`` keeps each ``uv.lock`` pointing at wheels that still
exist (resolution-level freshness); it does NOT catch API/import drift, because
a lock can resolve and install cleanly while the Python API underneath has
moved. This script closes that gap:

  1. Discover every template (a directory with ``launch.py`` + ``pyproject.toml``).
  2. For each, run ``ACCELERATOR=cpu uv run python launch.py --dry_run``. The
     dry run imports the whole module graph (launch imports model/train/bpe/...)
     and constructs every ``ExecutorStep`` config without launching any job, so
     it surfaces import errors and config/API drift while touching no accelerator
     and no cloud storage.
  3. If everything dry-runs cleanly, exit 0 — nothing to do.
  4. If anything fails, hand the captured failures to Claude Code (headless),
     which fixes the import/API drift in the template source, re-verifies the
     dry run, and opens a PR with automerge armed, CC @hammer.

Deterministic detection lives here; only the open-ended "figure out where the
symbol moved and rewrite the import" judgement is delegated to the agent, and
only when there is real breakage to fix. This mirrors marin's
``nightshift_ci_tests.py`` wrapper-plus-agent split.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("nightly_dry_run")

# The agent it CCs on every PR it opens. Jeff Hammerbacher maintains these
# templates and the upstream nightshift workflows they are modelled on.
REVIEWER = "hammer"

# Per-template dry-run wall-clock cap. The dry run does no training and no
# network I/O, so a clean run is fast; a hang past this is itself a failure
# worth surfacing to the agent.
DRY_RUN_TIMEOUT_SECONDS = 600

# How much captured output to forward to the agent per failing template. Import
# tracebacks live in the tail, but draccus/marin can emit a lot of INFO logging
# first, so keep enough to include the traceback without bloating the prompt.
MAX_OUTPUT_CHARS = 8000

# Suppress Claude Code's default "Co-Authored-By: Claude" / "Generated with
# Claude Code" trailers on the commits and PRs the agent creates. AGENTS.md
# forbids self-credit, and a prose instruction alone does not reliably override
# the harness default — this setting does. Mirrors marin's nightshift scripts.
NO_SELF_CREDIT_SETTINGS = ("--settings", '{"attribution":{"commit":"","pr":""}}')


@dataclasses.dataclass(frozen=True)
class TemplateRunResult:
    """Outcome of running one template's launch.py (dry run or e2e)."""

    name: str  # template directory name, e.g. "tiny-stories"
    ok: bool
    output: str  # combined stdout+stderr, tail-truncated to MAX_OUTPUT_CHARS

    @property
    def output_tail(self) -> str:
        if len(self.output) <= MAX_OUTPUT_CHARS:
            return self.output
        return "...(truncated)...\n" + self.output[-MAX_OUTPUT_CHARS:]


def repo_root() -> Path:
    """Return the git repository root."""
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    return Path(out.strip())


def discover_templates(root: Path) -> list[Path]:
    """Find every template directory under the repo root.

    A template is any immediate subdirectory carrying both ``launch.py`` (the
    single ``executor_main`` entry point) and ``pyproject.toml`` (its own
    environment). Discovering by shape rather than a hardcoded list means a
    freshly copied template is covered automatically.
    """
    templates = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir()
        and (child / "launch.py").is_file()
        and (child / "pyproject.toml").is_file()
    ]
    return templates


def _run(
    cmd: list[str], cwd: Path, env: dict[str, str], timeout: int | None = None
) -> subprocess.CompletedProcess:
    """Run a command capturing combined stdout+stderr as text."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def prepare_env(template: Path, env: dict[str, str]) -> tuple[bool, str]:
    """Materialize the template's virtualenv from its committed lock.

    The normal path is a plain ``uv sync`` against the lock that
    ``repin-lockfiles.yml`` keeps fresh on ``main``. A failure here usually
    means the lock went stale (a rolling wheel 404'd) before repin caught it;
    recover by repinning in place so a stale lock is not misreported as an
    import error. Any lock churn from that recovery is legitimately part of the
    fix and the agent may include it in the PR.
    """
    sync = _run(["uv", "sync"], cwd=template, env=env)
    if sync.returncode == 0:
        return True, sync.stdout
    logger.warning(
        "`uv sync` failed for %s; attempting `uv lock --upgrade` recovery",
        template.name,
    )
    relock = _run(["uv", "lock", "--upgrade"], cwd=template, env=env)
    if relock.returncode != 0:
        return False, sync.stdout + "\n--- uv lock --upgrade ---\n" + relock.stdout
    resync = _run(["uv", "sync"], cwd=template, env=env)
    return (
        resync.returncode == 0,
        sync.stdout + "\n--- uv lock --upgrade ---\n" + relock.stdout + resync.stdout,
    )


def run_template(
    template: Path,
    marin_prefix: Path,
    *,
    launch_args: tuple[str, ...],
    timeout: int,
) -> TemplateRunResult:
    """Run one template's launch.py on CPU and capture the outcome.

    Shared by the nightly dry run (``launch_args=("--dry_run", "true")``) and
    the e2e canary (no extra args, so the pipeline actually executes).
    """
    # CPU keeps require_accelerator False and steers the templates onto their
    # smoke-test branches; a local MARIN_PREFIX keeps the executor's status
    # reads on the local filesystem instead of GCS.
    env = {
        **os.environ,
        "ACCELERATOR": "cpu",
        "MARIN_PREFIX": str(marin_prefix / template.name),
        # Never let a stray credential push wandb/network work during CI runs.
        "WANDB_MODE": "disabled",
    }

    prepared, prep_output = prepare_env(template, env)
    if not prepared:
        logger.error("Environment prep failed for %s", template.name)
        return TemplateRunResult(name=template.name, ok=False, output=prep_output)

    logger.info(
        "Running %s (launch args: %s)", template.name, list(launch_args) or "none"
    )
    try:
        proc = _run(
            ["uv", "run", "python", "launch.py", *launch_args],
            cwd=template,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        captured = exc.output or ""
        if isinstance(captured, bytes):
            captured = captured.decode("utf-8", errors="replace")
        return TemplateRunResult(
            name=template.name,
            ok=False,
            output=f"{captured}\n\nTIMEOUT after {timeout}s",
        )

    ok = proc.returncode == 0
    log = logger.info if ok else logger.error
    log("Run for %s exited %d", template.name, proc.returncode)
    return TemplateRunResult(name=template.name, ok=ok, output=proc.stdout)


def dry_run_template(template: Path, marin_prefix: Path) -> TemplateRunResult:
    """Dry-run one template and report whether it imports and resolves cleanly."""
    # draccus renders the bool `dry_run` field as a flag that takes an explicit
    # value, so it must be `--dry_run true`, not a bare flag.
    return run_template(
        template,
        marin_prefix,
        launch_args=("--dry_run", "true"),
        timeout=DRY_RUN_TIMEOUT_SECONDS,
    )


def build_prompt(failures: list[TemplateRunResult], branch: str) -> str:
    """Build the headless-agent prompt for fixing the failing templates."""
    sections = []
    for result in failures:
        sections.append(
            f"### Template: `{result.name}`\n\n"
            f"Command (run from `{result.name}/`):\n"
            "```\n"
            "ACCELERATOR=cpu MARIN_PREFIX=/tmp/marin/<name> uv run python launch.py --dry_run true\n"
            "```\n\n"
            "Captured output (tail):\n"
            "```\n"
            f"{result.output_tail}\n"
            "```\n"
        )
    failures_block = "\n".join(sections)
    failing_names = ", ".join(f"`{r.name}`" for r in failures)

    return f"""\
You are the marin-experiments nightly dry-run fixer.

A deterministic wrapper just ran each template's canonical smoke test
(`ACCELERATOR=cpu uv run python launch.py --dry_run`, which imports the whole
module graph and constructs every ExecutorStep config without launching a job).
These templates FAILED: {failing_names}.

Read `AGENTS.md` first and follow it. Especially: imports go at the top of the
file; there is NO backward compatibility — fix breakage by updating call sites
and imports to the new upstream locations, never by adding shims, fallbacks,
`hasattr` guards, or `try/except ImportError`. Never credit yourself: no
`Co-Authored-By: Claude` or "Generated with Claude Code" trailer in commits,
and no self-attribution in the PR description.

## Failures

{failures_block}

## Your job

For each failing template:

1. Read the captured output to find the root cause. Expect one of two upstream
   drift shapes:
   - **Moved/renamed symbol** (most common): a symbol the template imports from
     `marin`, `levanter`, `fray`, `haliax`, `iris`, or a sibling library was
     moved, renamed, or removed (this is what marin#6007 did — it moved
     `this_output_path`/`versioned` and flattened `fray.v2` into `fray`). Find
     where the symbol lives NOW and update the template's import to match.
   - **Missing dependency**: marin now imports a module the template's
     `pyproject.toml` does not pull in, so it is never installed (e.g.
     `ModuleNotFoundError: No module named 'dupekit'`). The distribution that
     provides the module is usually a marin-published package whose name differs
     from the import (the `dupekit` module is provided by the `marin-dupekit`
     distribution on PyPI). Declare that distribution in `[project].dependencies`,
     then `uv lock`. Prefer the canonical PyPI `marin-*` package over an ad-hoc
     find-links artifact of the bare module name.

2. Locate the fix. Useful moves:
   - `cd <template> && uv run python -c "import marin.execution as m; print([n for n in dir(m)])"`
   - Grep the installed package source under the template's
     `.venv/lib/python*/site-packages/` for the symbol's new home.
   - `cd <template> && uv run python -c "from <new.module> import <Symbol>"` to confirm.
   - For a missing module `X`, find the distribution that provides it: marin
     publishes these under `marin-*` names on PyPI, so try declaring `marin-X`
     and confirm with `cd <template> && uv run python -c "import X"`.

3. Re-verify: from the template directory run
   `ACCELERATOR=cpu MARIN_PREFIX=/tmp/marin/<name> WANDB_MODE=disabled uv run python launch.py --dry_run true`
   (the bool flag needs an explicit value) and confirm it now exits 0. Do not
   declare a template fixed until its dry run is green.

If a template's failure is NOT import/API drift (a genuine logic bug, a network
or infrastructure problem, or breakage that needs a real design decision), do
NOT hack around it. Fix the templates you legitimately can, and describe any you
could not fix (with the root cause) in the PR body so a human can follow up. If
you cannot fix anything, exit cleanly without opening a PR and explain why.

## Opening the PR

Only if you fixed at least one template and its dry run is green:

1. You are already on branch `{branch}` (created off the latest `origin/main`).
   Commit your changes there with a clear, plain commit message describing the
   upstream drift and the fix. NO self-credit trailer.
2. Push the branch and open a PR against `main` with:
   - Title: `[nightly] fix dry-run import/API drift`
   - A plain-text body (no marketing, no self-credit) that states, per template,
     which upstream symbol moved and how you updated the import, and notes that
     you verified each fix with `launch.py --dry_run` exiting 0. List any
     templates you could NOT fix and why.
   - End the body with a line: `CC @{REVIEWER}` so the maintainer is notified.
   - Try to add labels `agent-generated` and `nightly`
     (`gh pr edit <PR> --add-label agent-generated --add-label nightly`); if a
     label does not exist, skip it rather than failing.
3. Request review from @{REVIEWER}:
   `gh pr edit <PR> --add-reviewer {REVIEWER}` (do not fail the run if this errors).
4. Arm automerge so the verified fix lands without manual babysitting:
   `gh pr merge --auto --squash <PR>`. If automerge cannot be armed on this
   repository, leave the PR open for @{REVIEWER} rather than force-merging, and
   say so in your final summary.

Always finish with a short plain-text summary of what you changed, what you
verified, and the PR URL (or why no PR was opened).
"""


def run_agent(prompt: str, root: Path) -> int:
    """Invoke Claude Code headlessly to fix the failures and open the PR."""
    cmd = [
        "claude",
        "--model=opus",
        "--print",
        "--dangerously-skip-permissions",
        *NO_SELF_CREDIT_SETTINGS,
        "--tools=Read,Write,Edit,Glob,Grep,Bash",
        "--max-turns",
        "300",
        "--",
        prompt,
    ]
    logger.info(
        "Invoking Claude Code to fix %d failing template(s)",
        prompt.count("### Template:"),
    )
    proc = subprocess.run(cmd, cwd=root, check=False)
    return proc.returncode


def checkout_branch(root: Path, branch: str) -> None:
    """Reset a local working branch to the latest origin/main."""
    subprocess.run(["git", "fetch", "origin", "main"], cwd=root, check=True)
    subprocess.run(
        ["git", "checkout", "-B", branch, "origin/main"], cwd=root, check=True
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Nightly dry-run smoke test + import-drift fixer"
    )
    parser.add_argument(
        "--run-id",
        default="local",
        help="GitHub Actions run id, used to make the fix branch name unique.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run the dry-run checks and report, but never invoke the agent or open a PR.",
    )
    args = parser.parse_args()

    root = repo_root()
    templates = discover_templates(root)
    if not templates:
        logger.error(
            "No templates found under %s (expected dirs with launch.py + pyproject.toml).",
            root,
        )
        return 1
    logger.info(
        "Discovered %d template(s): %s",
        len(templates),
        ", ".join(t.name for t in templates),
    )

    with tempfile.TemporaryDirectory(prefix="nightly-dry-run-") as tmp:
        marin_prefix = Path(tmp)
        results = [dry_run_template(t, marin_prefix) for t in templates]

    failures = [r for r in results if not r.ok]
    for result in results:
        logger.info("  %s: %s", result.name, "OK" if result.ok else "FAILED")

    if not failures:
        logger.info(
            "All %d template(s) dry-run cleanly. Nothing to fix.", len(templates)
        )
        return 0

    logger.warning(
        "%d template(s) failed dry run: %s",
        len(failures),
        ", ".join(r.name for r in failures),
    )

    if args.check_only:
        logger.info("--check-only set; not invoking the agent.")
        # Non-zero so a manual/CI check surfaces the breakage.
        return 1

    branch = f"nightly/import-fix-{args.run_id}"
    checkout_branch(root, branch)
    prompt = build_prompt(failures, branch)
    return run_agent(prompt, root)


if __name__ == "__main__":
    sys.exit(main())
