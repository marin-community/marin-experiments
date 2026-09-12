# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightly end-to-end CPU canary for the marin-experiments templates.

Third rung of the nightly coverage ladder:

  1. ``repin-lockfiles.yml`` (05:23 UTC) — the locks resolve and install
     (resolution-level freshness).
  2. ``dry-run-import-fix.yml`` (06:40 UTC) — the templates import and
     construct every ``ExecutorStep`` config (import/construction drift, with
     an agent auto-fix).
  3. This canary (08:17 UTC) — each template's documented CPU smoke test
     actually RUNS end-to-end (download → tokenize → train → checkpoint) on
     the GitHub runner.

A dry run cannot catch *runtime* drift: a nightly wheel can import and
construct configs cleanly while behavior underneath has changed (e.g.
levanter's ``load_checkpoint()`` dropping its ``discover_latest`` kwarg, which
only explodes once training executes). This script closes that gap:

  1. Discover every template (directory with ``launch.py`` + ``pyproject.toml``,
     shared with ``nightly_dry_run.py``). Templates that cannot run
     accelerator-free opt out via ``[tool.marin-experiments] e2e_canary = false``
     in their ``pyproject.toml``; everything else is covered by default so a
     freshly copied template is picked up automatically.
  2. For each, run ``ACCELERATOR=cpu uv run python launch.py`` (no ``--dry_run``)
     against a throwaway ``MARIN_PREFIX``, so every step executes instead of
     hitting the executor cache.
  3. Cross-check the executor's own bookkeeping: every step's plain-text
     ``.executor_status`` under the prefix must read ``SUCCESS``.
  4. ``report`` (a separate workflow step, so it survives a ``run`` step
     timeout) files or refreshes a GitHub issue per failing template.

Unlike the dry-run workflow there is deliberately no auto-fix agent here:
import-shaped breakage was already fixed at 06:40, so what reaches this canary
is runtime behavior change, which deserves a human-read diagnosis. This
mirrors marin's canary-ferry triage split (deterministic detection, issue on
failure) scaled down to a CPU smoke test.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from nightly_dry_run import (
    REVIEWER,
    TemplateRunResult,
    discover_templates,
    repo_root,
    run_template,
)

logger = logging.getLogger("nightly_e2e_canary")

# Per-template wall-clock cap for the full pipeline. The documented CPU smoke
# baselines are <1 min (tiny-stories) and ~3 min (speech-asr); 20 minutes is
# generous headroom for a slow runner, and a hang past it is itself a failure
# worth reporting.
E2E_TIMEOUT_SECONDS = 1200

ISSUE_LABELS = ("canary", "nightly")


def e2e_enabled(template: Path) -> bool:
    """Whether a template opts into the e2e canary (default: yes)."""
    with open(template / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)
    return bool(
        pyproject.get("tool", {}).get("marin-experiments", {}).get("e2e_canary", True)
    )


def validate_statuses(output_root: Path) -> str | None:
    """Cross-check the executor's step bookkeeping after a clean exit.

    ``executor_main`` writes a plain-text ``.executor_status`` (``SUCCESS`` /
    ``FAILED`` / ...) into every step's output directory. A zero exit with
    missing or non-``SUCCESS`` statuses means the status protocol itself
    drifted — worth failing loudly on. Returns a problem description, or None.
    """
    statuses = sorted(output_root.rglob(".executor_status"))
    if not statuses:
        return f"launch.py exited 0 but no .executor_status files were written under {output_root}"
    problems = []
    for path in statuses:
        value = path.read_text().strip()
        if value != "SUCCESS":
            problems.append(f"{path.parent.relative_to(output_root)}: {value}")
    if problems:
        return "steps did not report SUCCESS:\n" + "\n".join(problems)
    return None


def run_one(template: Path, marin_prefix: Path) -> TemplateRunResult:
    """Run one template's full pipeline and validate the executor statuses."""
    result = run_template(
        template, marin_prefix, launch_args=(), timeout=E2E_TIMEOUT_SECONDS
    )
    if not result.ok:
        return result
    problem = validate_statuses(marin_prefix / template.name)
    if problem is not None:
        return TemplateRunResult(
            name=result.name,
            ok=False,
            output=f"{result.output}\n\nVALIDATION FAILED: {problem}",
        )
    return result


def cmd_run(args: argparse.Namespace) -> int:
    """Run every opted-in template e2e; write per-template logs and a results JSON."""
    root = repo_root()
    templates = discover_templates(root)
    if not templates:
        logger.error(
            "No templates found under %s (expected dirs with launch.py + pyproject.toml).",
            root,
        )
        return 1

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    with tempfile.TemporaryDirectory(prefix="e2e-canary-") as tmp:
        marin_prefix = Path(tmp)
        for template in templates:
            if not e2e_enabled(template):
                logger.info(
                    "Skipping %s ([tool.marin-experiments] e2e_canary = false)",
                    template.name,
                )
                entries.append(
                    {
                        "name": template.name,
                        "ok": True,
                        "skipped": True,
                        "output_tail": "",
                    }
                )
                continue
            result = run_one(template, marin_prefix)
            (logs_dir / f"{template.name}.log").write_text(result.output)
            entries.append(
                {
                    "name": result.name,
                    "ok": result.ok,
                    "skipped": False,
                    "output_tail": result.output_tail,
                }
            )

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({"templates": entries}, indent=2))

    for entry in entries:
        state = "SKIPPED" if entry["skipped"] else ("OK" if entry["ok"] else "FAILED")
        logger.info("  %s: %s", entry["name"], state)

    failures = [e for e in entries if not e["ok"]]
    if failures:
        logger.error(
            "%d template(s) failed the e2e canary: %s",
            len(failures),
            ", ".join(e["name"] for e in failures),
        )
        return 1
    logger.info("All templates passed the e2e canary.")
    return 0


def _run_url() -> str:
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return "(local run; no GITHUB_RUN_ID)"


def _gh(cli_args: list[str], dry_run: bool) -> subprocess.CompletedProcess | None:
    """Run a gh command, or just print it under --dry-run. Returns None when skipped."""
    cmd = ["gh", *cli_args]
    if dry_run:
        print(f"DRY-RUN: {shlex.join(cmd)}")
        return None
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def find_open_issue(title: str, dry_run: bool) -> int | None:
    """Find an open issue with exactly this title, so a broken week updates one thread."""
    proc = _gh(
        [
            "issue",
            "list",
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title",
        ],
        dry_run,
    )
    if proc is None:
        return None
    if proc.returncode != 0:
        # A failed lookup only risks a duplicate issue; keep reporting.
        logger.warning("gh issue list failed (%d): %s", proc.returncode, proc.stdout)
        return None
    for item in json.loads(proc.stdout):
        if item["title"] == title:
            return int(item["number"])
    return None


def file_issue(title: str, body: str, dry_run: bool) -> int:
    """Create the issue, or comment on the existing open one. Returns 0 on success."""
    number = find_open_issue(title, dry_run)
    if number is not None:
        logger.info("Updating existing open issue #%d: %s", number, title)
        proc = _gh(["issue", "comment", str(number), "--body", body], dry_run)
    else:
        logger.info("Creating issue: %s", title)
        label_args = [arg for label in ISSUE_LABELS for arg in ("--label", label)]
        proc = _gh(
            ["issue", "create", "--title", title, "--body", body, *label_args], dry_run
        )
        if proc is not None and proc.returncode != 0:
            # Most likely a label that does not exist in this repo; an
            # unlabelled issue beats no issue.
            logger.warning(
                "gh issue create with labels failed (%s); retrying without labels",
                proc.stdout.strip(),
            )
            proc = _gh(["issue", "create", "--title", title, "--body", body], dry_run)
    if proc is not None and proc.returncode != 0:
        logger.error("Failed to report %r: %s", title, proc.stdout)
        return 1
    return 0


def build_issue_body(name: str, output_tail: str, run_url: str) -> str:
    return f"""\
The nightly e2e canary failed for template `{name}`.

Workflow run: {run_url}

The canary runs the template's documented CPU smoke test end-to-end
(download → tokenize → train) against the latest nightly marin wheels:

    ACCELERATOR=cpu MARIN_PREFIX=<tmpdir> uv run python launch.py

The dry-run workflow auto-fixes import-level drift before this canary runs, so
this is likely **runtime** drift: behavior that only breaks once the pipeline
executes. Full logs are attached to the workflow run as the `canary-logs`
artifact.

Captured output (tail):

```
{output_tail}
```

CC @{REVIEWER}
"""


def cmd_report(args: argparse.Namespace) -> int:
    """File or refresh a GitHub issue per failing template."""
    run_url = _run_url()
    results_path = Path(args.results)
    if not results_path.is_file():
        # The run step died before writing results (e.g. a step-level timeout
        # during environment prep). Report that as its own failure so the red
        # run is still routed to an issue.
        body = (
            f"The e2e canary run failed before producing per-template results —\n"
            f"likely a step timeout or infrastructure problem, not a template failure.\n\n"
            f"Workflow run: {run_url}\n\nCC @{REVIEWER}\n"
        )
        return file_issue(
            "[canary] e2e canary run produced no results", body, args.dry_run
        )

    data = json.loads(results_path.read_text())
    failures = [entry for entry in data["templates"] if not entry["ok"]]
    if not failures:
        logger.info("No failures recorded in %s; nothing to report.", results_path)
        return 0

    status = 0
    for entry in failures:
        title = f"[canary] {entry['name']}: nightly e2e run failed"
        body = build_issue_body(entry["name"], entry["output_tail"], run_url)
        status |= file_issue(title, body, args.dry_run)
    return status


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Nightly end-to-end CPU canary for the templates"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run every opted-in template e2e on CPU."
    )
    run_parser.add_argument(
        "--results", required=True, help="Path to write the results JSON."
    )
    run_parser.add_argument(
        "--logs-dir", required=True, help="Directory to write full per-template logs."
    )
    run_parser.set_defaults(func=cmd_run)

    report_parser = subparsers.add_parser(
        "report", help="File/refresh GitHub issues for recorded failures."
    )
    report_parser.add_argument(
        "--results", required=True, help="Path to the results JSON written by `run`."
    )
    report_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gh commands instead of executing them.",
    )
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
