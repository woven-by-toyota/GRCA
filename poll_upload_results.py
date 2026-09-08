#!/usr/bin/env python3
"""
poll_upload_results.py — watches benchmark/ for new .zip files, copies them into
results/<branch> - <short-commit>/, commits, and pushes.

Pinned to the last CPU core via os.sched_setaffinity.

Usage:
    python3 poll_upload_results.py [--interval 10] [--repo .]
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


def _pin_to_core() -> None:
    os.sched_setaffinity(0, {PINNED_CORE})


def git(repo: Path, *args) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
        preexec_fn=_pin_to_core,
    )
    return result.stdout.strip()


def current_branch(repo: Path) -> str:
    try:
        return git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    except subprocess.CalledProcessError:
        return "main"


def short_commit(repo: Path) -> str:
    try:
        return git(repo, "rev-parse", "--short", "HEAD")
    except subprocess.CalledProcessError:
        return "unknown"


def commit_and_push(repo: Path, zip_path: Path, dest: Path) -> None:
    git(repo, "add", str(dest.relative_to(repo)))
    git(repo, "commit", "-m", f"results: add {zip_path.name}")
    print(f"  committed → {dest.relative_to(repo)}")
    git(repo, "stash")
    git(repo, "pull", "--rebase")
    # Only pop stash if there is a stash entry
    try:
        stash_list = git(repo, "stash", "list")
        if stash_list:
            git(repo, "stash", "pop")
    except subprocess.CalledProcessError as e:
        print(f"  [git] error during stash pop: {e.stderr.strip()}")
    try:
        git(repo, "push")
        print(f"  pushed.")
    except subprocess.CalledProcessError as e:
        print(f"  [git] push error: {e.stderr.strip()}")


def poll(repo: Path, interval: int) -> None:
    benchmark_dir = repo / "benchmark"
    seen: set[Path] = set()

    # Seed seen with zips that already exist so we don't re-commit old ones
    for z in benchmark_dir.glob("*.zip"):
        seen.add(z)
    print(f"Poller started (core {PINNED_CORE}, interval={interval}s). Watching {benchmark_dir}/")

    while True:
        time.sleep(interval)

        branch = current_branch(repo)
        commit = short_commit(repo)
        dest_dir = repo / "results" / f"{branch} - {commit}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        for zip_path in sorted(benchmark_dir.glob("*.zip")):
            if zip_path in seen:
                continue
            seen.add(zip_path)

            dest = dest_dir / zip_path.name
            print(f"New zip detected: {zip_path.name}")
            shutil.copy2(zip_path, dest)
            print(f"  copied  → {dest.relative_to(repo)}")

            try:
                commit_and_push(repo, zip_path, dest)
            except subprocess.CalledProcessError as e:
                print(f"  [git] error: {e.stderr.strip()}")


PINNED_CORE = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=10,
                        help="Poll interval in seconds (default: 10)")
    parser.add_argument("--repo", type=str, default=".",
                        help="Path to the GRCA repo root (default: .)")
    args = parser.parse_args()

    try:
        os.sched_setaffinity(0, {PINNED_CORE})
        print(f"Pinned to CPU core {PINNED_CORE}.")
    except AttributeError:
        print("Warning: os.sched_setaffinity not available on this platform.")
    except PermissionError:
        print("Warning: no permission to set CPU affinity — continuing unpinned.")

    repo = Path(args.repo).resolve()
    poll(repo, args.interval)


if __name__ == "__main__":
    main()
