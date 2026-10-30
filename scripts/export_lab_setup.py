#!/usr/bin/env python3
"""Save local LeRobot changes, including untracked source, for another machine."""

from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "lerobot"
DEST = PROJECT / "patches"


def git(*args, allowed=(0,)):
    result = subprocess.run(
        ["git", "-C", str(SOURCE), *args], capture_output=True, check=False
    )
    if result.returncode not in allowed:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def main():
    commit = git("rev-parse", "HEAD")
    patch = git("diff", "--binary", "HEAD", "--", "src", "tests")
    for filename in git(
        "ls-files", "--others", "--exclude-standard", "-z", "--", "src", "tests"
    ).split(b"\0"):
        if filename:
            patch += git(
                "diff", "--no-index", "--binary", "--", "/dev/null",
                filename.decode(), allowed=(0, 1),
            )
    # Do not leave a partial export if a git command above fails.
    DEST.mkdir(exist_ok=True)
    (DEST / "lerobot-base-commit.txt").write_bytes(commit)
    (DEST / "lerobot-local.patch").write_bytes(patch)
    print(f"Saved LeRobot {commit.decode().strip()} and local changes to {DEST}")
    print("Include patches/, calibration/ and your camera/recording scripts in the commit.")


if __name__ == "__main__":
    main()
