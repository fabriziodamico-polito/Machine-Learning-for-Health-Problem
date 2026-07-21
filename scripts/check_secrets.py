"""Detect common credential formats in tracked and untracked repository files."""

from pathlib import Path
import re
import subprocess
import sys


PATTERNS = {
    "Telegram bot token": re.compile(rb"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "OpenAI API key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "Private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def files_to_check():
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
    )
    return [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]


def main():
    findings = []
    paths = files_to_check()
    for path in paths:
        if not path.is_file():
            continue
        data = path.read_bytes()
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                findings.append((str(path), label))

    for path, label in findings:
        print(f"Potential {label} found in {path}")
    if findings:
        return 1

    print(f"Secret scan passed: {len(paths)} repository files checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
