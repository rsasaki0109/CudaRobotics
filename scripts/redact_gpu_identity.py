#!/usr/bin/env python3
"""Keep physical GPU identities out of committed evidence.

Benchmark and release tooling records the GPU model name and UUID reported by
``nvidia-smi``. Those strings identify the machine that produced the evidence,
so committed files carry a generic label and a pseudonymous UUID instead, and
drop details that reveal the generation (compute capability, memory size):

    python3 scripts/redact_gpu_identity.py
    python3 scripts/redact_gpu_identity.py --check

The pseudonym is a stable hash of the real UUID, so distinct devices stay
distinct and repeated runs on one device still bind to the same identity.
Rewriting a file changes its digest; every full or >=12-character SHA-256
reference to a rewritten file elsewhere in the tree is updated to match, and
tracked file names containing a model slug are renamed with ``git mv``.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PREFIXES = ("data/", "gif/")
DEFAULT_LABEL = "NVIDIA consumer GPU"
SLUG = "reference_gpu"
PROSE_MODEL = re.compile(
    r"(?:\bNVIDIA[\s~]+)?(?:\bGeForce[\s~]+)?\b(?:GTX|RTX)[\s~]*\d{3,4}(?:[\s~]*(?:Ti|SUPER|Super)\b)?"
)
SLUG_MODEL = re.compile(r"(?<![A-Za-z])(?:gtx|rtx)[_-]?\d{3,4}(?:[_-]?(?:ti|super))?(?![a-z0-9])")
GPU_UUID = re.compile(
    r"\bGPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
PLACEHOLDER_UUID = "GPU-00000000-0000-0000-0000-000000000000"
SHA256_PREFIX_MIN = 12
# Details that narrow the model down to a generation or a single SKU.
DETAILS = (
    (re.compile(r'("compute_capability"\s*:\s*)"\d+\.\d+"'), r'\1"redacted"'),
    (re.compile(r'("memory_total_mib"\s*:\s*)"?\d+"?'), r'\1"redacted"'),
    (re.compile(r"\b(GPU),\s*\d+\s*GB\b"), r"\1"),
    (re.compile(r"\btargeting sm_\d+"), "targeting the local GPU architecture"),
)
ARCH_LABEL = re.compile(
    r"\b(?:Kepler|Maxwell|Pascal|Volta|Turing|Ampere|Ada|Hopper|Blackwell)-class\b"
)


def tracked_files() -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return [p for p in output.decode("utf-8").split("\0") if p and not p.startswith(SKIP_PREFIXES)]


def read_text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def lf_sha256(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def pseudonym(match: re.Match[str]) -> str:
    uuid = match.group(0)
    if uuid == PLACEHOLDER_UUID:
        return uuid
    return "GPU-anon-" + hashlib.sha256(uuid.lower().encode("ascii")).hexdigest()[:12]


def redact(text: str, label: str, relabel: tuple[str, ...] = ()) -> str:
    for old in relabel:
        text = text.replace(old, label)
    text = PROSE_MODEL.sub(lambda _: label, text)
    for pattern, replacement in DETAILS:
        text = pattern.sub(replacement, text)
    if label[:1].lower() in "aeiou" or label.startswith(("NVIDIA", "RTX")):
        text = re.sub(r"\b([aA])(\s+)(?=" + re.escape(label) + ")", r"\1n\2", text)
    text = SLUG_MODEL.sub(SLUG, text)
    return GPU_UUID.sub(pseudonym, text)


def findings(path: str, text: str) -> list[str]:
    found = []
    for pattern in (PROSE_MODEL, SLUG_MODEL):
        found += [f"{path}: GPU model '{m.group(0)}'" for m in pattern.finditer(text)]
    for pattern in [p for p, _ in DETAILS] + [ARCH_LABEL]:
        found += [f"{path}: GPU detail '{m.group(0)}'" for m in pattern.finditer(text)]
    found += [
        f"{path}: GPU UUID '{m.group(0)}'"
        for m in GPU_UUID.finditer(text)
        if m.group(0) != PLACEHOLDER_UUID
    ]
    if SLUG_MODEL.search(path):
        found.append(f"{path}: GPU model in file name")
    return found


def replace_digests(text: str, digests: dict[str, str]) -> str:
    for old, new in digests.items():
        text = text.replace(old, new)
        text = re.sub(
            r"(?<![0-9a-f])" + old[:SHA256_PREFIX_MIN] + r"([0-9a-f]{0,%d})(?![0-9a-f])"
            % (64 - SHA256_PREFIX_MIN),
            lambda m: new[: SHA256_PREFIX_MIN + len(m.group(1))]
            if old.startswith(m.group(0))
            else m.group(0),
            text,
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label", default=DEFAULT_LABEL, help="public name for redacted GPUs")
    parser.add_argument(
        "--relabel", action="append", default=[], metavar="OLD",
        help="earlier label to replace with --label (repeatable)",
    )
    parser.add_argument("--check", action="store_true", help="report leaks without editing")
    args = parser.parse_args()

    texts = {}
    for path in tracked_files():
        text = read_text(ROOT / path)
        if text is not None:
            texts[path] = text

    if args.check:
        leaks = [line for path, text in texts.items() for line in findings(path, text)]
        leaks += [f"{p}: GPU model in file name" for p in tracked_files() if p not in texts and SLUG_MODEL.search(p)]
        for line in leaks:
            print(line)
        if leaks:
            print(
                "error: physical GPU identities found; run scripts/redact_gpu_identity.py --label ...",
                file=sys.stderr,
            )
        return 1 if leaks else 0

    relabel = tuple(args.relabel)
    redacted = {path: redact(text, args.label, relabel) for path, text in texts.items()}
    # Rewriting a file changes its digest, and rewriting a digest reference
    # changes the digest of the file holding it, so iterate to a fixed point.
    digests: dict[str, str] = {}
    for _ in range(16):
        edited = {
            path: replace_digests(text, digests)
            for path, text in redacted.items()
        }
        edited = {path: text for path, text in edited.items() if text != texts[path]}
        updated = {lf_sha256(texts[p]): lf_sha256(t) for p, t in edited.items()}
        if updated == digests:
            break
        digests = updated
    else:
        raise SystemExit("error: digest references did not converge")

    for path, text in edited.items():
        (ROOT / path).write_bytes(text.encode("utf-8"))
        print(f"redacted {path}")
    for path in tracked_files():
        renamed = SLUG_MODEL.sub(SLUG, path)
        if renamed != path:
            subprocess.run(["git", "mv", path, renamed], cwd=ROOT, check=True)
            print(f"renamed {path} -> {renamed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
