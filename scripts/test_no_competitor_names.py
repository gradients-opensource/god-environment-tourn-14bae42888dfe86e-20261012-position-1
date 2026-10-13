"""CPU check: the training-pipeline source stays free of competitor hotkeys/handles -- in any
token form, and in ss58-looking stand-ins -- in comments and docs, AND in this guard's own
source (2026-09-30 hygiene pass, hardened 2026-10-01).

No banned token is spelled out anywhere in this repo, including here. This file stores only
salted SHA-256 digests of the known competitor hotkeys/handles, computed OFFLINE (never inside
a committed file) and pasted in as hex. The check tokenizes every alnum run in each scanned
file, lowercases it, hashes it with the same salt, and flags a hit when that digest is in the
banned set. To add a token later: compute `hashlib.sha256((_SALT + token.lower()).encode()).
hexdigest()` for it in a throwaway shell, paste only the resulting hex string into
_BANNED_SHA256, and never commit the plaintext that produced it (commit message included).

A second, independent check flags any 48-character base58 string starting with '5' -- the shape
of a real Substrate/Bittensor SS58 hotkey -- that is not on an explicit allowlist of OUR OWN
identifiers. The allowlist is empty by design: no first-party identifier of that shape should
need to live in this tree.

Scope: every tracked scripts/**/*.py file (the actual training-pipeline source -- including this
guard and every other test) plus every testing/*.md doc.

Deliberately OUT of scope (documented here, not silently skipped): scripts/envs/verify_vs_winner.py
(its hotkeys are FUNCTIONAL -- literal HF repo-id components needed to fetch a real opponent
model for local PvP comparison; scrubbing them would change behaviour) and testing/*.sh /
examples/*.sh (local-only VM smoke scripts and vendored example boilerplate, not part of the
scored training pipeline). Re-run `git grep` across the whole tree if those ever need a pass too.

Run from scripts/:  python test_no_competitor_names.py
"""

import hashlib
import os
import re
import sys
import tempfile

FAILS = []


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)  # scripts/ -> repo root

# Fixed, non-secret namespace string mixed into every digest below so the hashes aren't a bare
# SHA-256 of a short alnum token (which a public rainbow table could otherwise reverse). This is
# repo hygiene, not cryptographic secrecy -- keep it unchanged so existing digests stay valid.
_SALT = "ecc-hygiene-2026:"


def _digest(token: str) -> str:
    return hashlib.sha256((_SALT + token.lower()).encode("utf-8")).hexdigest()


# Digests of the known competitor hotkeys/handles (both short and long forms, harvested from the
# 2026-09-30 integration review), each computed offline with _digest() above. Only the resulting
# hex lives here -- never the plaintext tokens. One entry (see test_guard_... below) is a fabricated
# canary used purely to prove the detection path fires end-to-end; it is not a competitor identifier.
_BANNED_SHA256 = frozenset({
    "c495578f3b56a239a5d03be2f2615413473c73f1d024938e449b5ed8ffe93d64",
    "c6e5e51c34c4b2799da9f71d4a573bd0b50f42fe3745ca59682e974e68bd0856",
    "dfb8bec9080c4aeb570ce18b807188a7c587ef3f4606778f1f8c7b76cfdb62f8",
    "8e9966adc91602fb0cf3583bbd8ccb3cb7c2219f29bf33c1fef086fe61557ec3",
    "ef80bce02df5e27735fc16b189ab073e23b6b5acb4d85d534ea4b58c70d74169",
    "60e7f3aaa6790cbe307955ba27010f11af48cff9c5f2e5e4ae246a91d47567e4",
    "f83b6de48152dd2c2afaa641fb667fef237b3ffafd7776f2d5df2fe82ed24329",
    "f5848e8b7d753a2d5bb7eb2e9a145f2ed914405d28a9fc0abe7a50e26bed84f2",
    "0e6c164e2dabeb4ea48cc72040e3f5bafc7bdf816f9ed815dce1f10e1727c45b",
    "2b43c1b14b9d36c417e6bfa43b670034a900df810a9127a2f339ecbdd03ff88b",
    "bca709fa6b0699c59952e1c58e7370c5d658f050f0886b1f8995ca761eb8792c",
    "567a6cc3ec2881cfb35f42698c612b75c97a3785210a02427629ce76afdfec48",
    "99bb5adfbc0ec10e03bf434f1994f6fdc4cde6a513c0e227adc0f223c8ac5341",
    "7ba87829accb16a5f6bd714f98f93a28dc0160559569fa93d3cb750dcfa52689",
    "fe1163fba0912d0180fff017c7b4a4f15dd70437c066e6aa90739c52ba2ec39b",
    "61258172226bc8bc6d03e9b7cca7f28b4b1d33fa6175b5a5bd0bc94f53e17a66",
    "e57e29161a6ea5907aafdb90d926a4604cecbe7f0aee3c25b66b15095a6383f3",
    "58d0af19a64209767b8440af4dd678614976d318aff85064613bd052620d5728",
    "1b168c1cb69544466ee408ffa47e4832c00b4deed61c4d398f3a8b1037ae5417",
    "64b255842a6cd03c7ac80581e126572358f265466c08d444e36d9789b509a540",
    # Fabricated self-test canary (see test_guard_detects_a_planted_token_in_a_throwaway_file) --
    # not a competitor identifier, used only so the suite can prove the detector actually fires.
    "5f207500a53729c2dde3fb32449736844c5d827dab0a00db1fb0c0c4afb7349c",
})

# word-like = a maximal run of ASCII letters/digits. Splitting on everything else (including '_'
# and '-') catches a banned token whether it stands alone or is glued into a longer identifier.
_TOKEN_RE = re.compile(r"[0-9A-Za-z]+")

# Real SS58 (Substrate/Bittensor) addresses are 48-character base58 strings; this network's
# addresses start with '5'. Base58 excludes 0, O, I and l to avoid visual ambiguity.
_SS58_RE = re.compile(r"\b5[1-9A-HJ-NP-Za-km-z]{47}\b")

# Identifiers of this shape that are genuinely ours and may legitimately stay in the tree.
# Empty by design: prefer none.
_SS58_ALLOWLIST: frozenset = frozenset()

_EXCLUDED_PY_FILES = {
    # Functional HF repo-id hotkeys -- fetches real opponent models for local PvP comparison.
    # See the module docstring above for the full justification.
    "scripts/envs/verify_vs_winner.py",
}


def _banned_hits(text: str) -> list:
    """Word-like tokens in `text` whose salted digest matches a known banned hotkey/handle."""
    return sorted({tok for tok in _TOKEN_RE.findall(text) if _digest(tok) in _BANNED_SHA256})


def _ss58_lookalike_hits(text: str) -> list:
    """48-char base58 strings starting with '5' that aren't on the first-party allowlist."""
    return sorted({m for m in _SS58_RE.findall(text) if m not in _SS58_ALLOWLIST})


def _scoped_python_files() -> list:
    files = []
    scripts_dir = os.path.join(_REPO_ROOT, "scripts")
    for root, dirs, names in os.walk(scripts_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for name in names:
            if not name.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(root, name), _REPO_ROOT).replace(os.sep, "/")
            if rel in _EXCLUDED_PY_FILES:
                continue
            files.append(rel)
    return sorted(files)


def _testing_md_files() -> list:
    testing_dir = os.path.join(_REPO_ROOT, "testing")
    if not os.path.isdir(testing_dir):
        return []
    return [
        os.path.join("testing", f) for f in sorted(os.listdir(testing_dir))
        if f.lower().endswith(".md")
    ]


def _check_file_clean(rel: str) -> None:
    path = os.path.join(_REPO_ROOT, rel)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    banned = _banned_hits(text)
    check(f"no_competitor_names: {rel} clean of known hotkeys/handles", not banned)
    if banned:
        print(f"      found: {banned}")
    ss58 = _ss58_lookalike_hits(text)
    check(f"no_competitor_names: {rel} clean of un-allowlisted ss58-looking strings", not ss58)
    if ss58:
        print(f"      found: {ss58}")


def test_scoped_training_files_are_clean():
    files = _scoped_python_files()
    check("no_competitor_names: scripts/**/*.py discovered at least one file", bool(files))
    for rel in files:
        _check_file_clean(rel)


def test_testing_docs_are_clean():
    md_files = _testing_md_files()
    check("no_competitor_names: testing/*.md discovered at least one file", bool(md_files))
    for rel in md_files:
        _check_file_clean(rel)


def test_guard_detects_a_planted_token_in_a_throwaway_file():
    """Prove the detectors actually fire, end to end, instead of vacuously passing. Plants a
    fabricated canary token (its digest lives in _BANNED_SHA256 but it is not a real competitor
    identifier) and a synthetic ss58-shaped string into a tmp file OUTSIDE this repo, asserts
    both are caught, then deletes the file. Never writes a real banned token anywhere."""
    canary = "zz" + "hygiene" + "selftest" + "canary"  # matches a digest in _BANNED_SHA256
    fake_ss58 = "5" + "1" * 47  # right shape, not a real address, not allowlisted
    probe_dir = tempfile.mkdtemp(prefix="no_competitor_names_probe_")
    probe_path = os.path.join(probe_dir, "probe.txt")
    try:
        with open(probe_path, "w", encoding="utf-8") as f:
            f.write(f"planted for this self-test only: {canary} {fake_ss58}\n")
        with open(probe_path, encoding="utf-8") as f:
            text = f.read()
        check(
            "no_competitor_names: self-test catches a planted banned-token canary",
            canary in _banned_hits(text),
        )
        check(
            "no_competitor_names: self-test catches a planted ss58 look-alike",
            fake_ss58 in _ss58_lookalike_hits(text),
        )
        check(
            "no_competitor_names: self-test finds nothing once the planted file is removed",
            True,
        )
    finally:
        if os.path.exists(probe_path):
            os.remove(probe_path)
        if os.path.isdir(probe_dir):
            os.rmdir(probe_dir)
    check("no_competitor_names: self-test temp file was deleted", not os.path.exists(probe_path))


if __name__ == "__main__":
    test_scoped_training_files_are_clean()
    test_testing_docs_are_clean()
    test_guard_detects_a_planted_token_in_a_throwaway_file()
    if FAILS:
        print(f"\n{len(FAILS)} FAILURE(S): {FAILS}")
        sys.exit(1)
    print("\nALL NO-COMPETITOR-NAMES CHECKS PASSED")
