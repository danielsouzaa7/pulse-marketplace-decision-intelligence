# No credential ever reaches the repository.
#
# Two independent claims: .env is not tracked (so a key cannot be committed by
# accident), and nothing key-shaped is sitting in a tracked text file (so one
# cannot be committed on purpose either).
#
# git ls-files is the right oracle rather than walking the tree: it is exactly
# the set of files a `git push` would publish, and it costs nothing to ask.
import re
import subprocess

from pulse.config import ROOT

KEY_SHAPED = re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")

# tests/test_copilot.py raises RuntimeError("sk-ant-should-never-be-rendered")
# and then asserts "sk-ant" is absent from the rendered error, i.e. it is the
# test that PROVES a key never leaks into a message. Allowing the exact literal
# keeps that fixture; allowing the file would not -- a real key pasted into
# test_copilot.py still fails here.
ALLOWED = {"sk-ant-should-never-be-rendered"}

SCANNED_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".sql", ".json",
                    ".txt", ".cfg", ".ini", ".sh", ".example"}


def _tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.splitlines()


def test_env_is_not_tracked():
    tracked = _tracked()
    assert ".env" not in tracked
    assert ".env.example" in tracked, "the template must be tracked; the key must not"


def test_env_example_carries_no_value():
    """A template that ships a filled-in value is a leaked key with a polite name."""
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            _, _, value = line.partition("=")
            assert value.strip() == "", line


def test_no_key_shaped_strings_in_tracked_files():
    for name in _tracked():
        path = ROOT / name
        if path.suffix not in SCANNED_SUFFIXES or not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = {m for m in KEY_SHAPED.findall(text)} - ALLOWED
        assert not found, f"{name}: {found}"
