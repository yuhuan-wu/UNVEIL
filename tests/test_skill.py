"""docs/skill.md, the instructions a coding agent follows to install UNVEIL: checked against the code and README."""

import ast
import re
import shlex
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import yaml

from unveil import prompts as P

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "docs" / "skill.md").read_text()
FRONT, BODY = TEXT.split("---\n", 2)[1:]
FREE_TEXT = ("person", "dog")                    # the skill's free-text examples, not task prompts
# put there by the user, not in the repository
DOWNLOADED = ("checkpoints/unveil_v1.safetensors", "checkpoints/sam3.pt")
TOP = ("assets", "checkpoints", "configs", "datasets", "docs", "scripts", "tests", "tools", "unveil")


def fenced(text: str, lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", text, re.S)


def command_lines(text: str) -> list[str]:
    return [ln.strip() for block in fenced(text, "bash") for ln in block.splitlines() if ln.strip()]


@lru_cache(maxsize=None)
def script_flags(script: str, sub: str | None = None) -> set[str]:
    """The --flags of a script's argparse, from its --help."""
    args = [sys.executable, str(ROOT / "scripts" / script)] + ([sub] if sub else []) + ["--help"]
    out = subprocess.run(args, capture_output=True, text=True, cwd=ROOT, check=True).stdout
    return set(re.findall(r"(--[a-z][\w-]*)", out))


def test_frontmatter():
    meta = yaml.safe_load(FRONT)
    assert meta["name"] == "unveil"
    assert 0 < len(meta["description"]) <= 1024
    assert 100 <= len(TEXT.splitlines()) <= 165


def test_repository_paths_exist():
    paths = set(re.findall(rf"(?<![\w/.])((?:{'|'.join(TOP)})/[\w./-]*\w)", BODY))
    assert paths, "no repository paths found"
    missing = [p for p in sorted(paths) if p not in DOWNLOADED and not (ROOT / p).exists()]
    assert not missing, missing


def test_flags_exist_in_the_scripts():
    used = 0
    for line in command_lines(BODY):
        tokens = shlex.split(line)
        if len(tokens) < 2 or tokens[0] != "python" or not tokens[1].startswith("scripts/"):
            continue
        script = tokens[1].split("/", 1)[1]
        sub = tokens[2] if script == "visualize.py" else None
        flags = {t.split("=")[0] for t in tokens if t.startswith("--")}
        assert flags <= script_flags(script, sub), (line, flags - script_flags(script, sub))
        used += 1
    assert used >= 2
    inline = set(re.findall(r"`(--[a-z][\w-]*)", BODY))
    assert inline <= script_flags("predict.py") | script_flags("visualize.py", "panel"), inline


def test_prompts_are_defined():
    quoted = set(re.findall(r'`"([^"`]+)"`', BODY)) | set(re.findall(r'"([a-z ]+)"', FRONT))
    for line in command_lines(BODY):
        tokens = shlex.split(line)
        if "--prompts" in tokens:
            k = tokens.index("--prompts") + 1
            while k < len(tokens) and not tokens[k].startswith("--"):
                quoted.add(tokens[k])
                k += 1
    for block in fenced(BODY, "python"):
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "predict" and len(node.args) > 1:
                arg = node.args[1]
                items = arg.elts if isinstance(arg, ast.List) else [arg]
                quoted |= {x.value for x in items if isinstance(x, ast.Constant)}
    assert set(P.TASK_PROMPTS) <= quoted
    assert not set(FREE_TEXT) & set(P.TASK_PROMPTS) and all(f"`{w}`" in BODY for w in FREE_TEXT)
    assert quoted <= set(P.TASK_PROMPTS) | set(FREE_TEXT), quoted - set(P.TASK_PROMPTS)


def test_install_lines_equal_the_readme():
    readme = (ROOT / "README.md").read_text()
    quick = readme.split("## Quick start", 1)[1]

    def install(text):
        starts = ("git clone", "pip install", "wget")
        return [ln.split("  #")[0].strip() for ln in command_lines(text) if ln.startswith(starts)]

    assert install(BODY) == install(quick.split("\n## ", 1)[0])
    assert len(install(BODY)) == 5
    assert install(BODY)[-1] in command_lines((ROOT / "checkpoints" / "README.md").read_text())
