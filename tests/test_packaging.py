"""The built wheel installs into a clean environment and works from anywhere.

Run from a copy of the sources in a temporary folder, so neither the build's
scratch files nor the repository's own harness/ on the path can stand in for
what was installed.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UV = shutil.which("uv")


def _uv(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    # Offline first (the cache usually has setuptools); a clean machine goes online.
    for extra in (["--offline"], []):
        done = subprocess.run(
            [UV, *args, *extra], cwd=cwd, capture_output=True, text=True, check=False, timeout=600
        )
        if done.returncode == 0:
            return done
    raise AssertionError(f"uv {' '.join(args)} failed:\n{done.stderr}")


@pytest.mark.skipif(UV is None, reason="needs uv to build and install")
def test_the_wheel_installs_and_runs_outside_the_repository(tmp_path):
    src = tmp_path / "src-copy"
    shutil.copytree(ROOT / "src", src / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, src / name)
    _uv("build", "--wheel", "-o", str(tmp_path / "dist"), cwd=src)
    wheel = next((tmp_path / "dist").glob("*.whl"))
    _uv("venv", str(tmp_path / "venv"), cwd=tmp_path)
    _uv("pip", "install", "--python", str(tmp_path / "venv"), str(wheel), cwd=tmp_path)

    bindir = "Scripts" if sys.platform == "win32" else "bin"
    python = tmp_path / "venv" / bindir / ("python.exe" if sys.platform == "win32" else "python")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import harness, harness.bench, harness.bench_tasks; print(harness.__file__)",
        ],
        cwd=elsewhere,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert "site-packages" in probe.stdout and str(ROOT) not in probe.stdout
    cli = subprocess.run(
        [str(python), "-m", "harness", "--help"],
        cwd=elsewhere,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert cli.returncode == 0 and "--num-ctx" in cli.stdout
    exe = tmp_path / "venv" / bindir / ("harness.exe" if sys.platform == "win32" else "harness")
    assert exe.exists()  # the console script from [project.scripts]


def test_the_package_has_no_runtime_dependencies():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "\ndependencies = []\n" in text
