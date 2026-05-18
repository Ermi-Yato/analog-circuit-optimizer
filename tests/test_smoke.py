"""Phase 0 smoke tests — verify project structure and imports."""
import importlib
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED_DIRS = [
    "app",
    "app/views",
    "app/widgets",
    "app/workers",
    "core",
    "core/simulation",
    "core/dataset",
    "core/models",
    "core/optimization",
    "core/validation",
    "registry",
    "registry/circuits",
    "trained_models",
    "tests",
    "docs",
    "circuits",
    "data",
    "outputs",
]


def test_folder_structure():
    """All required directories exist."""
    for d in EXPECTED_DIRS:
        full = os.path.join(PROJECT_ROOT, d.replace("/", os.sep))
        assert os.path.isdir(full), f"Missing directory: {d}"


def test_core_packages_importable():
    """core sub-packages are importable (no GUI dependency)."""
    packages = [
        "core",
        "core.simulation",
        "core.dataset",
        "core.models",
        "core.optimization",
        "core.validation",
        "registry",
    ]
    for pkg in packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Could not import {pkg}"


def test_requirements_file_exists():
    req = os.path.join(PROJECT_ROOT, "requirements.txt")
    assert os.path.isfile(req)
    content = open(req).read()
    for dep in ["PySide6", "deap", "schemdraw", "scikit-learn", "pandas", "numpy"]:
        assert dep in content, f"Missing dependency in requirements.txt: {dep}"


def test_main_entrypoint_exists():
    main = os.path.join(PROJECT_ROOT, "main.py")
    assert os.path.isfile(main)


def test_claude_md_exists():
    claude = os.path.join(PROJECT_ROOT, "CLAUDE.md")
    assert os.path.isfile(claude)


def test_tasks_md_exists():
    tasks = os.path.join(PROJECT_ROOT, "docs", "TASKS.md")
    assert os.path.isfile(tasks)
