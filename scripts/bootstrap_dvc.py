"""Create DVC pointer files for the local data and Task 2 artifacts."""

import subprocess
import sys
from pathlib import Path

TRACKED_PATHS = (
    Path("data/raw"),
    Path("data/processed"),
    Path("tasks/task-02-tables-to-notebooks/artifacts/models"),
    Path("tasks/task-02-tables-to-notebooks/artifacts/reports"),
)


def main() -> None:
    paths = [
        str(path)
        for path in TRACKED_PATHS
        if path.exists()
        and any(item.is_file() and item.name != ".gitkeep" for item in path.rglob("*"))
    ]
    if not paths:
        raise SystemExit(
            "No local data or Task 2 artifacts found; run Tasks 1 and 2 first "
            "before bootstrapping DVC."
        )
    result = subprocess.run([sys.executable, "-m", "dvc", "add", *paths])
    if result.returncode:
        raise SystemExit(
            "DVC could not take ownership of one or more paths. Remove tracked "
            "placeholder files from Git's index, commit that change, and rerun."
        )
    print("Created DVC pointer files. Commit the generated .dvc files, then run dvc push.")


if __name__ == "__main__":
    main()
