"""Create DVC pointer files for the local data and Task 2 artifacts."""

import subprocess
from pathlib import Path

TRACKED_PATHS = (
    Path("data/raw"),
    Path("data/processed"),
    Path("tasks/task-02-tables-to-notebooks/artifacts/models"),
    Path("tasks/task-02-tables-to-notebooks/artifacts/reports"),
)


def main() -> None:
    paths = [str(path) for path in TRACKED_PATHS if path.exists()]
    if not paths:
        raise SystemExit("No local data or Task 2 artifacts found; run Tasks 1 and 2 first")
    subprocess.run(["dvc", "add", *paths], check=True)
    print("Created DVC pointer files. Commit the generated .dvc files, then run dvc push.")


if __name__ == "__main__":
    main()
