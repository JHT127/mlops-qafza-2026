"""Validate the checked-in Task 3 expectation contract."""

import json
from pathlib import Path


def main() -> None:
    path = Path("config/expectations/orders.json")
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = {"description", "columns", "policy_on_failure"}
    missing = required - contract.keys()
    if missing:
        raise SystemExit(f"Expectation contract is missing keys: {sorted(missing)}")
    if contract["policy_on_failure"] != "reject":
        raise SystemExit("Task 3 must reject invalid requests before model inference")
    print(f"Validated expectation contract: {path}")


if __name__ == "__main__":
    main()