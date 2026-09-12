"""Input validation contract shared by the CLI and API."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from .features import RAW_FEATURES


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


def validate_orders(records: list[dict[str, Any]]) -> ValidationResult:
    """Apply the Task 3 data expectations before transformation.

    The checks mirror the expectation suite in ``config/expectations/orders.json``.
    Missing model features are allowed because the fitted Task 2 imputer handles them;
    date fields are required because date features cannot be derived without them.
    """
    errors: list[str] = []
    for index, record in enumerate(records):
        missing_dates = [
            field
            for field in ("order_purchase_timestamp", "order_estimated_delivery_date")
            if not record.get(field)
        ]
        if missing_dates:
            errors.append(f"records[{index}] missing required date fields: {missing_dates}")

        for field in RAW_FEATURES:
            value = record.get(field)
            if value is not None and isinstance(value, bool):
                errors.append(f"records[{index}].{field} must be numeric or text, not boolean")

        for field in (
            "n_items",
            "n_distinct_products",
            "n_distinct_sellers",
            "total_price",
            "total_freight_value",
            "avg_freight_value",
            "total_weight_g",
            "max_product_length_cm",
            "max_product_height_cm",
            "max_product_width_cm",
            "total_payment_value",
            "n_payment_transactions",
            "max_payment_installments",
        ):
            value = record.get(field)
            if value is not None:
                try:
                    if float(value) < 0:
                        errors.append(f"records[{index}].{field} must be >= 0")
                except (TypeError, ValueError):
                    errors.append(f"records[{index}].{field} must be numeric")

        for field in ("purchase_dayofweek", "purchase_month"):
            value = record.get(field)
            if value is not None:
                try:
                    numeric = float(value)
                    limits = (0, 6) if field == "purchase_dayofweek" else (1, 12)
                    if not limits[0] <= numeric <= limits[1]:
                        errors.append(
                            f"records[{index}].{field} must be between {limits[0]} and {limits[1]}"
                        )
                except (TypeError, ValueError):
                    errors.append(f"records[{index}].{field} must be numeric")

        for field in ("order_purchase_timestamp", "order_estimated_delivery_date"):
            try:
                datetime.fromisoformat(str(record[field]).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                errors.append(f"records[{index}].{field} must be an ISO date or datetime")

    return ValidationResult(not errors, tuple(errors))


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert validated records into the frame expected by the fitted transformer."""
    frame = pd.DataFrame(records)
    for column in ("order_purchase_timestamp", "order_estimated_delivery_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame