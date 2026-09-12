"""Input validation contract shared by the CLI and API."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations.core.expectation_suite import ExpectationSuite

from .features import RAW_FEATURES

NON_NEGATIVE_FEATURES = [
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
]


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


def _great_expectations_result(records: list[dict[str, Any]]) -> tuple[str, ...]:
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="task3_requests")
    asset = data_source.add_dataframe_asset(name="orders")
    batch_definition = asset.add_batch_definition_whole_dataframe("request_batch")
    frame = pd.DataFrame(records)
    for column in RAW_FEATURES:
        if column not in frame:
            frame[column] = pd.NA
    suite = ExpectationSuite(name="task3_orders")
    for column in ("order_purchase_timestamp", "order_estimated_delivery_date"):
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeDateutilParseable(
                column=column, mostly=1.0
            )
        )
    for column in NON_NEGATIVE_FEATURES:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column, min_value=0, mostly=1.0
            )
        )
    result = batch_definition.get_batch({"dataframe": frame}).validate(suite)
    return tuple(
        f"Great Expectations failed: {item.expectation_config.type}"
        for item in result.results
        if not item.success
    )


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

    try:
        errors.extend(_great_expectations_result(records))
    except Exception as exc:
        errors.append(f"Great Expectations could not validate the request: {exc}")
    return ValidationResult(not errors, tuple(errors))


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert validated records into the frame expected by the fitted transformer."""
    frame = pd.DataFrame(records)
    for column in ("order_purchase_timestamp", "order_estimated_delivery_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame