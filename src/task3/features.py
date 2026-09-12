"""Feature contract shared by notebook outputs and inference."""

NUMERIC_FEATURES = [
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
    "purchase_dayofweek",
    "purchase_month",
    "promised_delivery_days",
]

CATEGORICAL_FEATURES = [
    "main_product_category",
    "main_payment_type",
    "customer_state",
    "main_seller_state",
]

RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DATE_FEATURE_INPUTS = [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
]


def add_date_features(frame):
    """Create the same date features used by Task 2 Notebook 5."""
    result = frame.copy()
    purchase = result["order_purchase_timestamp"]
    estimated = result["order_estimated_delivery_date"]
    result["purchase_dayofweek"] = purchase.dt.dayofweek
    result["purchase_month"] = purchase.dt.month
    result["promised_delivery_days"] = (estimated - purchase).dt.days
    return result
