-- Reference schema for the Olist dataset.
-- Not required if you're using pandas.to_sql() in the notebook (which infers
-- types automatically) — this is here for anyone who wants to create the
-- tables by hand first and load data with COPY / \copy instead.

CREATE TABLE IF NOT EXISTS olist_customers (
    customer_id             VARCHAR PRIMARY KEY,
    customer_unique_id      VARCHAR,
    customer_zip_code_prefix VARCHAR,
    customer_city           VARCHAR,
    customer_state          VARCHAR
);

CREATE TABLE IF NOT EXISTS olist_orders (
    order_id                      VARCHAR PRIMARY KEY,
    customer_id                   VARCHAR REFERENCES olist_customers(customer_id),
    order_status                  VARCHAR,
    order_purchase_timestamp      TIMESTAMP,
    order_approved_at             TIMESTAMP,
    order_delivered_carrier_date  TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP
);

CREATE TABLE IF NOT EXISTS olist_sellers (
    seller_id             VARCHAR PRIMARY KEY,
    seller_zip_code_prefix VARCHAR,
    seller_city           VARCHAR,
    seller_state          VARCHAR
);

CREATE TABLE IF NOT EXISTS olist_products (
    product_id                 VARCHAR PRIMARY KEY,
    product_category_name      VARCHAR,
    product_name_lenght        NUMERIC,
    product_description_lenght NUMERIC,
    product_photos_qty         NUMERIC,
    product_weight_g           NUMERIC,
    product_length_cm          NUMERIC,
    product_height_cm          NUMERIC,
    product_width_cm           NUMERIC
);

CREATE TABLE IF NOT EXISTS olist_order_items (
    order_id            VARCHAR REFERENCES olist_orders(order_id),
    order_item_id        INTEGER,
    product_id           VARCHAR REFERENCES olist_products(product_id),
    seller_id            VARCHAR REFERENCES olist_sellers(seller_id),
    shipping_limit_date  TIMESTAMP,
    price                NUMERIC,
    freight_value        NUMERIC,
    PRIMARY KEY (order_id, order_item_id)
);

CREATE TABLE IF NOT EXISTS olist_order_payments (
    order_id             VARCHAR REFERENCES olist_orders(order_id),
    payment_sequential   INTEGER,
    payment_type         VARCHAR,
    payment_installments INTEGER,
    payment_value        NUMERIC,
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE IF NOT EXISTS olist_order_reviews (
    review_id               VARCHAR PRIMARY KEY,
    order_id                VARCHAR REFERENCES olist_orders(order_id),
    review_score            INTEGER,
    review_comment_title    VARCHAR,
    review_comment_message  TEXT,
    review_creation_date    TIMESTAMP,
    review_answer_timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS olist_geolocation (
    geolocation_zip_code_prefix VARCHAR,
    geolocation_lat  NUMERIC,
    geolocation_lng  NUMERIC,
    geolocation_city VARCHAR,
    geolocation_state VARCHAR
);

-- Quick sanity check query used in the notebook:
-- SELECT o.order_id, o.order_status, c.customer_city, c.customer_state
-- FROM olist_orders o
-- JOIN olist_customers c ON o.customer_id = c.customer_id
-- LIMIT 10;
