# Task 1 — Get the Data Into a Database

**Goal:** download the Olist dataset and load it into a real, running PostgreSQL database, then prove it works with a few queries and joins.

## What's in this folder

| File             | Purpose                                                                        |
| ---------------- | ------------------------------------------------------------------------------ |
| `notebook.ipynb` | Full walkthrough: download → inspect → connect → load → query                  |
| `sql/schema.sql` | Reference SQL if you'd rather create tables by hand instead of `pandas.to_sql` |

## How to run it

1. From the repo root, start the database:
   ```bash
   docker compose up -d
   ```
2. Make sure your `.env` file is filled in (see `.env.example` at the repo root).
3. Open the notebook and run the cells top to bottom:
   ```bash
   jupyter notebook tasks/task-01-data-into-database/notebook.ipynb
   ```

## Done-when checklist (from the task brief)

- [x] Database running locally with the data inside
- [x] Can query the tables and join them
- [x] Understand the tables and the relationships between them
- [x] Understand the problem we're solving (late-delivery classification)

## Notes to self

- No EDA yet — that's a later task. This one is purely "get it in, prove it works."
- The active CSV copy is stored in the repository's `data/raw/` folder. The notebook copies the Kaggle download there before loading it, so clearing Downloads or the Kaggle cache will not break the project.
- Aggregation matters: `order_items` and `payments` are **multiple rows per order**, so anything built later at order-level needs a `GROUP BY` first, or joins will silently duplicate rows.
- Leakage watch: `order_delivered_customer_date` and review data are only available _after_ the fact — fine for analysis, not fine as model inputs later on.
