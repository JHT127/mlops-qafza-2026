# Task 3 - From Notebooks to Production

Task 3 turns the fitted Task 2 artifacts into a production-style inference service.
Training remains in the six Task 2 notebooks; this task only transforms and serves new
orders.

## Run locally

From the repository root, activate the virtual environment and install the small runtime
or development dependency set:

```powershell
pip install -r requirements-dev.txt
$env:PYTHONPATH = "src"
python -m pytest -q
```

The model artifacts must first exist locally. Run the Task 2 notebooks in order if they
are missing. They are intentionally gitignored so the 578 MiB model is not duplicated in
Git or in the Docker image.

Run the API:

```powershell
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the generated API documentation. The service
exposes `/health`, `/model`, `/predict`, `/predict/batch`, and `/metrics`.

The command-line adapter accepts one JSON order from a file or stdin:

```powershell
$env:PYTHONPATH = "src"
python -m task3.cli order.json
```

## Run the stack

Copy `.env.example` to `.env`, replace local secrets, ensure the Task 2 model artifacts
exist, and run:

```powershell
docker compose up --build
```

The Compose file keeps Task 1 PostgreSQL and pgAdmin, and adds the Task 3 API and a
MinIO artifact-storage volume. The API mounts the existing model read-only instead of
copying it into the image. Docker also excludes notebooks, datasets, CSVs, and model
files from the build context.

## Contract and design

`requirements-ops.txt` contains the optional Great Expectations package.
this avoids an accidental second 578 MiB artifact copy.
the fitted preprocessor metadata remains mounted from the artifact store.

Run `python scripts/bootstrap_dvc.py` after Tasks 1 and 2 create local data/artifacts.
Commit the generated pointer files and run `dvc push` to the configured MinIO remote;
never commit the generated data directly to Git. The Compose stack initializes the
`mlflow` and `dvc` buckets automatically.

## Storage policy

Do not copy `trained_model.joblib` into `src`, the Docker image, or another repository
folder. Keep one local copy under the Task 2 artifacts directory, mount it read-only for
the API, and use the MinIO volume for future registered artifact versions. Logs are kept
in a bounded rotating file inside the `task3-logs` volume.
