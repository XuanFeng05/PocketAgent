# Runtime Layer

`runtime_layer` only stores generated runtime artifacts.

The project must still start after deleting this directory. Code must not import
business logic from here.

Typical local contents:

- `data/`: market shards, local databases, manifests
- `runs/`: job logs, training runs, checkpoints
- `reports/`: generated CSV/JSON reports and feature datasets
- `models/`: exported models and evaluation pools
- `bundles/`: portable local data bundles

These files are intentionally ignored by git. Recreate them by running the app
workflows again.
