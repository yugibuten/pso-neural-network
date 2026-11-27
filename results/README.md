Experiment outputs used in the report.

- `csv/expN.csv`: one row per run (configuration, train/test metrics, convergence history).
- `csv/expN.metadata.json`: configurations, seeds, dataset hash and library versions for each suite.
- `csv/expN_artifacts/*.npz`: best particle and scaler statistics for each run.
- `figures/expN/`: summary tables (mean ± SD over seeds 0-9) and convergence plots.
- `architecture.json`: architecture selected from Experiment 1 by mean training accuracy.
