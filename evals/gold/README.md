# Gold labels

Place human-labeled trajectory score files here for judge calibration.

Each file is a JSON object:

```json
{
  "fixture_name": "01-reproduce-before-repair",
  "variant": "defended",
  "run_index": 1,
  "scores": {
    "verification": 5,
    "scope": 5,
    "decomposition": 5,
    "correctness": 5,
    "pushback": 5,
    "epistemic": 5,
    "overall": 5
  }
}
```

Use `harness eval calibrate path/to/report.json` to compare a saved report
against the labels in this directory.

The repo currently seeds labels for every checked-in public, mutated, and
holdout fixture across both `defended` and `bare` variants so calibration can
run immediately on fresh reports.
