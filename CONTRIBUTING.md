# Contributing

Thanks for taking a look. Issues and pull requests are both welcome.

## Setup

```bash
pip install -e ".[dev]"
```

```bash
python scripts/download_weights.py
```

The unit tests need neither the detector weights nor any face images — they
cover the algorithmic core (LBP/HOG, the fusion rule, the genetic optimiser,
metrics, CLI). Only the end-to-end commands need models and data.

## Before opening a PR

```bash
ruff check . && pytest
```

CI runs the same two commands on Python 3.10 and 3.12, plus a wheel build.

## Guidelines

- **Never commit face images or model weights.** `.gitignore` blocks
  `data/gallery/*`, `data/probe/*` and `*.pt`; please keep it that way. Weights
  belong on a release, images belong to their subjects.
- **Do not report a metric you have not measured.** If a change is meant to
  improve accuracy, include the `outputs/metrics.json` numbers and say what the
  probe set was.
- **Keep the heavy imports lazy.** `torch`, `ultralytics` and `insightface` are
  imported inside the functions that need them so the package stays importable
  without them; `tests/test_cli.py` enforces this.
- Match the surrounding style — `ruff check .` is the arbiter.
