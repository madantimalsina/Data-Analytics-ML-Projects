# Data Analytics & ML Projects

[![CI](https://github.com/madantimalsina/Data-Analytics-ML-Projects/actions/workflows/ci.yml/badge.svg)](https://github.com/madantimalsina/Data-Analytics-ML-Projects/actions/workflows/ci.yml)

A collection of machine learning projects (classification and regression), originally developed as
Jupyter notebooks and since reorganized into a runnable Python project. Each project is implemented
in both Keras/TensorFlow and PyTorch.

## Layout

| Folder | Contents |
|---|---|
| [notebooks/](notebooks/README.md) | Original exploratory Jupyter notebooks (Keras/TensorFlow). |
| [src/](src/README.md) | Standalone Python scripts converted from the notebooks, in parallel [keras/](src/keras/README.md) and [pytorch/](src/pytorch/README.md) folders. |
| [sql/](sql/README.md) | SQL files used for data-selection/preview steps, run against an in-memory SQLite database built from the CSVs. |
| [data/](data/README.md) | CSV datasets used by the notebooks and scripts. |
| [models/](models/README.md) | Saved/trained model files. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running a notebook

```bash
source .venv/bin/activate
jupyter notebook
```

Then open any file in [notebooks/](notebooks/README.md).

## Running a script

```bash
source .venv/bin/activate
python src/keras/house_price_regression.py --data data/kc_house_data.csv --sql sql/kc_house_price_by_year.sql
python src/pytorch/house_price_regression.py --data data/kc_house_data.csv --sql sql/kc_house_price_by_year.sql
```

See [src/README.md](src/README.md) for all available scripts and options.

## CI

Every push/PR to `main` runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml), which:
validates all notebooks parse, byte-compiles every script, runs the SQL data-selection
queries, and smoke-tests each Keras and PyTorch script end-to-end with a few epochs.
