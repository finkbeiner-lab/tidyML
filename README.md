# tidyML

Machine learning pipelines are too often built from spaghetti. This package provides utilities to consolidate logic
and minimize duplication, which helps you write maintainable code.

## Modules

- #### DataMediator

- #### ExperimentTracker

  - `NeptuneExperimentTracker`
  - `WandbExperimentTracker`

- #### HyperparameterOptimizer

  - `BayesianOptimizer`
  - `GaussianProcessRegressor`
  - `RegressorCollection`

## Environment

- [Python 3.9 or greater](https://www.python.org/downloads/)

## Development quickstart

0. Install Python, open a terminal in the tidyML directory

1. Install dependencies with `pip install -r requirements.txt`
