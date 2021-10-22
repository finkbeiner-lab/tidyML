"""
Hyperparameter optimization for SciKit models.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Union

from pandas import DataFrame, Series
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score
import numpy as np

@dataclass
class HyperparameterSpace():
    """
    Dataclass that defines hyperparameter spaces.
    """
    name: str
    datatype: str
    upperBound: Union[int, float]
    lowerBound: Union[int, float]

class SciKitHyperparameterOptimizer():
    """
    Optimally parameterize a SciKit model.
    """

    def __init__(
            self,
            model,
            trainingData: DataFrame,
            classes: Series,
            cvStrategy: Union[int, Iterable],
            scoringMethod: Union[str, Callable],
            iterations: int,
            hyperparamsToOptimize: Union[list[HyperparameterSpace], dict],
            verbose: bool = True
        ):
        self.model = model
        self.trainingData = trainingData
        self.classes = classes
        self.cvStrategy = cvStrategy
        self.scoringMethod = scoringMethod
        self.iterations = iterations
        self.hyperparameterSpaces = list()
        self.modelName = type(model).__name__
        self.verbose = verbose

        # TODO: determine which properties are kwargs for specific models

        if isinstance(hyperparamsToOptimize, dict):
            self.hyperparameterSpaces = hyperparamsToOptimize
        elif all(isinstance(item, HyperparameterSpace) for item in hyperparamsToOptimize):
            for hyperparam in hyperparamsToOptimize:
                # TODO: refactor using match case upon python 3.10 release
                if hyperparam.datatype == "real":
                    self.hyperparameterSpaces.append(
                        Real(
                            hyperparam.lowerBound,
                            hyperparam.upperBound,
                            hyperparam.name
                        )
                    )
                elif hyperparam.datatype == "integer":
                    self.hyperparameterSpaces.append(
                        Integer(
                            hyperparam.lowerBound,
                            hyperparam.upperBound,
                            hyperparam.name
                        )
                    )
        elif all(isinstance(item, dict) for item in hyperparamsToOptimize):
            self.hyperparameterSpaces = hyperparamsToOptimize
        else:
            raise ValueError("hyperparametersToOptimize contains items of unknown type")


    def optimize(self):
        """
        Return a copy of the current model with optimized hyperparameters.
        Optimization occurs by minimizing the cross-validation score of training data.
        """
        # TODO: implement strategy for neural nets & AdaBoost

        @use_named_args(self.hyperparameterSpaces)
        def bayesianObjective(**hyperparams):
            self.model.set_params(**hyperparams)
            return -np.mean(
                cross_val_score(
                    self.model,
                    self.trainingData,
                    self.classes,
                    cv=self.cvStrategy,
                    scoring=self.scoringMethod,
                    n_jobs=-1,
                )
            )

        optimizeResult = gp_minimize(
            bayesianObjective,
            self.hyperparameterSpaces,
            n_calls=self.iterations
        )
        if self.verbose:
            print(f"{self.modelName} Best score=%.4f" % optimizeResult.fun)
            print(f"""{self.modelName} Best parameters:
                    - max_depth=%d \n
                    - learning_rate=%.6f 
                    - max_features=%d
                    - min_samples_split=%d
                    - min_samples_leaf=%d
                    - n_estimators=%d""" %
                    (optimizeResult.x[0], optimizeResult.x[1],
                    optimizeResult.x[2], optimizeResult.x[3],
                    optimizeResult.x[4], optimizeResult.x[5])
            )

            # TODO: debug print statements for logistic regression

        return optimizeResult
