"""
Hyperparameter optimization for SciKit models.
"""

from os import cpu_count
from dataclasses import dataclass
from functools import partial
from joblib import Parallel, delayed
import numpy as np
from skopt import Optimizer, Space

# sci-kit
from skopt.learning import (
    GaussianProcessRegressor as ScikitGaussianProcessRegressor,
    GradientBoostingQuantileRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from skopt.learning.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    ConstantKernel,
)

# types
import skopt.learning.gaussian_process.kernels as Kernel
from typing import List, Callable, Iterable, Union


@dataclass
class GaussianProcessRegressor:
    """
    Interface for the Scikit Guassian Process regressor, used as a
    surrogate model during Bayesian hyperparameter optimization.
    """

    kernels: List[str] = (
        "RBF",
        "RationalQuadratic",
        "ExpSineSquared",
        # "ConstantKernel",
        "Matern",
    )
    alpha: float = 1e-10
    kernelOptimizer: Union[str, Callable] = "fmin_l_bfgs_b"
    kernelRestartCount: int = 0
    normalizeTargetValues: bool = False
    randomState: int = None
    noise: str = None

    def __post_init__(self):
        # kernel parameters will be optimized using `kernelOptimizer`
        defaultKernels = {
            "RBF": 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
            "RationalQuadratic": 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
            "ExpSineSquared": 1.0
            * ExpSineSquared(
                length_scale=1.0,
                periodicity=3.0,
                length_scale_bounds=(0.1, 10.0),
                periodicity_bounds=(1.0, 10.0),
            ),
            "ConstantKernel": ConstantKernel(0.1, (0.01, 10.0))
            * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
            "Matern": 1.0
            * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2.5),
        }
        self.selectedKernels = [defaultKernels[kernel] for kernel in self.kernels]

    def addKernel(self, kernel: Kernel):
        """
        Add a Gaussian Process kernel from scikit-learn. Use this method
        to add combined kernels from WhiteKernel, Sum & Product.
        """
        self.selectedKernels.append(kernel)

    def load(self, currentKernel: Kernel):
        """
        Initialize a new GaussianProcessRegressor with the given kernel.
        """
        return ScikitGaussianProcessRegressor(
            kernel=currentKernel,
            alpha=self.alpha,
            optimizer=self.kernelOptimizer,
            n_restarts_optimizer=self.kernelRestartCount,
            normalize_y=self.normalizeTargetValues,
            copy_X_train=True,
            random_state=self.randomState,
            noise=self.noise,
        )


@dataclass
class RegressorCollection:
    """
    Encapsulates base estimators implemented in scikit-learn.
    These estimators are used in regression to optimize model hyperparameters.

    Keyword arguments are passed as a dictionary for each regressor type.
    Default parameters can be set by passing "default" instead.
    """

    GradiantBoostingQuantile: Union[dict, str] = None
    GaussianProcess: Union[dict, str] = None
    RandomForest: Union[dict, str] = None
    ExtraTrees: Union[dict, str] = None

    def __post_init__(self):
        regressors = {
            "GradiantBoostingQuantile": GradientBoostingQuantileRegressor,
            "GaussianProcess": GaussianProcessRegressor,
            "RandomForest": RandomForestRegressor,
            "ExtraTrees": ExtraTreesRegressor,
        }
        self.selectedRegressors = list()
        for regressor in self.__dataclass_fields__:
            kwargs = getattr(self, regressor)
            # enumerate GaussianProcessRegressors if selected with different kernels
            if regressor == "GaussianProcess":
                if kwargs == "default":
                    # pass empty args to run post-init
                    gaussianRegressors = regressors[regressor]()
                elif kwargs != None:
                    gaussianRegressors = regressors[regressor](**kwargs)
                else:
                    continue
                for kernel in gaussianRegressors.selectedKernels:
                    self.selectedRegressors.append(gaussianRegressors.load(kernel))
            elif kwargs == "default":
                self.selectedRegressors.append(regressors[regressor])
            elif kwargs != None:
                self.selectedRegressors.append(regressors[regressor](**kwargs))


@dataclass
class BayesianOptimizer:
    """
    Run a Bayesian optimization loop to find model hyperparameters
    by comparing various regression methods.
    """

    regressorCollection: RegressorCollection
    iterations: int = 10
    initialPoints: int = 10
    initialPointGenerator: str = "random"
    acquisitionFunction: str = "gp_hedge"
    acquisitionOptimizer: str = "auto"
    acquisitionFunctionKwargs: dict = None
    acquisitionOptimizerKwargs: dict = None
    modelQueueSize: int = None
    randomState: int = None
    jobCount: int = -1
    verbose: bool = True

    def optimize(
        self,
        model,
        hyperparameterSpaces: Iterable[Space],
        trainingData,
        trainingLabels,
        objectiveToMinimize: Callable,
        pointSampleCount: int = cpu_count(),
        pointSampleStrategy="cl_min",
    ) -> dict:
        """
        Optimize a given Scikit model and objective function to minimize in the hyperparameter space.
        Returns a dictionary of best hyperparameters keyed by name.
        """
        self.bestParametersByRegressor = dict()
        bestParameters = {"regressor": None, "score": float("inf"), "parameters": None}
        loadedObjective = partial(
            objectiveToMinimize,
            **{
                "model": model,
                "trainingData": trainingData,
                "trainingLabels": trainingLabels,
            },
        )
        hyperparameterNames = [parameter.name for parameter in hyperparameterSpaces]
        for regressor in self.regressorCollection.selectedRegressors:
            regressorName = regressor.__class__.__name__ + (
                str(regressor.kernel.k2).split("(")[0]
                if regressor.__class__.__name__ == "GaussianProcessRegressor"
                else ""
            )
            optimizer = Optimizer(
                dimensions=hyperparameterSpaces,
                base_estimator=regressor,
                n_initial_points=self.initialPoints,
                initial_point_generator=self.initialPointGenerator,
                acq_func=self.acquisitionFunction,
                acq_optimizer=self.acquisitionOptimizer,
                random_state=self.randomState,
                n_jobs=self.jobCount,
                acq_func_kwargs=self.acquisitionFunctionKwargs,
                acq_optimizer_kwargs=self.acquisitionOptimizerKwargs,
                model_queue_size=self.modelQueueSize,
            )
            if self.verbose:
                print(
                    f"Starting {model.__class__.__name__} hyperparameter optimization with {regressorName}...",
                    end="\r",
                )
            for i in range(0, self.iterations):
                sampledPoints = optimizer.ask(
                    n_points=pointSampleCount, strategy=pointSampleStrategy
                )
                sampledParameters = [
                    {
                        hyperparameter: point
                        for hyperparameter, point in zip(hyperparameterNames, points)
                    }
                    for points in sampledPoints
                ]
                optimizedParameters = Parallel(n_jobs=2)(
                    delayed(loadedObjective)(**point) for point in sampledParameters
                )
                # remove points that did not converge
                if np.isnan(optimizedParameters).any():
                    pointIndicesToDrop = np.flatnonzero(np.isnan(optimizedParameters))
                    for index in sorted(pointIndicesToDrop, reverse=True):
                        del optimizedParameters[index]
                        del sampledPoints[index]
                optimizer.tell(sampledPoints, optimizedParameters)
                bestScore = min(optimizer.yi)
                if self.verbose:
                    print(
                        f"Finished iteration {i+1} of {self.iterations} optimizing {model.__class__.__name__} with {regressorName}. Best score: {bestScore}",
                        end="\r",
                    )
            self.bestParametersByRegressor[regressorName] = {
                name: parameter
                for (name, parameter) in zip(
                    hyperparameterNames,
                    optimizer.Xi[np.argmin(optimizer.yi)],
                )
            }
            if self.verbose:
                print(
                    f"\n{model.__class__.__name__} using {regressorName} has best score of {bestScore}"
                )
            if bestParameters["score"] > bestScore:
                bestParameters["regressor"] = regressorName
                bestParameters["score"] = bestScore
                bestParameters["parameters"] = self.bestParametersByRegressor[
                    regressorName
                ]
        if self.verbose:
            print(
                f"{model.__class__.__name__} has global best parameters using {bestParameters['regressor']} with score {bestParameters['score']}"
            )
            for name, parameter in bestParameters["parameters"].items():
                print(f"{name}: {parameter}")
        return bestParameters
