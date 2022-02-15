"""
Experiment trackers for machine learning pipelines.
"""

from abc import ABC, abstractmethod

from numpy.lib.arraysetops import isin

import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
from neptune.new.types import File
import wandb
from io import BytesIO

# typing
from numpy import ndarray
from pandas import DataFrame
from typing import Union, List
from matplotlib.figure import Figure

# TODO: implementation-specific documentation


class ExperimentTracker(ABC):
    """
    Encapsulates metadata for experiment tracking across runs.
    """

    @abstractmethod
    def __init__(self, projectID: str, entityID: str, **kwargs):

        self.entityID = entityID
        self.projectID = projectID

        for flag, value in kwargs.items():
            if flag == "apiToken":
                self.apiToken = value

    @abstractmethod
    def start(self, **kwargs):
        """
        Initialize tracker with a given model.
        """

    @abstractmethod
    def summarize(self, **kwargs):
        """
        Generate classifier summary.
        """

    @abstractmethod
    def log(self, **kwargs):
        """
        Log a value to track.
        """

    @abstractmethod
    def addTags(self, **kwargs):
        """
        Append tags to the current tracking run.
        """

    @abstractmethod
    def getRuns(self, **kwargs):
        """
        Fetch the latest runs by ID or tag. All runs are fetched by default.
        """

    @abstractmethod
    def stop(self, **kwargs):
        """
        Send halt signal to experiment tracker and avoid memory leaks.
        """


class NeptuneExperimentTracker(ExperimentTracker):
    """
    Interface for experiment tracking using Neptune.
    """

    def __init__(self, projectID: str, entityID="", **kwargs):
        super().__init__(projectID, entityID, **kwargs)

    def start(self, model, analysisName):
        self.model = model
        self.tracker = neptune.init(
            project=self.projectID,
            api_token=self.apiToken,
            name=analysisName,
            tags=[self.model.__class__.__name__],
            capture_hardware_metrics=False,
        )

    def summarize(
        self,
        trainingData: ndarray,
        testingData: ndarray,
        trainingLabels: ndarray,
        testingLabels: ndarray,
    ):
        self.tracker["summary"] = npt_utils.create_classifier_summary(
            self.model, trainingData, testingData, trainingLabels, testingLabels
        )

    def log(
        self,
        path: str,
        value: Union[float, int, dict, str, Figure, DataFrame],
        metric=False,
        fileExtension=None,
    ):
        if metric:
            self.tracker[f"{path}"].log(value)
        elif isinstance(value, DataFrame) or type(value) == Figure:
            if fileExtension and type(value) == Figure:
                fileHandle = BytesIO()
                value.savefig(fileHandle, format=fileExtension)
                self.tracker[f"{path} preview"].upload(File.as_image(value))
                self.tracker[f"{path}"].upload(
                    File.from_stream(fileHandle, extension=fileExtension)
                )
            else:
                try:
                    self.tracker[f"{path}"].upload(File.as_html(value))
                except Exception:
                    if type(value) == Figure:
                        self.tracker[f"{path}"].upload(File.as_image(value))
                    print("Continuing past exception:" + str(Exception))
        # elif isFilePath:
        #     self.tracker.track_files()
        elif isinstance(value, str):
            self.tracker[f"{path}"].upload(value)
        else:
            self.tracker[f"{path}"] = value

    def addTags(self, tags: List):
        self.tracker["sys/tags"].add(tags)

    def getRuns(
        self,
        runID: Union[List, str] = None,
        tag: Union[List, str] = None,
    ):
        project = neptune.get_project(name=self.projectID, api_token=self.apiToken)
        self.runs = project.fetch_runs_table(id=runID, tag=tag)

    def stop(self):
        self.tracker.stop()


class WandbExperimentTracker(ExperimentTracker):
    def __init__(self, projectID: str, entityID: str, **kwargs):
        super().__init__(projectID, entityID, **kwargs)

    def start(self, model, type="sklearn"):
        self.tracker = wandb.init(
            project=self.projectID, entity=self.entityID, reinit=True
        )
        self.api = wandb.Api()
        if type != "sklearn":
            self.tracker.watch(model)

    def summarize(
        self,
        model,
        trainingData: ndarray,
        testingData: ndarray,
        trainingLabels: ndarray,
        testingLabels: ndarray,
        testPredictions: ndarray,
        testProbabilities: ndarray,
        classLabels: List[str],
        featureLabels: List[str],
        modelParameters: dict,
    ):
        self.tracker.config.update(modelParameters)
        self.tracker.sklearn.plot_classifier(
            model=model,
            X_train=trainingData,
            X_test=testingData,
            y_train=trainingLabels,
            y_test=testingLabels,
            y_pred=testPredictions,
            y_probas=testProbabilities,
            labels=classLabels,
            model_name=self.model.__class__.__name__,
            feature_names=featureLabels,
        )

    def log(self, valueMap: dict):
        for i, (key, value) in enumerate(valueMap.items()):
            commitStatus = False if i + 1 < len(valueMap) else True
            # TODO: parse values with WandB data types here
            if isinstance(value, Figure):
                figHandle = BytesIO()
                value.savefig(figHandle)
                self.tracker.log({key: figHandle}, commit=commitStatus)
            elif isinstance(value, DataFrame):
                self.tracker.log(
                    {key: wandb.Table(dataframe=value)}, commit=commitStatus
                )
            else:
                self.tracker.log({key: value}, commit=commitStatus)
                # artifact = wandb.Artifact(name=key, type=type)
                # artifact.add(value, name=key)

    def addTags(self, tags: List):
        self.tracker.tags.append(tags)

    def getRuns(self):
        self.runs = self.api.runs(self.entityID + "/" + self.projectID)

    def stop(self):
        self.tracker.finish()
