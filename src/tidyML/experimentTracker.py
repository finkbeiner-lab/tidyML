"""
Experiment trackers for machine learning pipelines.
"""

from abc import ABC, abstractmethod

import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
from neptune.new.types import File

# typing
from numpy import ndarray
from pandas import DataFrame
from typing import Union

class ExperimentTracker(ABC):
    """
    Encapsulates metadata for experiment tracking across runs.
    """

    @abstractmethod
    def __init__(
            self,
            projectID: str,
            **kwargs
        ):

        self.projectID = projectID

        for flag, value in kwargs.items():
            if flag == "apiToken":
                self.apiToken = value

    @abstractmethod
    def start(self, modelName, model):
        """
        Initialize tracker with a given model.
        """

    @abstractmethod
    def summarize(self, model, trainingData, testingData, trainingClasses, testingClasses):
        """
        Generate classifier summary.
        """
    
    @abstractmethod
    def logValue(self, valueGroup: str, valueMap: dict, metric = False):
        """
        Append values to track.
        """

    @abstractmethod
    def stop(self):
        """
        Send halt signal to experiment tracker to avoid memory leaks.
        """

class NeptuneExperimentTracker(ExperimentTracker):
    """
    Interface for experiment tracking using Neptune.
    """

    def __init__(self, projectID: str, **kwargs):
        super().__init__(projectID, **kwargs)

    def start(self, modelName, model, analysisName):
        self.modelName = modelName
        self.model = model
        self.tracker = neptune.init(
            project = self.projectID,
            api_token = self.apiToken,
            name = analysisName,
            tags = [self.modelName],
            capture_hardware_metrics = False
        )

    def summarize(
        self, trainingData: ndarray, testingData: ndarray, trainingClasses: ndarray, testingClasses: ndarray
    ):
        self.tracker["summary"] = npt_utils.create_classifier_summary(
            self.model,
            trainingData,
            testingData,
            trainingClasses,
            testingClasses
        )

    def logValue(self, valueGroup: str, valueMap: dict, metric = False):
        if metric:
            self.tracker[f"{valueGroup}"].log(valueMap)
        else:
            self.tracker[f"{valueGroup}"] = valueMap

    def addTags(self, tags: list):
        self.tracker["sys/tags"].add(tags)

    def uploadTable(self, fileName: str, table: Union[DataFrame, str]):
        if isinstance(table, DataFrame):
            self.tracker[f'data/{fileName}'].upload(File.as_html(table))
        elif isinstance(table, str):
            self.tracker[f'data/{fileName}'].upload(table)

    def stop(self):
        self.tracker.stop()
