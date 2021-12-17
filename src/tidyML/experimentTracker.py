"""
Experiment trackers for machine learning pipelines.
"""

from abc import ABC, abstractmethod

import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
from neptune.new.types import File
import os

# typing
from numpy import ndarray
from pandas import DataFrame
from typing import Union, List
from matplotlib.figure import Figure


class ExperimentTracker(ABC):
    """
    Encapsulates metadata for experiment tracking across runs.
    """

    @abstractmethod
    def __init__(self, projectID: str, **kwargs):

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
    def summarize(self, model, trainingData, validation, trainingLabels, testingLabels):
        """
        Generate classifier summary.
        """

    @abstractmethod
    def logValue(self, path: str, value: dict, metric=False):
        """
        Append values to track.
        """

    @abstractmethod
    def addTags(self, tags: List):
        """
        Append tags to the current tracking run.
        """

    @abstractmethod
    def getRuns(
        self,
        runID: Union[List, str] = None,
        tag: Union[List, str] = None,
    ):
        """
        Fetch the latest runs by ID or tag. All runs are fetched by default.
        """

    @abstractmethod
    def uploadTable(self, fileName: str, table: Union[DataFrame, Figure, str]):
        """
        Upload a table from a CSV or Pandas dataframe.
        """

    @abstractmethod
    def stop(self):
        """
        Send halt signal to experiment tracker and avoid memory leaks.
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
            project=self.projectID,
            api_token=self.apiToken,
            name=analysisName,
            tags=[self.modelName],
            capture_hardware_metrics=False,
        )

    def summarize(
        self,
        trainingData: ndarray,
        validation: ndarray,
        trainingLabels: ndarray,
        testingLabels: ndarray,
    ):
        self.tracker["summary"] = npt_utils.create_classifier_summary(
            self.model, trainingData, validation, trainingLabels, testingLabels
        )

    def logValue(self, path: str, value: Union[float, int, dict], metric=False):
        if metric:
            self.tracker[f"{path}"].log(value)
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

    def uploadTable(
        self,
        fileName: str,
        table: Union[DataFrame, Figure, str],
        fileExtension: str = None,
    ):
        if isinstance(table, DataFrame) or type(table) == Figure:
            if fileExtension == "pdf" and type(table) == Figure:
                # write temporary PDF & upload
                fileName = fileName + "." + fileExtension
                table.savefig(fileName)
                self.tracker[f"uploads/{fileName}"].upload(fileName)
                os.remove(fileName)
            else:
                try:
                    self.tracker[f"uploads/{fileName}"].upload(File.as_html(table))
                except Exception:
                    if type(table) == Figure:
                        self.tracker[f"uploads/{fileName}"].upload(File.as_image(table))
                    print("Continuing past exception:" + str(Exception))
        elif isinstance(table, str):
            self.tracker[f"data/{fileName}"].upload(table)

    def stop(self):
        self.tracker.stop()
