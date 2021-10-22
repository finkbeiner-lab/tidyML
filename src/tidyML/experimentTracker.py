"""
Experiment trackers for machine learning pipelines.
"""

from abc import ABC, abstractmethod

import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils

class ExperimentTracker(ABC):
    """
    Encapsulates metadata for experiment tracking across runs.
    """

    @abstractmethod
    def __init__(
            self,
            projectName: str,
            analysisName: str,
            tags: list,
            valuesToTrack: dict,
            model,
            **kwargs
        ):

        self.projectName = projectName
        self.analysisName = analysisName
        self.tags = tags
        self.valuesToTrack = valuesToTrack
        self.model = model

        for flag, value in kwargs.items:
            if flag == "apiToken":
                self.apiToken = value
            if flag == "trainingData":
                self.trainingData = value
            if flag == "testingData":
                self.testingData = value
            if flag == "trainingClasses":
                self.trainingClasses = value
            if flag == "testingClasses":
                self.testingClasses = value

    @abstractmethod
    def trackValue(self, name, value):
        """
        Append additional columns and values to track.
        """

    @abstractmethod
    def stopTracker(self):
        """
        Send halt signal to experiment tracker and avoid memory leaks.
        """

class NeptuneExperimentTracker(ExperimentTracker):
    """
    Interface for experiment tracking using Neptune.
    """

    def __init__(self,
        projectName: str,
        analysisName: str,
        tags: list,
        valuesToTrack: dict,
        **kwargs
    ):
        super().__init__(projectName, analysisName, tags, valuesToTrack, **kwargs)

        self.tracker = neptune.init(
            project = self.projectName,
            api_token = self.apiToken,
            name = self.analysisName,
            tags = self.tags,
            capture_hardware_metrics=False
        )

        for column, value in self.valuesToTrack:
            self.tracker[column] = value

        self.tracker["summary"] = npt_utils.create_classifier_summary(
            self.model,
            self.trainingData,
            self.testingData,
            self.trainingClasses,
            self.testingClasses
        )

    def trackValue(self, name, value):
        self.tracker[name] = value

    def stopTracker(self):
        self.tracker.stop()
