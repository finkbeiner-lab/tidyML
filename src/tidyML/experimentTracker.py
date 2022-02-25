"""
Experiment trackers for machine learning pipelines.
"""

from abc import ABC, abstractmethod

from numpy.lib.arraysetops import isin

import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
from neptune.new.types import File
import wandb
from io import StringIO
# typing
from numpy import ndarray
from pandas import DataFrame
from typing import Union, List
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# TODO: implementation-specific documentation


class ExperimentTracker(ABC):
    """
    Encapsulates metadata for experiment tracking across runs.
    """

    @abstractmethod
    def __init__(self, projectID: str, entityID: str, analysisName: str, **kwargs):

        self.entityID = entityID
        self.projectID = projectID
        self.analysisName = analysisName

        for flag, value in kwargs.items():
            if flag == "apiToken":
                self.apiToken = value

    def start(self, **kwargs):
        """
        Initialize tracker with a given model.
        """

    def summarize(self, **kwargs):
        """
        Generate classifier summary.
        """

    def log(self, **kwargs):
        """
        Log a value to track.
        """

    def addTags(self, **kwargs):
        """
        Append tags to the current tracking run.
        """

    def getRuns(self, **kwargs):
        """
        Fetch the latest runs by ID or tag. All runs are fetched by default.
        """

    def stop(self, **kwargs):
        """
        Send halt signal to experiment tracker and avoid memory leaks.
        """


class NeptuneExperimentTracker(ExperimentTracker):
    """
    Interface for experiment tracking using Neptune.
    """

    def __init__(self, projectID: str, entityID: str, analysisName: str, **kwargs):
        super().__init__(projectID, entityID, analysisName, **kwargs)
        
    def start(self, model):
        self.model = model
        self.tracker = neptune.init(
            project=self.project_id + "/" + self.entityID,
            api_token=self.apiToken,
            name=self.analysisName,
            tags=[self.model.__class__.__name__],
            capture_hardware_metrics=False,
        )
        
    def summarize(
        self,
        model,
        trainingData: ndarray,
        testingData: ndarray,
        trainingLabels: ndarray,
        testingLabels: ndarray,
        **kwargs # TODO: implement multimethods to emulate function overloading
    ):
        self.tracker["summary"] = npt_utils.create_classifier_summary(
            model, trainingData, testingData, trainingLabels, testingLabels
        )
            
    def log(
        self,
        path: str,
        value: Union[float, int, dict, str, Figure, DataFrame],
        metric: bool = False,
        **kwargs
    ):
        if metric:
            self.tracker[f"{path}"].log(value)
        elif isinstance(value, DataFrame) or type(value) == Figure:
            if type(value) == Figure:
                fileHandle = BytesIO()
                value.savefig(fileHandle, format="svg")
                self.tracker[f"{path} preview"].upload(File.as_image(value))
                self.tracker[f"{path}"].upload(
                    File.from_stream(fileHandle, extension="svg")
                )
            else:
                try:
                    self.tracker[f"{path}"].upload(File.as_html(value))
                except Exception:
                    if type(value) == Figure:
                        self.tracker[f"{path}"].upload(File.as_image(value))
                    print("Continuing past exception:" + str(Exception))
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
    """
    Interface for experiment tracking using Weights & Biases.
    """
    def __init__(self, projectID: str, entityID: str, **kwargs):
        super().__init__(projectID, entityID, **kwargs)
        self.api = wandb.Api()
    
    def start(self, model, type="sklearn"):
        wandb.finish() # clear any hanging runs
        self.tracker = wandb.init(
            project=self.projectID, entity=self.entityID, reinit=True
        )
        if type != "sklearn":
            self.tracker.watch(model)
        self.tracker.name = model.__class__.__name__
    
    def summarize(
        self,
        model,
        hyperparameters: dict,
        trainingData: DataFrame,
        testingData: DataFrame,
        trainingLabels: ndarray,
        testingLabels: ndarray,
        testPredictions: ndarray,
        testProbabilities: ndarray,
        classLabels: List[str] = None,
        featureLabels: List[str] = None,
        isSklearn: bool = True 
    ):
        self.tracker.config.update(hyperparameters)
        if isSklearn:
            wandb.sklearn.plot_classifier(
                model=model,
                X_train=trainingData,
                X_test=testingData,
                y_train=trainingLabels,
                y_test=testingLabels,
                y_pred=testPredictions,
                y_probas=testProbabilities,
                labels=classLabels,
                model_name=model.__class__.__name__,
                feature_names=featureLabels,
            )
    
    def log(self, path: str, valueMap: dict, step: int = None):
        runningLog = dict()
        for (key, value) in valueMap.items():
            if isinstance(value, Figure):
                value.tight_layout()
                svgHandle = StringIO()
                value.savefig(svgHandle, format="svg", bbox_inches="tight")
                runningLog[key] = wandb.Html(svgHandle)
                runningLog[key+" preview"] = wandb.Image(value)
            elif isinstance(value, DataFrame):
                runningLog[key]= wandb.Table(dataframe=value)
            else:
                runningLog[key] = value
                
        self.tracker.log(
            {path: runningLog}, 
            step=step
        )
                
    def addTags(self, tags: List):
        self.tracker.tags.append(tags)
    
    def getRuns(self):
        self.runs = self.api.runs(self.entityID + "/" + self.projectID)
    
    def stop(self):
        self.tracker.finish()
