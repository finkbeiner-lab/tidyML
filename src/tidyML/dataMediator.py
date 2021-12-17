"""
Contains a data mediator to split, balance, and holdout experimental
and control data for machine learning pipelines.
"""

from typing import Callable, Optional, Union, List
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split

class DataMediator:
    """
    Split & balance a dataframe with shape (sample, variables) into `experimentalData`
    and `controlData` class variables.

    TODO: Filtering may be done with a user-defined mapping of conditions to columns.

    [Args]
        `dataframe` (DataFrame): Data to split, balance, and take holdout \n
        `IDlabel` (str): Column name of IDs in the dataframe. \n
        `controlIDs` (list[str]): List of sample IDs in the control set. \n
        `experimentalIDs` (list[str]): List of sample IDs in the experimental set. \n

    [Optional]
        `holdout` (float): Proportion of data from experimental & control sets to holdout. Creates `experimentalHoldout` and `controlHoldout` class variables. \n
        `balancingMethod` (str): Sampling method used for data balancing. The default is "downsampling"; "upsampling" or "smote" are additional options. Sampling occurs via the Pandas `sample()` method, unless another callback is defined. \n
        `balancingMethodCallback` (Callable): A custom sampling method callback, which is called instead of Pandas sample() on control, experimental & holdout dataframes. \n
        `filterMap`: TODO \n
        `verbose` (bool): Flag that determines whether DataMediator logs activity to STDOUT. \n
    """

    def __init__(
        self,
        dataframe: DataFrame,
        IDlabel: str,
        controlIDs: list,
        experimentalIDs: list,
        **kwargs,
    ) -> None:
        self.dataframe = dataframe
        self.IDlabel = IDlabel

        # split experimental & control data with given IDs
        self.experimentalData = self.__splitDataFrame(experimentalIDs)
        self.controlData = self.__splitDataFrame(controlIDs)

        # record initial state of dataframes
        self.__experimentalData = self.experimentalData.copy(deep=True)
        self.__controlData = self.controlData.copy(deep=True)

        # set flags
        # take holdouts before any data balancing
        if "holdout" in kwargs:
            self.holdoutProportion = kwargs["holdout"]
            self.__createHoldout(self.holdoutProportion)
            del kwargs["holdout"]
        for flag, value in kwargs.items():
            # TODO: refactor using match case upon python 3.10 release
            if flag == "balancingMethod":
                # balance with given sampling method
                self.balancingMethod = value
                if "balancingMethodCallback" in kwargs:
                    self.balancingMethodCallback = kwargs["balancingMethodCallback"]
                    self.__balance(value, self.balancingMethodCallback)
                else:
                    self.__balance(value)
            if flag == "filterMap":
                self.filter(value)
            if flag == "verbose":
                self.verbose = value

    @staticmethod
    def transposeDataFrame(
        dataframe,
        columnToTranspose: str,
        columnsToDrop: Union[List[str], slice] = None,
    ) -> DataFrame:
        """
        Static method to transpose a dataframe by a given column,
        with a new row index.
        """

        # drop transposed column IDs from data
        transposed = dataframe.drop(columns=dataframe.columns[columnsToDrop]).T.iloc[1:, :]
        columnIDs = dataframe[columnToTranspose].tolist()
        
        # set new indices
        transposed.columns = columnIDs

        return transposed

    def __splitDataFrame(self, IDs: list) -> DataFrame:
        """
        Private method to excise a list of desired sample IDs from the
        dataframe attached to this instance, into experimental
        & control dataframes.
        """

        return self.dataframe.loc[[ID for ID in IDs if ID in self.dataframe.index]]

    def __createHoldout(self, testSize: float) -> None:
        """
        Private method to randomly sequester samples into holdout dataframes.
        Sequestered data is excluded from the experimental & control class variables.

        Holdout dataframes may be accessed by the `experimentalHoldout` and
        `controlHoldout` class variables.
        """
        if testSize > 1 or testSize < 0:
            raise ValueError("Proportion must be in the range of (0, 1)")

        self.controlHoldout = self.controlData.sample(
            int(len(self.controlData) * testSize)
        )
        self.experimentalHoldout = self.experimentalData.sample(
            int(len(self.experimentalData) * testSize)
        )

        # ignore chained assigment warning in Pandas since we are dropping rows in-place
        pd.options.mode.chained_assignment = None
        # remove holdouts from experimental & control data
        self.controlData.drop(self.controlHoldout.index, inplace=True)
        self.experimentalData.drop(self.experimentalHoldout.index, inplace=True)
        # restore chained assignment warning
        pd.options.mode.chained_assignment = "warn"

    def __balance(
        self,
        balancingMethod: str = "downsampling",
        balancingMethodCallback: Optional[Callable] = None,
    ) -> None:
        """
        Private method to balance control, experimental & holdout datasets, with a
        given sampling method. The default is "downsampling"; "upsampling" or "smote"
        are additional options. A custom sampling method callback may also be passed,
        which is called instead on split & holdout dataframes.
        """
        largeSplit = max([self.experimentalData, self.controlData], key=len)
        smallSplit = min([self.experimentalData, self.controlData], key=len)

        if hasattr(self, "verbose"):
            print(f"Unbalanced classes— {False if largeSplit == smallSplit else True}")
            print(f"Minority split count: {len(largeSplit)}")
            print(f"Majority split count: {len(smallSplit)}")

        # Balancing the Data
        # ignore chained assigment warning in Pandas since we are dropping rows in-place
        pd.options.mode.chained_assignment = None
        if balancingMethodCallback and balancingMethod == "downsampling":
            dataToDrop = balancingMethodCallback(largeSplit)
            largeSplit.drop(
                dataToDrop.index.symmetric_difference(largeSplit.index), inplace=True
            )
        elif balancingMethodCallback and balancingMethod == "upsampling":
            dataToAdd = balancingMethodCallback(smallSplit)
            smallSplit.merge(dataToAdd)
        elif balancingMethod == "downsampling":
            dataToDrop = largeSplit.sample(len(smallSplit))
            largeSplit.drop(
                dataToDrop.index.symmetric_difference(largeSplit.index), inplace=True
            )
        elif balancingMethod == "upsampling":
            dataToAdd = smallSplit.sample(len(largeSplit), replace=True)
            smallSplit.merge(dataToAdd)
        # restore chained assignment warning
        pd.options.mode.chained_assignment = "warn"
        # TODO: implement smote

    @property
    def featureCount(self, columnStratified=True):
        """
        Return number of features from the input dataframe. If features are
        stratified by row, set `columnStratified` to falsy.
        """
        return self.dataframe.shape[1 if columnStratified else 0]

    def resample(self, keepFilters=False) -> None:
        """
        Reinitialize experimental & control data; redo holdout sequestration
        and dataset balancing.
        """
        if not keepFilters:
            self.experimentalData = self.__experimentalData.copy(deep=True)
            self.controlData = self.__controlData.copy(deep=True)
        if self.balancingMethod:
            self.__balance(self.balancingMethod, self.balancingMethodCallback)
        if self.holdoutProportion:
            self.__createHoldout(self.holdoutProportion)

    def trainTestSplit(self, testSize: float) -> None:
        """
        Split experimental and control data by a given testSize into training/testing sets, with
        classification targets.

        [Input]
            `testSize`: proportion of data to split for testing, between 0 and 1 \n
        [New attributes]
            `trainingData` \n
            `trainingLabels` \n
            `testingData` \n
            `testingLabels` \n
            `trainTestIndex` \n
        """
        allData = pd.concat([self.controlData, self.experimentalData])

        totalLabels = np.array(
            [0] * len(self.controlData) + [1] * len(self.experimentalData)
        )
        self.trainTestIndex = allData.index.tolist()
        allData.reset_index(drop=True, inplace=True)
        (
            self.trainingData,
            self.testingData,
            self.trainingLabels,
            self.testingLabels,
        ) = train_test_split(allData.astype(float), totalLabels, test_size=testSize)

    def loadPredictions(self, predictedLabels: list, testData: str = True):
        """
        Index an array-like of numeric predictions into a dataframe. Training & testing
        data must be split before using this method.

        [Input]
            `testData`: Boolean to indicate whether predictions are obtained from training or testing. 
            
        [New attributes]
            `predictions` 
        """
        self.predictions = pd.DataFrame(predictedLabels)
        self.predictions.index = [
            self.trainTestIndex[i] for i in (self.testingData if testData else self.trainingData).index.tolist()
        ]

        self.predictions["y_real"] = self.testingLabels if testData else self.trainingLabels
        self.predictions["y_pred"] = np.argmax(predictedLabels, axis=1)

    def filterByMetric(self, column, lowerBound=0, upperBound=float("inf")) -> None:
        """
        Filter samples in experimental & control dataframes using a predefined
        condition-column mapping.
        """
        self.experimentalData = self.experimentalData.loc[
            self.experimentalData[column >= lowerBound] &
            self.experimentalData[column <= upperBound]
        ]
        self.controlData = self.controlData.loc[
            self.controlData[column >= lowerBound] &
            self.controlData[column <= upperBound]
        ]
        self.resample(keepFilters=True)
