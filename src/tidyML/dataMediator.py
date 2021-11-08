"""
Contains a data mediator to split, balance, and holdout experimental
and control data for machine learning pipelines.
"""

from typing import Callable, Optional, Union
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split


class DataMediator:
    """
    Split & balance a dataframe with shape (sample, variables) into `experimentalData`
    and `controlData` class variables. Holdouts may also be taken by defining a
    proportion of data to sequester from experimental & control sets, which are
    accessible via the `experimentalHoldout` and `controlHoldout` class variables.

    TODO: Filtering may be done with a user-defined mapping of conditions to columns.

    Args:
        `dataframe` (DataFrame): Data to split, balance, and take holdouts from.
        `IDlabel` (str): Column name of IDs in the dataframe.
        `controlIDs` (list[str]): List of sample IDs in the control set.
        `experimentalIDs` (list[str]): List of sample IDs in the experimental set.

    Kwargs:
        `holdout` (float): Proportion of data from experimental & control sets to holdout.
        `balancingMethod` (str): Sampling method used for data balancing. The default is
            "downsampling"; "upsampling" or "smote" are additional options. Sampling occurs via
            the Pandas `sample()` method, unless another callback is defined.
        `balancingMethodCallback` (Callable): A custom sampling method callback,
            which is called instead of Pandas sample() on control, experimental & holdout
            dataframes.
        `filterMap`: TODO
        `verbose` (bool): Flag that determines whether DataMediator logs activity to
            STDOUT.
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
        newColumnIndex: str,
        newRowIndex: str,
    ) -> DataFrame:
        """
        Static method to transpose a dataframe by a given column,
        with a new row index.
        """
        columnIDs = dataframe[newColumnIndex].tolist()
        # drop transposed column IDs from data
        transposed = dataframe.T.iloc[1:, :]

        # set new indices
        transposed.columns = columnIDs
        transposed.index.name = newRowIndex

        return transposed

    def __splitDataFrame(self, IDs: list) -> DataFrame:
        """
        Private method to excise a list of desired sample IDs from the
        dataframe attached to this instance, into experimental
        & control dataframes.
        """

        return self.dataframe.loc[[ID for ID in IDs if ID in self.dataframe.index]]

    def __createHoldout(self, proportion: float) -> None:
        """
        Private method to randomly sequester samples into holdout dataframes.
        Sequestered data is excluded from the experimental & control class variables.

        Holdout dataframes may be accessed by the `experimentalHoldout` and
        `controlHoldout` class variables.
        """
        if proportion > 1 or proportion < 0:
            raise ValueError("Proportion must be in the range of (0, 1)")

        self.controlHoldout = self.controlData.sample(
            int(len(self.controlData) * proportion)
        )
        self.experimentalHoldout = self.experimentalData.sample(
            int(len(self.experimentalData) * proportion)
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
            print("Unbalanced classes")
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

    def resample(self) -> None:
        """
        Reinitialize experimental & control data; redo holdout sequestration
        and dataset balancing.
        """
        self.experimentalData = self.__experimentalData.copy(deep=True)
        self.controlData = self.__controlData.copy(deep=True)

        if self.balancingMethod:
            self.__balance(self.balancingMethod, self.balancingMethodCallback)
        if self.holdoutProportion:
            self.__createHoldout(self.holdoutProportion)

    def trainTestSplit(self, proportion: float, dropIndex: bool = True) -> None:
        """
        Split experimental and control data by a given proportion into training/testing sets, with
        classification targets. Access using class variables `trainingData`, `trainingLabels`,
        `testingData`, and `testingLabels`.
        """
        allData = pd.concat([self.controlData, self.experimentalData])
        totalLabels = np.array(
            [0] * len(self.controlData) + [1] * len(self.experimentalData)
        )
        if dropIndex:
            allData.reset_index(drop=True, inplace=True)
        (
            self.trainingData,
            self.testingData,
            self.trainingLabels,
            self.testingLabels,
        ) = train_test_split(allData.astype(float), totalLabels, test_size=proportion)

    def filter(self, filterMap: Union[DataFrame, dict]) -> None:
        """
        Filter samples in experimental & control dataframes using a predefined
        condition-column mapping.
        """
        # check if hasattr(self, 'experimentalData' | 'controlData') for filtering
