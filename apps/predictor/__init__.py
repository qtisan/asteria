# Predictor initialize.

from abc import ABCMeta, abstractmethod
from typing import TypeVar

import numpy as np
import pandas as pd

from tpot import TPOTClassifier, TPOTRegressor
from tpot.base import TPOTBase

from utils.functional import hooked

classifier = TPOTClassifier(generations=2,
                            population_size=10,
                            verbosity=2,
                            random_state=42,
                            max_time_mins=1)
regressor = TPOTRegressor(generations=2,
                          population_size=10,
                          verbosity=2,
                          random_state=42,
                          max_time_mins=1)

Indicator = TypeVar('Indicator', str, list)


class IProcessor(metaclass=ABCMeta):
    @abstractmethod
    def get_solution(self, indicator: Indicator, name=None) -> dict:
        '''
        Get solution by indicator and solution name.
        
        Parameters:
        ---
        indicator : `Indicator`
        name: `str` or `list` according to the indicator.
        
        Returns:
        ---
        solution : `dict`
        '''
        pass

    @abstractmethod
    def get_original_data(self, indicator: Indicator) -> pd.DataFrame:
        '''
        Read the original data from disk or website, etc...
        
        Parameters:
        ---
        indicator : `any` 
            Some configuration indicate where or what to read.
            
        Returns:
        ---
        dataset : `DataFrame`
            The original dataset.
        '''
        pass

    @abstractmethod
    def get_y_name(self, y_value, col=0) -> str:
        '''
        Get y name by value, with the column name or index. 
        Useful for classify, maybe category name or anything else. 
        Maybe original value in regress, or some formatted string 
        for display.
        
        Parameters:
        ---
        y_value : `any`
            The value of y.
        col : `int` or `str`
            `int` for index, and `str` for column name in `pd.DataFrame`.
            
        Returns:
        ---
        y_name : `str`
            y category name in classify or some formatted string in regress.
        '''
        pass


def create_processor():
    '''
    Factory Method for creating processor with default actions.
    '''
    class Processor(IProcessor):
        pass

    return Processor()


class Predictor(object):
    def __init__(self, processor: IProcessor):
        self.processor = processor
        self._hooks = {
            'before_start': [],
            'data_ready': [],
            'xys_made': [],
            ''
        }
        pass

    def regress(self, indicator=None, solution_name: str = None):
        pass

    def classify(self, indicator=None, solution_name: str = None):
        pass

    def predict(self, indicator=None, solution_name: str = None):
        pass
