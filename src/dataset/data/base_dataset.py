from abc import ABC, abstractmethod

import pandas as pd

class Dataset(ABC):
    def __init__(self, *args, **kwargs):
        """
        Args:
            *args:
            **kwargs:
        """
        self.x_train, self.y_train = None, None
        self.x_validation, self.y_validation = None, None
        self.x_test, self.y_test = None, None

    @abstractmethod
    def load(self):
        """
        Read the raw_data as pd.dataframe given that is located in the field 'path' and set the field self.raw_data.
        """

    def get_train_set(self):
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """
        return self.x_train, self.y_train

    def get_validation_set(self):
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """
        return self.x_validation, self.y_validation

    def get_test_set(self):
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """
        return self.x_test, self.y_test