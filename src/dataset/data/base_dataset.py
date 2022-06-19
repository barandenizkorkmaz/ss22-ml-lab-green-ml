from abc import ABC, abstractmethod

import pandas as pd

class Dataset(ABC):
    def __init__(self, *args, **kwargs):
        """
        Args:
            *args:
            **kwargs:
        """

    @abstractmethod
    def load(self):
        """
        Read the raw_data as pd.dataframe given that is located in the field 'path' and set the field self.raw_data.
        """

    @abstractmethod
    def prepare(self, *args, **kwargs):
        """
            Initializes self.x and self.y numpy arrays which will contain the dataset given in the root_path.

            Returns:
                x, y

        """

    @abstractmethod
    def preprocessing(self, *args, **kwargs):
        """
            Preprocesses the dataset so that it can be fed for training and testing.

            Args:
                x, y

            Returns:
                x, y

        """