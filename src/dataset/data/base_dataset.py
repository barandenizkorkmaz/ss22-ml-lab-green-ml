from abc import ABC, abstractmethod

import pandas as pd

class Dataset(ABC):
    def __init__(self, file_path, subset):
        self.file_path = file_path
        self.subset = subset
        self.load()

    @abstractmethod
    def load(self):
        """
        Read the raw_data as pd.dataframe given that is located in the field 'path' and set the field self.raw_data.
        """

    @abstractmethod
    def prepare(self,*args):
        """
            Initializes self.x and self.y numpy arrays which will contain the dataset given in the root_path.

            Returns:
                x, y

        """

    @abstractmethod
    def preprocessing(self, *args):
        """
            Preprocesses the dataset so that it can be fed for training and testing.

            Args:
                x, y

            Returns:
                x, y

        """