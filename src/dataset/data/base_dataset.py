from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, path):
        self.path = path
        self.raw_data = None
        self.x = None
        self.y = None

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
    def split(self, *args):
        '''
            Splits the dataset.
        '''