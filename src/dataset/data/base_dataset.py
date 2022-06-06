from abc import ABC, abstractmethod

class Dataset(ABC):
    def __init__(self, root_path, validation_split_ratio=0.2, test_split_ratio=0.25):
        self.root_path = root_path
        self.x = None # np.ndarray
        self.y = None # np.ndarray
        self.train_split_ratio = 1 - (validation_split_ratio + test_split_ratio)
        self.validation_split_ratio = validation_split_ratio
        self.test_split_ratio = test_split_ratio

    @abstractmethod
    def load(self):
        '''
            Initializes self.x and self.y numpy arrays which will contain the dataset given in the root_path.
        '''

    @abstractmethod
    def split(self, mode, shuffle, seed):
        '''
            Splits the dataset.
        '''