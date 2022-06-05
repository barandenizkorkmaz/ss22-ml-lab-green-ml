from abc import ABC, abstractmethod

class Dataset(ABC):
    def __init__(self, root_path):
        self.root_path = root_path
        self.x = None # np.ndarray
        self.y = None # np.ndarray

    @abstractmethod
    def load(self):
        '''
            Initializes self.x and self.y numpy arrays which will contain the dataset given in the root_path.
        '''