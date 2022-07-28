from abc import ABC, abstractmethod

class Network(ABC):
    """
    Abstract class for the network implementations.
    """
    def __init__(self, *args, **kwargs):
        pass

    def train(self, x_train, y_train, x_val, y_val):
        """
        Trains the model.
        """
        pass

    def predict(self, x_test):
        """
        Inference on the given test data.
        """
        pass