from abc import ABC, abstractmethod

class Network(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def train(self, x_train, y_train, x_val, y_val):
        pass

    def predict(self, x_test):
        pass