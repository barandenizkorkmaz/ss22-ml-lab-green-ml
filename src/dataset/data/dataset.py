from .base_dataset import Dataset

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

class RegressorDataset(Dataset):
    def __init__(self, root_path, validation_split_ratio, test_split_ratio):
        super().__init__(root_path, validation_split_ratio, test_split_ratio)
        self.load()

    def load(self):
        x = list()
        y = list()
        for root, directories, files in os.walk(self.root_path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                if file == 'x.txt':
                    x.append(json.load(open(file_path))['x'])
                if file == 'y.txt':
                    y.append(json.load(open(file_path))['y'])
        self.x = np.array(x, dtype=int)
        self.y = np.array(y, dtype=float)
        del x,y

    def split(self, mode, shuffle, seed):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.test_split_ratio, random_state=seed, shuffle=shuffle)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.validation_split_ratio, random_state=None, shuffle=False)
        if mode == "train":
            return x_train, x_test
        elif mode == "validation":
            return x_val, y_val
        elif mode == "test":
            return x_test, y_test
        else:
            raise TypeError()
