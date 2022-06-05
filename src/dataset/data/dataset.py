from base_dataset import Dataset

import os
import json
import numpy as np

class RegressorDataset(Dataset):
    def __init__(self, root_path):
        super().__init__(root_path)
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