from .base_dataloader import DataLoader
import numpy as np

class RegressorDataLoader(DataLoader):
    def __init__(self, x_data, y_data, batch_size):
        super().__init__(x_data, y_data, batch_size)
        self.num_batches = np.ceil(len(x_data) / batch_size)
        self.batch_idx = np.array_split(range(len(x_data)), self.num_batches)

    def __len__(self):
        return len(self.batch_idx)

    def __getitem__(self, idx):
        batch_x = self.x[self.batch_idx[idx]]
        batch_y = self.y[self.batch_idx[idx]]
        return batch_x, batch_y