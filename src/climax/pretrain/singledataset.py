import numpy as np
import torch
from torch.utils.data import Dataset

class SingleNPZDataset(Dataset):
    def __init__(self, npz_path, variables, out_variables):
        data = np.load(npz_path)
        self.variables = variables
        self.out_variables = out_variables

        self.inputs = np.concatenate([data[var] for var in variables], axis=1)  # [T, C_in, H, W]
        self.outputs = np.concatenate([data[var] for var in out_variables], axis=1)  # [T, C_out, H, W]

        self.length = self.inputs.shape[0]

    def __len__(self):
        return self.length - 1  # next-day prediction

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.outputs[idx + 1]).float()
        return x, y