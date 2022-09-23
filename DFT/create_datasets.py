import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset


class ComplexNumbersDataset(Dataset):
    def __init__(self, nb_batches, batch_size, N, complex=True):
        self.nb_batches = nb_batches
        self.batch_size = batch_size
        if (complex==True):
            self.samples = np.random.randn(nb_batches, batch_size, N) + 1j*np.random.randn(nb_batches, batch_size, N)
        else:
            #dc_offset = np.ones((nb_batches, batch_size, N), dtype=float )
            #scale_factor = np.random.uniform(0.0, 0.125, 1)
            self.samples = np.random.randn(nb_batches, batch_size, N)  
            #self.samples = np.zeros((nb_batches, batch_size, N), dtype=float)

            #for b in range (nb_batches):
            #    dc_offset = np.ones((batch_size, N), dtype=float )
            #    scale_factor = np.random.uniform(0.01, 0.125, 1)
            #    white_noise = np.random.randn(batch_size, N)
            #    self.samples[b,:, :] = white_noise

        print(f'size of samples = {(self.samples).shape}')
    #
    def __len__(self):
        return self.nb_batches
    #
    def __getitem__(self, idx):
        return self.samples[idx, :, :]


if __name__ == '__main__':
    dataset = ComplexNumbersDataset(3, 2, 10)   # create a dataset of complex numbers
    print(f'len = {len(dataset)}')
    #print(f' shape of 1st batch = {(dataset[0,:,:]).shape}')
    print(f' dataset of 0 = {dataset[0]}')
    
