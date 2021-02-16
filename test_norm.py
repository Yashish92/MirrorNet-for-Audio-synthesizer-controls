#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import argparse

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

import os
import numpy as np
import h5py
import time
import subprocess
import logging

from datetime import date
import matplotlib.pyplot as plt
import pickle

#get_ipython().run_line_magic('matplotlib', 'inline')

#get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES = 0')

# In[61]:


class BoDataset(Dataset):
    """
        Wrapper for the WSJ Dataset.
    """
    
    def __init__(self, path):
        super(BoDataset, self).__init__()

        self.h5pyLoader = h5py.File(path, 'r')
        self.wav = self.h5pyLoader['speaker']
        self.spec = self.h5pyLoader['spec513']
        self.specWorld = self.h5pyLoader['spec513_pw']
        self.hidden = self.h5pyLoader['hidden']
        self._len = self.wav.shape[0]
    
    def __getitem__(self, index):

        wav_tensor = torch.from_numpy(self.wav[index])   # raw waveform with normalized power
        spec_tensor = torch.from_numpy(self.spec[index])         # magnitude of spectrogram of raw waveform
        hidden_tensor = torch.from_numpy(self.hidden[index])         # parameters of world
        specWorld_tensor = torch.from_numpy(self.specWorld[index])
            
        return wav_tensor, spec_tensor, hidden_tensor, specWorld_tensor
    
    def __len__(self):
        return self._len

print("Loading data")
training_data_path = "data/"+"train.data"
train_loader = DataLoader(BoDataset(training_data_path), 
                  batch_size=int(1e32), 
                  shuffle=False)

print("start")
for batch_idx, data in enumerate(train_loader):
    batch_h = data[2]
    f0 = batch_h[:,0:1,:].data.cpu().numpy().astype('float64')
    sp = batch_h[:,1:514,:].data.cpu().numpy().astype('float64')
    ap = batch_h[:,514:,:].data.cpu().numpy().astype('float64')


    print(ap.shape)

    for i in range(len(ap)):
        plt.plot(ap[i,1,:])
    
        plt.show()







