#!/usr/bin/env python
# coding: utf-8

# March 22: Only change from the old code (10th Feb code) is removing the Sp=exp(SP) function

from __future__ import print_function
import argparse
import time

import matplotlib

SERVER = True  # This variable enable or disable matplotlib, set it to true when you use the server!
if SERVER:
    matplotlib.use('Agg')

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

# from IPython.display import Audio
# import pyworld as pw
import librosa

from datetime import date
import matplotlib.pyplot as plt
import pickle as pkl
import datetime

from scipy.io.wavfile import write

from random_generation import get_f0, get_ap
import inspect
import nsltools as nsl
import music_synthesize_DIVA_melodies_7params as music_syn
from scipy.io import savemat

# get_ipython().run_line_magic('matplotlib', 'inline')

# get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES = 0')

# In[61]:
spec_len = 250
N = 5
params = 7    # number of parameters in the latent space
params_ideal = 10   # parameters used to generate the training set
pad_render = 0.1


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
        # print (path)
        print('h5py loader', type(self.h5pyLoader))
        print('wav', self.wav)
        print('spec', self.spec)
        print('self.specWorld', self.specWorld)
        print('self.hidden', self.hidden)

    def __getitem__(self, index):
        # print (index)
        wav_tensor = torch.from_numpy(self.wav[index])  # raw waveform with normalized power
        # print (len(self.hidden[index]))
        # spec_tensor = torch.from_numpy(self.spec[index])         # magnitude of spectrogram of raw waveform
        if len(self.hidden[index]) > 0:
            hidden_tensor = torch.from_numpy(self.hidden[index])  # parameters of world
        else:
            hidden_tensor = []
        specWorld_tensor = torch.from_numpy(self.specWorld[index])

        return wav_tensor, [], hidden_tensor, specWorld_tensor

    def __len__(self):
        return self._len


def log_print(content):
    print(content)
    logging.info(content)


# In[70]:


class CNN1D(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel,
                 dilation=1, stride=1, padding=0, ds=2):
        super(CNN1D, self).__init__()

        self.causal = False

        if self.causal:
            self.padding1 = (kernel - 1) * dilation
            self.padding2 = (kernel - 1) * dilation * ds
        else:
            self.padding1 = (kernel - 1) // 2 * dilation
            self.padding2 = (kernel - 1) // 2 * dilation * ds

        self.conv1d1 = nn.Conv1d(input_channel, hidden_channel, kernel,
                                 stride=stride, padding=self.padding1,
                                 dilation=dilation)
        self.conv1d2 = nn.Conv1d(input_channel, hidden_channel, kernel,
                                 stride=stride, padding=self.padding2,
                                 dilation=dilation * ds)

        self.reg1 = nn.BatchNorm1d(hidden_channel, eps=1e-16)
        self.reg2 = nn.BatchNorm1d(hidden_channel, eps=1e-16)
        # self.reg1 = GlobalLayerNorm(hidden_channel)
        # self.reg2 = GlobalLayerNorm(hidden_channel)

        self.nonlinearity = nn.ReLU()

    def forward(self, input):
        # res=input

        output = self.nonlinearity(self.reg1(self.conv1d1(input)))
        if self.causal:
            output = output[:, :, :-self.padding1]
        output = self.reg2(self.conv1d2(output))
        if self.causal:
            output = output[:, :, :-self.padding2]
        output = self.nonlinearity(output + input)
        # output=output+res

        return output


# In[71]:
EPS = 1e-8


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        return gLN_y


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        # self.AE_win = 320
        # self.AE_stride = 80
        self.AE_channel = 256
        self.BN_channel = 256

        self.AE_win = 320
        self.AE_stride = 128
        # self.AE_channel = 128
        # self.BN_channel = 128

        self.kernel = 3
        self.CNN_layer = 4
        self.stack = 3

        self.L1 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.Conv1d(128, 256, 1),
            nn.GroupNorm(1, 256, eps=1e-16),
            nn.Conv1d(256, 256,1)       # added new
            #nn.BatchNorm1d(128, 128, 1),  # added new
            #nn.ReLU()
        )
    #self.encoder = nn.Conv1d(128, 128, 1)

        #self.L1 = nn.Conv1d(128, 256, 1)
        # self.encoder = nn.Conv1d(1, self.AE_channel, self.AE_win,
        #                          stride=self.AE_stride, bias=False)
        #self.enc_Norm = nn.GroupNorm(1, 256, eps=1e-16)

        #self.L2 = nn.Conv1d(256, 256, 1)

        self.CNN = nn.ModuleList([])
        for s in range(self.stack):
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 0))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 2))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 4))

        # self.f0_seq = nn.Sequential(
        #     nn.Conv1d(self.BN_channel, 64, 1),  # Kernel size 1 with 0 padding maintains length of output signal=401
        #     nn.ReLU(),
        #     nn.Conv1d(64, 16, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(16, 1, 1),
        #     nn.Sigmoid()
        # )
        # self.sp_seq = nn.Sequential(
        #     nn.Conv1d(self.BN_channel, 256, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(256, 513, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(513, 513, 1),
        #     nn.Tanh()
        #     # nn.ReLU(),
        #
        # )

        self.L2 = nn.Sequential(
            nn.Conv1d(self.BN_channel, 256,1, bias=False),
            nn.BatchNorm1d(256, 256, 1),
            nn.ReLU(),
            nn.AvgPool1d(5, stride=5),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128, 128, 1),
            nn.ReLU(),
            nn.AvgPool1d(5, stride=5)
            # nn.Conv1d(128, params, 1, bias=False),
            # nn.BatchNorm1d(params, params, 1),         ## adding filter resonance
            # nn.AvgPool1d(2, stride=2),
            # nn.Sigmoid()
        )

        self.last_layers = nn.Sequential(
            nn.Conv1d(128, params, 1, bias=False),
            nn.BatchNorm1d(params, params, 1),  ## adding filter resonance
            nn.AvgPool1d(2, stride=2),
            nn.Sigmoid()
        )


    def forward(self, input):
        # input shape: B, T
        batch_size = input.size(0)
        nsample = input.size(1)
        # print (input.size(), ' encoder input.size() encoder') #64-32320

        # pad_num = 160
        # pad_aux = Variable(torch.zeros(batch_size, pad_num)).type(input.type())
        # input = torch.cat([pad_aux, input, pad_aux], 1)
        # output = input.unsqueeze(1)  # B, 1, T=32320
        # print (output.size(), 'encoder output')

        # encoder
        this_input = self.L1(input)  # B, C, T #T=[{(32320-320)/80}+1]=401
        #print(enc_output.shape)

        # enc_output = self.encoder(output) # B, C, T #T=[{(32320-320)/80}+1]=401
        # Shape is now 64-256-401
        # print (enc_output.size(), 'enc_output.size()') #B-256-T

        # temporal convolutional network (TCN)
        #this_input = self.L1(self.enc_Norm(enc_output))
        #this_input = self.L1(self.enc_Norm(enc_output))

        # print (this_input.size(), 'this_input after L1, before stack. Encoder') #64-256-401
        for i in range(len(self.CNN)):
            this_input = self.CNN[i](this_input)
        # print (this_input.size(), 'this_input after CNN stack. Ecnoder-------') #64-256-401

        # f0 = self.f0_seq(this_input) * 800  # limit the F0 range in [0, 800] #64-1-401
        # sp = self.sp_seq(this_input) * 30  # limit the log(sp) in [-50, 50]   #64-513-401
        # sp = self.sp_seq(this_input) + 1e-20    # limit the log(sp) in [-50, 50]   #64-513-401

        music = self.L2(this_input)  # 64-513-401
        music_last = self.last_layers(music)

        return music_last


# In[72]:


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.AE_win = 320
        self.AE_stride = 80
        self.AE_channel = 256
        self.BN_channel = 256
        #self.BN_channel = 128
        self.kernel = 3
        self.CNN_layer = 4
        self.stack = 3

        # self.encoder = nn.Conv1d(1, self.AE_channel, self.AE_win,
        #                          stride=self.AE_stride, bias=False)

        # self.L1 = nn.Sequential(
        #                         nn.Conv1d(1+513*2, 256, 1)

        # )
        self.begin_layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(params, 128, 1, bias=False),  # changed 1 + 513 * 2 to 16
            nn.BatchNorm1d(128, 128, 1),
            nn.ReLU()
        )


        self.L1 = nn.Sequential(

            # nn.Upsample(scale_factor=2),
            # nn.Conv1d(params, 128, 1, bias=False),  # changed 1 + 513 * 2 to 16
            # nn.BatchNorm1d(128, 128, 1),
            # nn.ReLU(),
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 256, 1, bias=False),  # changed 1 + 513 * 2 to 16
            nn.BatchNorm1d(256, 256, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=5),
            nn.Conv1d(256, 256, 1, bias=False),  # changed 1 + 513 * 2 to 16
            nn.BatchNorm1d(256, 256, 1),
            nn.ReLU()

            #nn.Upsample(scale_factor=12)
            #nn.ConvTranspose1d(4, 128, kernel_size=51),  # changed 1 + 513 * 2 to 16
            #nn.BatchNorm1d(128, 128, 1),
            #nn.ReLU(),
            #nn.ConvTranspose1d(128, 256, kernel_size=99),  # changed 1 + 513 * 2 to 16
            #nn.BatchNorm1d(256, 256, 1),
            #nn.ReLU(),
            #nn.ConvTranspose1d(256, 256, kernel_size=100),  # changed 1 + 513 * 2 to 16
            #nn.BatchNorm1d(256, 256, 1),
            #nn.ReLU()


        )

        self.CNN = nn.ModuleList([])
        for s in range(self.stack):
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 0))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 2))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 4))

        # self.L2 = nn.Sequential(
        #                         nn.Conv1d(self.AE_channel, 256, 1)

        # )

        self.L2 = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.GroupNorm(1, self.AE_channel, eps=1e-16),
            nn.Conv1d(256, 128,1),
            nn.Conv1d(128, 128,1)	# added new
            #nn.BatchNorm1d(128, 128, 1),  # added new
            #nn.ReLU()    		# added new

        )

    def forward(self, input):
        # input shape: B, T
        batch_size = input.size(0)
        nsample = input.size(1)

        #
        # print (input.size(), 'decoder input.size()')    #64-1027-251 shape of latent space
        #print(input.size)
        #input = input.unsqueeze(dim=3)
        input1 = self.begin_layers(input)
        this_input = self.L1(input1)  # 64-128-250
        #print(this_input.shape)
        # print (this_input.size(),'this_input before CNN stack decoder--------' )
        for i in range(len(self.CNN)):
            this_input = self.CNN[i](this_input)  # 64-128-250
            # print (this_input.size(), 'Size(this_input) inside CNN stack decoder')
        #print(this_input.shape)
        this_input = self.L2(this_input)
        #print (this_input.size()) #output is 64-128-250

        return this_input


criteron = nn.MSELoss().cuda()

class encoder_ext(nn.Module):
    def __init__(self):
        super(encoder_ext, self).__init__()
        self.E = encoder()
        # self.encoder = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
        self.E.load_state_dict(torch.load(model_dir + exp_new + 'net/music_model_weight_E1.pt'))
        # self.E.train()

        self.last_layers = nn.Sequential(
            nn.Conv1d(128, params, 1, bias=False),
            nn.BatchNorm1d(params, params, 1),  ## adding filter resonance
            nn.AvgPool1d(2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, input):

        enc_output = self.E.L1(input)  # B, C, T #T=[{(32320-320)/80}+1]=401
        for i in range(len(self.E.CNN)):
            this_input = self.E.CNN[i](enc_output)
        music = self.E.L2(this_input)  # 64-513-401
        music_last = self.last_layers(music)

        return music_last


class decoder_ext(nn.Module):
    def __init__(self):
        super(decoder_ext, self).__init__()
        self.D = decoder()
        # self.encoder = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
        self.D.load_state_dict(torch.load(model_dir + exp_new + 'net/music_model_weight_D1.pt'))
        # self.D.train()

        self.begin_layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(params, 128, 1, bias=False),  # changed 1 + 513 * 2 to 16
            nn.BatchNorm1d(128, 128, 1),
            nn.ReLU()
        )

    def forward(self, input):
        input1 = self.begin_layers(input)
        this_input = self.D.L1(input1)  # 64-128-250
        for i in range(len(self.D.CNN)):
            this_input = self.D.CNN[i](this_input)  # 64-128-250
            # print (this_input.size(), 'Size(this_input) inside CNN stack decoder')
        #print(this_input.shape)
        this_input = self.D.L2(this_input)

        return this_input


def transform_params(enc_out):
    # print(music)
    # print(music.shape)
    enc_out.data[:, 0] = (enc_out.data[:, 0] * 12) + 62
    enc_out.data[:, 1] = (enc_out.data[:, 1] * 0.3) + 0.7
    # music.data[:, 2] = music.data[:, 2]
    enc_out.data[:, 3] = (enc_out.data[:, 3] * 50) + 50

    return enc_out


def quickEval(epoch, ideal_h=False, train_E=True, train_D=True, loader="evaluation"):
    """
    Compute the loss function for the test dataset. Specifying ideal_h, train_E and train_D evaluate
    what the training with the same parameters is supposed to do, c.f. README for more details.
    """

    #    #with torch.no_grad():
    start_time = time.time()

    D.eval()
    E.eval()

    train_loss1 = 0.
    train_loss2 = 0.
    train_loss = 0.

    pad = 35

    if loader == "train":
        loader = train_loader
    elif loader == "evaluation":
        loader = validation_loader
    elif loader == "initialization":
        loader = initialization_loader
    elif loader == "train_random":
        loader = train_random

    # we can try to turn gradients off at this point to make the algorithm faster and then remember to turn them back again!
    # with torch.no_grad:
    for batch_idx, data in enumerate(loader):

        batch_wav = Variable(data[0]).contiguous().float()
        batch_h = Variable(data[2]).contiguous().float()
        batch_spec = Variable(data[3]).contiguous().float()  # We use the spec of the resynthesized spec

        if args.cuda:
            batch_wav = batch_wav.cuda()
            batch_spec = batch_spec.cuda()
            batch_h = batch_h.cuda()

        if train_E:
            # predict parameters through waveform
            # note (Cong): we can predict h through batch_spec, too. But I found waveform is better. batch_spec is derived from batch_wav.
            # batch_f0, batch_sp, batch_ap = E(batch_spec)
            batch_music = E(batch_spec)
            # batch_music = transform_params(batch_music)
            # batch_music[:][0][:] = (batch_music[:][0][:] * 12) +62
            # batch_music[:][1][:] = (batch_music[:][1][:] * 0.3) + 0.7
            # batch_music[:][2][:] = (batch_music[:][2][:] * 0.5) + 0.4
            # batch_music[:][3][:] = (batch_music[:][3][:] * 50) + 50

            # print(batch_music.shape)
            batch_h_pred = batch_music  # torch.cat([batch_f0, batch_sp, batch_music], 1)

        fs = 4000

        # print(batch_h.size(), 'batch_h.size()')
        # f0_temp = batch_h[:, 0:1, :].data.cpu().numpy().astype('float64')
        # sp_temp = batch_h[:, 1:514, :].data.cpu().numpy().astype('float64')
        music_temp = batch_h[:, params:, :].data.cpu().numpy().astype('float64')  # Doubtful about the shape

        #### TRAINING ####

        if train_D:
            s1 = D(batch_h)
            # DECODER
            loss2 = criteron(s1, batch_spec)
            train_loss2 += loss2.data.item()

        if train_E:
            if not ideal_h:
                s1_pred = D(batch_h_pred)
                # ENCODER
                loss1 = criteron(s1_pred, batch_spec)
                # FIX ENCODER many to one
                # loss3 = criteron(s1_pred, s2)
            else:
                loss1 = criteron(batch_h, batch_h_pred)
                # loss1 = (loss1 + loss3)/2

            # loss1 = loss1.requires_grad=True
            train_loss1 += loss1.data.item()

        if train_E and train_D:
            loss = loss1 + loss2
            train_loss += loss.data.item()

        # if (batch_idx+1) % args.log_step == 0:
        #     elapsed = time.time() - start_time
        #     log_print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | eval loss1 (Encoder) {:5.8f} | eval loss2 (Decoder) {:5.8f} | eval loss {:5.8f} |'.format(
        #         epoch, batch_idx+1, len(train_loader),
        #         elapsed * 1000 / (batch_idx+1), train_loss1 / (batch_idx+1), train_loss2 / (batch_idx+1), train_loss / (batch_idx+1)))

    train_loss1 /= (batch_idx + 1)
    train_loss2 /= (batch_idx + 1)
    train_loss /= (batch_idx + 1)

    # train_loss1 /= len(loader)
    # train_loss2 /= len(loader)
    # train_loss /= len(loader)

    log_print('-' * 99)
    log_print(
        '    | end of evaluating epoch {:3d} | time: {:5.2f}s | eval loss1 (encoder) {:5.8f} | eval loss2 (decoder){:5.8f}|'.format(
            epoch, (time.time() - start_time), train_loss1, train_loss2))

    return train_loss, train_loss1, train_loss2


def new_training_technique(epoch, train_D=False, train_E=False, init=False):
    '''
    This function train the networks for one epoch using the new training technique.
    train_D and train_E can be specified to train only one network or both in the same time.
    More details about the loss functions and the architecture in the README.md
    '''
    start_time = time.time()

    D.eval()
    E.eval()

    if train_D:
        D.train()
    if train_E:
        E.train()

    train_loss1 = 0.
    train_loss2 = 0.
    train_loss = 0.
    new_loss = 0.

    pad = 10
    # mel_basis = librosa.filters.mel(16000, 1024, n_mels=256)
    # if args.cuda:
    #     mel_basis = torch.from_numpy(mel_basis).cuda().float().unsqueeze(0)
    # else:
    #     mel_basis = torch.from_numpy(mel_basis).float().unsqueeze(0)

    for (batch_idx, data_random), (_, data_train) in zip(enumerate(train_random), enumerate(train_loader)):

        batch_wav_random = Variable(data_random[0]).contiguous().float()
        batch_h_random = Variable(data_random[2]).contiguous().float()
        batch_spec_random = Variable(data_random[3]).contiguous().float()

        batch_wav_train = Variable(data_train[0]).contiguous().float()
        batch_spec_train = Variable(data_train[3]).contiguous().float()

        if args.cuda:
            batch_wav_random = batch_wav_random.cuda()
            batch_spec_random = batch_spec_random.cuda()
            batch_h_random = batch_h_random.cuda()
            batch_wav_train = batch_wav_train.cuda()
            batch_spec_train = batch_spec_train.cuda()

        fs = 8000

        # predict parameters through waveform
        # note: we can predict h through batch_spec, too. But I found waveform is better. batch_spec is derived from batch_wav.
        h_hat = E(batch_spec_train)
        # h_hat = transform_params(h_hat)
        # print(batch_h_random)
        # h_hat = batch_music #torch.cat([batch_f0, batch_sp, batch_music], 1)

        if train_D and not init:
            # if not init:
            # f0_temp = batch_f0.data.cpu().numpy().astype('float64')
            # sp_temp = batch_sp.data.cpu().numpy().astype('float64')
            music_tmp = h_hat.data.cpu().numpy().astype('float64')
            # music_tmp[:, 0, :] = (music_tmp[:, 0, :] * 12) +62
            # music_tmp[:, 1, :] = (music_tmp[:, 1, :] * 0.3) + 0.7
            # music_tmp[:, 2, :] = (music_tmp[:, 2, :] * 0.5) + 0.4
            # music_tmp[:, 3, :] = (music_tmp[:, 3, :] * 50) + 50

            # print(music_tmp.shape[0])

            spec_wav = np.zeros((batch_wav_train.shape[0], 128, spec_len)).astype('float64')
            # spec_wav = np.zeros((batch_wav_train.shape[0], 128, 250)).astype('float64')
            frmlen = 8
            tc = 8
            sampling_rate = 8000.0
            paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000.0)]

            for j in range(music_tmp.shape[0]):
                # f0 = f0_temp[j, :, :].reshape(-1).copy(order='C')  # 406
                # sp = sp_temp[j, :, :].T.copy(order='C')  # 406, 513
                # sp = np.exp(sp)  # from log scale to original scale required by WORLD
                music = music_tmp[j, :, :].copy(order='C')  # 406, 513

                y = music_syn.generate(music, engine, generator, parameters, rev_idx, pad=pad_render)
                y = music_syn.resample(y, 44100, fs)  # resampling
                y = y[pad:pad + 32000]
                y = y / np.sqrt(np.sum(y ** 2))  # normalize the power of waveform to 1

                # time.sleep(0.4)

                # # transfer wavform to power of mel-spectrogram
                # spec_tmp = librosa.core.stft(y, n_fft=1024, hop_length = 80)
                # spec_tmp = np.abs(spec_tmp) ** 2
                # spec_tmp = np.dot(mel_b, spec_tmp)
                # spec_tmp = np.log(spec_tmp)
                # spec_tmp[spec_tmp == float("-Inf")] = -50
                # spec_wav[j,:,:] = spec_tmp
                spec_h_temp = (nsl.wav2aud(nsl.unitseq(y), paras_c)).T
                # print(spec_h_temp.shape[1])
                spec_wav[j, :, 0:spec_len] = (spec_h_temp)[:, 0:spec_len]

            # print("**************Outside for loop**********************")
            # print(spec_wav.shape)

            if args.cuda:
                ## s2 = W(h_hat)
                s2 = Variable(torch.from_numpy(spec_wav)).contiguous().cuda().float()
            else:
                s2 = Variable(torch.from_numpy(spec_wav)).contiguous().float()

        #### TRAINING ####
        # print (batch_h_random.size(), 'Size(batch_h_random)')
        # print (batch_spec_random.size(), 'Size(batch_spec_random)')
        # print ((D(batch_h_random)).size(), 'SizeDecoder((batch_h_random))')

        if train_D:
            D_optimizer.zero_grad(set_to_none=True)

            if init:
                loss2 = criteron(D(batch_h_random), batch_spec_random)
                loss2.backward()

            else:
                # loss_D1 = criteron(D(batch_h_random), batch_spec_random)
                loss_D2 = criteron(D(h_hat), s2)
                # print(loss_D2)
                # loss2 = (loss_D1 + loss_D2) / 2
                loss2 = loss_D2
                loss2.backward()

            # print(loss2)
            D_optimizer.step()
            train_loss2 += loss2.item()

        if train_E:
            E_optimizer.zero_grad(set_to_none=True)

            if init:
                c = E(batch_spec_random)
                # c = transform_params(c)
                # c[:][0][:] = (c[:][0][:]*12) +62
                # c[:][1][:] = (c[:][1][:]*0.3) +0.7
                # c[:][2][:] = (c[:][2][:]*0.5) +0.4
                # c[:][3][:] = (c[:][3][:]*50) +50

                loss1 = criteron(c, batch_h_random)

            else:
                loss_E1 = criteron(D(h_hat), batch_spec_train)
                # c = E(batch_spec_random)
                # loss_E2 = criteron(c, batch_h_random)

                # loss1 = (loss_E1 + loss_E2)/2
                loss1 = loss_E1

            # print(loss1)
            # loss1 = loss1.requires_grad=True
            loss1.backward()
            E_optimizer.step()
            train_loss1 += loss1.item()

        if train_E and train_D:
            loss = loss1 + loss2
            train_loss += loss.item()

        if train_D and not init:
            new_loss += loss_D2.item()

        # loss.backward()
        # all_optimizer.step()

        if (batch_idx + 1) % args.log_step == 0:
            elapsed = time.time() - start_time
            log_print(
                '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | train loss1 (Encoder) {:5.8f} | train loss2 (Decoder) {:5.8f} | total loss {:5.8f} |'.format(
                    epoch, batch_idx + 1, len(train_loader),
                           elapsed * 1000 / (batch_idx + 1), train_loss1 / (batch_idx + 1),
                           train_loss2 / (batch_idx + 1), train_loss / (batch_idx + 1)))

    train_loss1 /= (batch_idx + 1)
    train_loss2 /= (batch_idx + 1)
    train_loss /= (batch_idx + 1)
    # train_loss1 /= len(train_loader)
    # train_loss2 /= len(train_loader)
    # train_loss /= len(train_loader)
    print(len(train_loader), 'len(train_loader)')
    print("Losses calculated")

    log_print('-' * 99)
    log_print(
        '    | end of training epoch {:3d} | time: {:5.2f}s | train loss1 (encoder) {:5.8f} | train loss2 (decoder){:5.8f}|'.format(
            epoch, (time.time() - start_time), train_loss1, train_loss2))

    return train_loss, train_loss1, train_loss2, new_loss


def trainTogether_newTechnique(epochs=None, save_better_model=False, loader_eval="evaluation", train_E=False,
                               train_D=False, init=False, name=""):
    '''
    Glob function for the new training technique, it iterates over all the epochs,
    process the learning rate decay, save the weights and call the evaluation funciton.
    '''
    reduce_epoch = []

    training_loss_encoder = []
    training_loss_decoder = []

    validation_loss_encoder = []
    validation_loss_decoder = []

    new_loss = []

    decay_cnt1 = 0
    decay_cnt2 = 0

    if epochs is None:
        epochs = args.epochs

    print()

    if train_E and not train_D:
        print("TRAINING E ONLY")
    if not train_E and train_D:
        print("TRAINING D ONLY")
    if train_E and train_D:
        print("TRAINING E AND D")
    if not train_E and not train_D:
        print("TRAINING NOTHING ...")

    # Training all
    for epoch in range(1, epochs + 1):
        if args.cuda:
            E.cuda()
            D.cuda()

        error = new_training_technique(epoch, train_D=train_D, train_E=train_E, init=init)

        training_loss_encoder.append(error[1])
        training_loss_decoder.append(error[2])
        new_loss.append(error[3])
        print("Difference between D en W:", error[3])

        decay_cnt2 += 1

        if np.min(training_loss_encoder) not in training_loss_encoder[-3:] and decay_cnt1 >= 3:
            E_scheduler.step()
            decay_cnt1 = 0
            log_print('      Learning rate decreased for E.')
            log_print('-' * 99)

        if np.min(training_loss_decoder) not in training_loss_decoder[-3:] and decay_cnt2 >= 3:
            D_scheduler.step()
            decay_cnt2 = 0
            log_print('      Learning rate decreased for D.')
            log_print('-' * 99)

    del training_loss_encoder
    del training_loss_decoder

    #     if train_D and epoch % 5 == 0:  # chnged 5 to 1
    #         print("EVAL")
    #         total_loss_eval, encoder_loss_eval, decoder_loss_eval = quickEval(epoch, loader='evaluation',
    #                                                                           train_D=train_D, train_E=train_E,
    #                                                                           ideal_h=True)
    #         print('Decoder:---', decoder_loss_eval)
    #         print(
    #             '--------------------------------------------------------------------------------------------------------')
    #         print('Encoder:---', encoder_loss_eval)
    #         print("___________________")
    #
    #     if train_E and epoch % 5 == 0:  # chnged 5 to 1
    #         print("EVAL")
    #         total_loss_eval, encoder_loss_eval, decoder_loss_eval = quickEval(epoch, loader='evaluation',
    #                                                                           train_D=train_D, train_E=train_E,
    #                                                                           ideal_h=True)
    #         print('Decoder:---', decoder_loss_eval)
    #         print(
    #             '--------------------------------------------------------------------------------------------------------')
    #         print('Encoder:---', encoder_loss_eval)
    #         print("___________________")
    #
    #     if train_D and epoch % 10 == 0:  # changed 10 to 1
    #
    #         if save_better_model:
    #
    #             if train_E:
    #                 validation_loss_encoder.append(encoder_loss_eval)
    #                 if epoch >= 10 and np.min(validation_loss_encoder) == validation_loss_encoder[-1]:
    #                     with open(base_dir + exp_new + '/net/Gui_model_weight_E1.pt', 'wb') as f:
    #                         torch.save(E.cpu().state_dict(), f)
    #                     print("We saved encoder!")
    #                     print()
    #
    #             if train_D:
    #                 validation_loss_decoder.append(decoder_loss_eval)
    #                 # If we found a best model we save it!
    #                 if epoch >= 10 and np.min(validation_loss_decoder) == validation_loss_decoder[-1]:
    #                     with open(base_dir + exp_new + '/net/Gui_model_weight_D1.pt', 'wb') as f:
    #                         torch.save(D.cpu().state_dict(), f)
    #                     print("We saved decoder!")
    #                     print()
    #
    # if save_better_model and train_D:
    #     D.load_state_dict(torch.load('tmp/net/Gui_model_weight_D1.pt'))
    # if save_better_model and train_E:
    #     E.load_state_dict(torch.load('tmp/net/Gui_model_weight_E1.pt'))
    #
    #     # if train_E
    #
    # if train_D:
    #     print("Sequence of losses between D and W:")
    #     print(new_loss)
    #
    #     plt.title("Evolution of the difference between D(h_hat) and W(h_hat) during training")
    #     plt.plot(new_loss)
    #     plt.savefig(out + "/loss/loss_" + name + ".eps")
    #     plt.close()
    #
    # if train_D:
    #     plt.title("Decoder Loss Eval")
    #     plt.plot(decoder_loss_eval)
    #     plt.savefig(out + "/loss/decoder_loss_eval_" + name + ".eps")
    #     plt.close()
    #
    # if train_E:
    #     plt.title("Encoder Loss Eval")
    #     plt.plot(encoder_loss_eval)
    #     plt.savefig(out + "/loss/encoder_loss_eval_" + name + ".eps")
    #     plt.close()


def generate_figures(mode="evaluation", name="", load_weights=("", "")):
    '''
    Generates a set of figures that help to evaluate the training, it allows to see the generated ap, f0 and sp for example.
    '''

    if mode == "evaluation":
        loader = validation_loader
    elif mode == "train":
        loader = train_loader
    elif mode == "train_random":
        loader = train_random
    else:
        print("We did not understand the mode, we skip this function.")
        return

    if name is not "":
        local_out_init = out + "/" + name + "_" + mode
    else:
        local_out_init = out + "/" + mode

    if not os.path.exists(local_out_init):
        os.makedirs(local_out_init)

    if load_weights != ("", ""):
        E.load_state_dict(torch.load(load_weights[0]))
        D.load_state_dict(torch.load(load_weights[1]))

    E.eval()
    D.eval()

    if args.cuda:
        E.cuda()
        D.cuda()

    pad = 10
    # mel_basis = librosa.filters.mel(16000, 1024, n_mels=256)
    # if args.cuda:
    #     mel_basis = torch.from_numpy(mel_basis).cuda().float().unsqueeze(0)
    # else:
    #     mel_basis = torch.from_numpy(mel_basis).float().unsqueeze(0)

    spectrograms = []

    for batch_idx, data in enumerate(loader):

        batch_wav = Variable(data[0]).contiguous().float()
        batch_h = Variable(data[2]).contiguous().float()
        batch_spec = Variable(data[3]).contiguous().float()

        if args.cuda:
            batch_wav = batch_wav.cuda()
            batch_spec = batch_spec.cuda()
            batch_h = batch_h.cuda()

        # ## mel-spectrogram
        # #Not needed for AudSpec
        # batch_spec = batch_spec ** 2
        # mel_basis_ep = mel_basis.expand(batch_spec.size(0),mel_basis.size(1),mel_basis.size(2))
        # batch_spec = torch.bmm(mel_basis_ep, batch_spec)
        # batch_spec = torch.log(batch_spec)
        # batch_spec[batch_spec == float("-Inf")] = -50

        # predict parameters through waveform
        batch_h_hat = E(batch_spec)
        # batch_h_hat = transform_params(batch_h_hat)
        # print(batch_h_hat)

        ############# World ###################################################################################################
        fs = 8000
        # mel_b = librosa.filters.mel(16000, 1024, n_mels=256)
        # f0_temp = batch_h_hat[:, 0:1, :].data.cpu().numpy().astype('float64')
        # sp_temp = batch_h_hat[:, 1:514, :].data.cpu().numpy().astype('float64')
        music_temp = batch_h_hat[:, 0:params, :].data.cpu().numpy().astype('float64')
        # music_temp[:, 0, :] = (music_temp[:, 0, :] * 12) +62
        # music_temp[:, 1, :] = (music_temp[:, 1, :] * 0.3) + 0.7
        # music_temp[:, 2, :] = (music_temp[:, 2, :] * 0.5) + 0.4
        # music_temp[:, 3, :] = (music_temp[:, 3, :] * 50) + 50

        # f0_temp = batch_h[:,0:1,:].data.cpu().numpy().astype('float64')
        # sp_temp = batch_h[:,1:129,:].data.cpu().numpy().astype('float64')
        # ap_temp = batch_h[:,129:,:].data.cpu().numpy().astype('float64')
        # print(music_temp[1,:,:])
        # print(music_temp[2,:,:])
        # print(music_temp[3,:,:])

        # print(music_temp.shape)
        if not os.path.exists(local_out_init + "/waveforms"):
            os.makedirs(local_out_init + "/waveforms")

        spec_wav = np.zeros((batch_wav.shape[0], 128, spec_len)).astype('float64')
        # spec_wav = np.zeros((batch_wav.shape[0], 128, 250)).astype('float64')
        frmlen = 8
        tc = 8
        sampling_rate = 4000.0
        paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000.0)]
        # print(paras_c)
        # print(np.log2(4000/ 16000))

        for j in range(music_temp.shape[0]):
            # f0 = f0_temp[j, :, :].reshape(-1).copy(order='C')  # 406
            # sp = sp_temp[j, :, :].T.copy(order='C')  # 406, 513
            # sp = np.exp(sp)  # from log scale to original scale required by WORLD
            music = music_temp[j, :, :].copy(order='C')  # 406, 513

            # print(music.shape)

            y = music_syn.generate(music, engine, generator, parameters, rev_idx, pad=pad_render)
            y = music_syn.resample(y, 44100, fs)  # resampling
            # print(y.shape)
            y = y[pad:pad + 32000]
            # print(y.shape)
            y = y / np.sqrt(np.sum(y ** 2))  # normalize the power of waveform to 1

            plt.title("Waveform")
            plt.subplot(221)
            plt.plot(batch_wav[j].cpu().numpy())
            plt.title('Real waveform')

            plt.subplot(222)
            plt.plot(y)
            plt.title('Generated waveform')

            plt.savefig(local_out_init + "/waveforms/" + str(j) + ".eps")

            ## Save waveform

            # write(local_out_init+"/waveforms/"+str(j)+".wav", fs, np.int16(y * 32767))

            if not SERVER:
                plt.show()
            else:
                plt.close()

                # transfer wavform to power of mel-spectrogram
            # spec_tmp = librosa.core.stft(y, n_fft=1024, hop_length = 80)
            # spec_tmp = np.abs(spec_tmp) ** 2
            # spec_tmp = np.dot(mel_b, spec_tmp)
            # spec_tmp = np.log(spec_tmp)
            # spec_tmp[spec_tmp == float("-Inf")] = -50
            # spec_wav[j,:,:] = spec_tmp
            spec_wav[j, :, 0:spec_len] = (nsl.wav2aud(nsl.unitseq(y), paras_c).T)[:, 0:spec_len]
            # print((nsl.wav2aud(nsl.unitseq(y), paras_c)).shape)

        # f0 = batch_h[:, 0:1, :].data.cpu().numpy().astype('float64')
        # sp = batch_h[:, 1:514, :].data.cpu().numpy().astype('float64')
        music = batch_h[:, 0:params_ideal, :].data.cpu().numpy().astype('float64')
        # f0 = batch_h[:,0:1,:].data.cpu().numpy().astype('float64')
        # sp = batch_h[:,1:129,:].data.cpu().numpy().astype('float64')
        # ap = batch_h[:,129:,:].data.cpu().numpy().astype('float64')
        ## SAVE ALL FIGURES

        # music_temp[:, 0, :] = (music_temp[:, 0, :] * 12) +62
        # music_temp[:, 1, :] = (music_temp[:, 1, :] * 0.3) + 0.7
        # music_temp[:, 2, :] = (music_temp[:, 2, :] * 0.5) + 0.4
        # music_temp[:, 3, :] = (music_temp[:, 3, :] * 50) + 50

        # f0_random = batch_f0 + (torch.rand(f0_temp.shape[0],f0_temp.shape[1], f0_temp.shape[2]) - 0.5)*10
        # sp_random = batch_sp + (torch.rand(sp_temp.shape[0], sp_temp.shape[1], sp_temp.shape[2]) - 0.5)*5
        # ap_random = batch_ap + torch.rand(ap_temp.shape[0], ap_temp.shape[1], ap_temp.shape[2]) - 0.5

        # batch_h_random = torch.cat([f0_random, sp_random, ap_random], 1)

        local_out_init += "/latent_space/"

        # local_out_init += "/lentent_space/"
        # print (f0.shape[0])
        num = np.min((music.shape[0], 20))
        # print (num, 'num')
        # Save ap
        # Save Sp
        music_hat_saved = np.zeros((num, params, N), dtype=np.float64)
        # sp_hat_saved = np.zeros((num, 513, 251), dtype=np.float64)
        music_ideal_saved = np.zeros((num, params_ideal, N), dtype=np.float64)
        # sp_ideal_saved = np.zeros((num, 513, 251), dtype=np.float64)
        # f0_ideal_saved = np.zeros((num, 1, 251), dtype=np.float64)
        # f0_hat_saved = np.zeros((num, 1, 251), dtype=np.float64)
        # print(num)

        for i in range(num):
            music_hat_saved[i, :, :] = music_temp[i, :, :]
            music_ideal_saved[i, :, :] = music[i, :, :]
            # sp_hat_saved[i, :, :] = sp_temp[i, :, :]
            # sp_ideal_saved[i, :, :] = sp[i, :, :]
            # f0_ideal_saved[i, :, :] = f0[i, :, :]
            # f0_hat_saved[i, :, :] = f0_temp[i, :, :]

        if not os.path.exists(local_out_init):
            os.makedirs(local_out_init)

        with open(local_out_init + 'ap_hat.pkl', 'wb') as f:
            pkl.dump(music_hat_saved, f)

        with open(local_out_init + 'ap_ideal.pkl', 'wb') as f:
            pkl.dump(music_ideal_saved, f)

        # with open(local_out_init + 'sp_hat.pkl', 'wb') as f:
        #     pkl.dump(sp_hat_saved, f)
        #
        # with open(local_out_init + 'sp_ideal.pkl', 'wb') as f:
        #     pkl.dump(sp_ideal_saved, f)
        #
        # with open(local_out_init + 'f0_ideal.pkl', 'wb') as f:
        #     pkl.dump(f0_ideal_saved, f)
        #
        # with open(local_out_init + 'f0_hat.pkl', 'wb') as f:
        #     pkl.dump(f0_hat_saved, f)

        if not os.path.exists(local_out_init):
            os.makedirs(local_out_init)

        for i in range(num):

            local_out = local_out_init + str(i) + "/"

            if not os.path.exists(local_out):
                os.makedirs(local_out)

            # plt.title("F0")
            # plt.subplot(121)
            # plt.title("from encoder (E(x))")
            # plt.plot(f0_temp[i, 0, :])
            # plt.subplot(122)
            # plt.title("from world (ideal h)")
            # plt.plot(f0[i, 0, :])
            # plt.savefig(local_out + "f0.eps")

            # if not SERVER:
            #     plt.show()
            # else:
            #     plt.close()
            #
            # plt.title("Sp (in the exp domain)")
            # plt.subplot(121)
            # plt.title("from encoder (E(x))")
            # plt.imshow(((sp_temp[i, :, :])), cmap='jet')
            # plt.subplot(122)
            # plt.title("from world (ideal h)")
            # plt.imshow(((sp[i, :, :])), cmap='jet')
            # plt.colorbar()
            # plt.savefig(local_out + "sp.eps")

            # if not SERVER:
            #     plt.show()
            # else:
            #     plt.close()

            # plt.imshow((sp[i, :, :]))
            # plt.title("SP original")
            # plt.colorbar()
            # plt.savefig(local_out + "/sp_only.eps")
            #
            # if not SERVER:
            #     plt.show()
            # else:
            #     plt.close()

            params_labels = [' ', 'pitch', 'duration', 'volume', 'filter', 'resonance', 'attack', 'decay']
            #params_ideal_labels = [' ', 'pitch', 'duration', 'volume', 'filter', 'resonance', 'attack', 'decay', 'vib1', 'vib2', 'vib3']

            fig, [ax1, ax2] = plt.subplots(1,2)
            plt.title("Music Parms")
            ax1 = plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.imshow(music_temp[i, :, :], cmap=plt.cm.BuPu_r)
            ax1.set_yticklabels(params_labels)

            ax2 = plt.subplot(122)
            plt.title("from dataset (ideal h)")
            plt.imshow(music[i, :, :], cmap=plt.cm.BuPu_r)
            #ax2.set_yticklabels(params_ideal_labels)
            plt.colorbar()
            fig.tight_layout()
            plt.savefig(local_out + "/music_all.eps")

            # plt.title("Music Parms (Filter)")
            # plt.subplot(121)
            # plt.title("from encoder (E(x))")
            # plt.imshow(music_temp[i, 3, :], cmap=plt.cm.BuPu_r)
            # plt.subplot(122)
            # plt.title("from dataset (ideal h)")
            # plt.imshow(music[i, 3, :], cmap=plt.cm.BuPu_r)
            # plt.colorbar()
            # plt.savefig(local_out + "/music_filter.eps")

            # plt.title("Music Params (Volume and duration)")
            # plt.subplot(121)
            # plt.title("from encoder (E(x))")
            # plt.imshow(music_temp[i, 1:3, :], cmap=plt.cm.BuPu_r)
            # plt.subplot(122)
            # plt.title("from dataset (ideal h)")
            # plt.imshow(music[i, 1:3, :], cmap=plt.cm.BuPu_r)
            # plt.colorbar()
            # plt.savefig(local_out + "/music_vol_dur.eps")
            # # print(music_temp[i, 1:3, :])

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("Music investigation")
            ax1 = plt.subplot(211)
            plt.title("DIVA(h_hat)")
            plt.imshow(spec_wav[i, :, :], cmap='jet', aspect="auto", origin='lower')
            plt.colorbar()

            ax2 = plt.subplot(212, sharex=ax1)
            plt.title("music_hat")
            plt.imshow(music_temp[i, :, :], cmap=plt.cm.BuPu_r, aspect="auto")
            plt.colorbar()
            plt.savefig(local_out + "/music_diva_VS_music_hat.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("Music investigation")
            ax1 = plt.subplot(211)
            plt.title("D(h_hat)")
            plt.imshow(D(batch_h_hat).detach().cpu().numpy()[i, :, :], cmap='jet', aspect="auto", origin='lower')
            plt.colorbar()

            ax2 = plt.subplot(212, sharex=ax1)
            plt.title("music_hat")
            plt.imshow(music_temp[i, :, :], cmap=plt.cm.BuPu_r, aspect="auto")
            plt.colorbar()
            plt.savefig(local_out + "/music_D(h_hat)_VS_music_hat.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("Spectogram")
            plt.subplot(121)
            plt.imshow(batch_spec[i, :, :].cpu().numpy(), cmap='jet', origin='lower')
            plt.title("original")
            plt.subplot(122)
            plt.imshow(D(batch_h_hat).detach().cpu().numpy()[i, :, :], cmap='jet', origin='lower')
            plt.title("generated")
            plt.colorbar()
            plt.savefig(local_out + "/spectrogram.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

        return  # Do only the first batch


def getSpectrograms(mode="evaluation"):
    '''
    Generates spectrogram for the original sound (x), D(ideal_h), D(E(x)), world(E(x)) in order to evaluate
    the accuracy and the decoder and the encoder separatly.
    '''
    if mode == "train":
        loader = train_loader

    elif mode == "evaluation":
        loader = validation_loader

    elif mode == "train_random":
        loader = train_random
    else:
        print("The mode is not understood, therefore we don't compute spectrograms.")
        return [([], [], [], [])]

    E.eval()
    D.eval()

    if args.cuda:
        E.cuda()
        D.cuda()

    pad = 10

    frmlen = 8
    tc = 8
    sampling_rate = 8000.0
    paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000.0)]

    # mel_basis = librosa.filters.mel(16000, 1024, n_mels=256)
    # if args.cuda:
    #     mel_basis = torch.from_numpy(mel_basis).cuda().float().unsqueeze(0)
    # else:
    #     mel_basis = torch.from_numpy(mel_basis).float().unsqueeze(0)

    spectrograms = []

    for batch_idx, data in enumerate(loader):

        batch_wav = Variable(data[0]).contiguous().float()
        batch_h = Variable(data[2]).contiguous().float()
        batch_spec = Variable(data[3]).contiguous().float()

        if args.cuda:
            batch_wav = batch_wav.cuda()
            batch_spec = batch_spec.cuda()
            batch_h = batch_h.cuda()

        # ## mel-spectrogram
        # #If we are using Aud Spec, we do no need to use this blocl of code and let batch_spec be the way it is
        # batch_spec = batch_spec ** 2
        # mel_basis_ep = mel_basis.expand(batch_spec.size(0),mel_basis.size(1),mel_basis.size(2))
        # batch_spec = torch.bmm(mel_basis_ep, batch_spec)
        # batch_spec = torch.log(batch_spec)
        # batch_spec[batch_spec == float("-Inf")] = -50

        # predict parameters through waveform
        batch_h_hat = E(batch_spec)
        # batch_h_hat = transform_params(batch_h_hat)
        # print(batch_music)
        # batch_h_hat = batch_music #torch.cat([batch_f0, batch_sp, batch_music], 1)

        ############# World ###################################################################################################
        fs = 8000
        # mel_b = librosa.filters.mel(4000, 1024, n_mels=256)
        # f0_temp = batch_h_hat[:, 0:1, :].data.cpu().numpy().astype('float64')
        # sp_temp = batch_h_hat[:, 1:514, :].data.cpu().numpy().astype('float64')
        music_temp = batch_h_hat[:, 0:params, :].data.cpu().numpy().astype('float64')
        # music_temp[:, 0, :] = (music_temp[:, 0, :] * 12) +62
        # music_temp[:, 1, :] = (music_temp[:, 1, :] * 0.3) + 0.7
        # music_temp[:, 2, :] = (music_temp[:, 2, :] * 0.5) + 0.4
        # music_temp[:, 3, :] = (music_temp[:, 3, :] * 50) + 50

        # f0_temp = batch_h_hat[:,0:1,:].data.cpu().numpy().astype('float64')
        # sp_temp = batch_h_hat[:,1:129,:].data.cpu().numpy().astype('float64')
        # ap_temp = batch_h_hat[:,129:,:].data.cpu().numpy().astype('float64')

        spec_wav = np.zeros((batch_wav.shape[0], 128, spec_len)).astype('float64')
        # spec_wav = np.zeros((batch_wav.shape[0], 128, 250)).astype('float64')
        # print("music temp shape")
        # print(music_temp.shape)

        for j in range(music_temp.shape[0]):
            # f0 = f0_temp[j, :, :].reshape(-1).copy(order='C')  # 406
            # sp = sp_temp[j, :, :].T.copy(order='C')  # 406, 513
            # sp = np.exp(sp)  # from log scale to original scale required by WORLD
            music = music_temp[j, :, :].copy(order='C')  # 406, 513
            # y = pw.synthesize(f0, sp, ap, fs, frame_period=8.0)
            # print(music.mean(axis=0))
            # print(music)
            # print(music.shape)

            y = music_syn.generate(music, engine, generator, parameters, rev_idx, pad=pad_render)
            y = music_syn.resample(y, 44100, fs)  # resampling
            # y = pw.synthesize(f0, sp, music, fs, frame_period=8.0)
            y = y[pad:pad + 32000]
            y = y / np.sqrt(np.sum(y ** 2))  # normalize the power of waveform to 1

            # music_syn.write_to_wav(y,fs,j)

            # transfer wavform to power of mel-spectrogram
            # spec_tmp = nsl.wav2aud(nsl.unitseq(y), paras_c)
            # spec_tmp = spec_tmp.T

            # #The below block may not be necessary if using Aud Specs
            # spec_tmp = librosa.core.stft(y, n_fft=1024, hop_length = 80)
            # spec_tmp = np.abs(spec_tmp) ** 2
            # spec_tmp = np.dot(mel_b, spec_tmp)
            # spec_tmp = np.log(spec_tmp)
            # spec_tmp[spec_tmp == float("-Inf")] = -50
            # spec_wav[j, :, :] = np.sqrt(spec_tmp)[:,0:250]
            spec_wav[j, :, 0:spec_len] = (((nsl.wav2aud(nsl.unitseq(y), paras_c)).T))[:, 0:spec_len]

        # print("out side initialization for loop")

        # if not os.path.exists(out + "/spectrograms"):
        #        os.makedirs(out + "/spectrograms")

        # plot DIVA spectrogram separately
        # plt.imshow(spec_wav[1], cmap='jet', origin='lower', aspect="auto")
        # plt.title('Spectrogram from DIVA (from encoder h)')  # World(E(H))
        # plt.savefig(out + "/spectrograms/" +"diva_spectrogram.eps")

        # f0 = batch_h[:, 0:1, :].data.cpu().numpy().astype('float64')
        # sp = batch_h[:, 1:514, :].data.cpu().numpy().astype('float64')
        # music = batch_h[:, params:, :].data.cpu().numpy().astype('float64')
        music = batch_h[:, 0:params, :].data.cpu().numpy().astype('float64')
        # Rescaling for plotting
        # batch_h[:, 0, :] = (batch_h[:, 0, :] - 62)/12
        # batch_h[:, 1, :] = (batch_h[:, 1, :] - 0.7)/0.3
        # batch_h[:, 2, :] = (batch_h[:, 2, :] - 0.4)/0.5
        # batch_h[:, 3, :] = (batch_h[:, 3, :] - 50)/50

        #####################################################################################################################

        realSpectrogram = np.array(batch_spec.detach().cpu().numpy())
        # decoderSpectrogram = np.array(D(batch_h).detach().cpu().numpy())
        modelSpectrogram = np.array(D(batch_h_hat).detach().cpu().numpy())
        worldSpectrogram = np.array(spec_wav)

        spectrograms.append((realSpectrogram, modelSpectrogram, worldSpectrogram))
        # In order to only save a few spectrograms (64)
        return spectrograms

    return spectrograms


def plotSpectrograms(spectrograms, name="", MAX_examples=20):
    '''
    Plot or save the spectrograms.
    '''

    if not os.path.exists(out + "/spectrograms/" + name):
        os.makedirs(out + "/spectrograms/" + name)

    if name == 'TRAIN_evaluation_data':
        realSpectrogram, modelSpectrogram, worldSpectrogram = spectrograms[0]
        # # with open('Pitch_1.pkl', 'wb') as f:
        # #     pkl.dump(X_pitch, f)
        for i in range(min(len(spectrograms[0][0]), MAX_examples)):
            with open(out + "/spectrograms/" + "RealSpectrogram%d.pkl" % (i), 'wb') as f:
                pkl.dump(realSpectrogram[i], f)

            # with open(out + "/spectrograms/" + "decoderSpectrogram%d.pkl" % (i), 'wb') as f:
            #     pkl.dump(decoderSpectrogram[i], f)

            with open(out + "/spectrograms/" + "modelSpectrogram%d.pkl" % (i), 'wb') as f:
                pkl.dump(modelSpectrogram[i], f)

            with open(out + "/spectrograms/" + "worldSpectrogram%d.pkl" % (i), 'wb') as f:
                pkl.dump(worldSpectrogram[i], f)

    for i in range(min(len(spectrograms[0][0]), MAX_examples)):

        realSpectrogram, modelSpectrogram, worldSpectrogram = spectrograms[0]

        # plt.subplot(221)
        # plt.imshow(realSpectrogram[i], cmap='jet', origin='lower')  # Ground Truth
        # plt.title('Real Spectrogram contrasted')
        #
        # plt.subplot(222)
        # plt.imshow(decoderSpectrogram[i], cmap='jet', origin='lower')
        # plt.title('Decoder Spectrogram (using ideal h)')  # D(ideal_h)
        #
        # plt.subplot(223)
        # plt.imshow(modelSpectrogram[i], cmap='jet', origin='lower')
        # if name == 'TRAIN_evaluation_data':
        #     np.save("spectrogram_{}".format(i), modelSpectrogram[i])
        # plt.title('Model Spectrogram: D(E(x))')  # (D(E(X)))
        #
        # plt.subplot(224)
        # plt.imshow(worldSpectrogram[i], cmap='jet', origin='lower')
        # plt.title('Spectrogram from DIVA (from encoder h)')  # World(E(H))
        #
        # plt.savefig(out + "/spectrograms/" + name + "/" + str(i) + ".eps")

        ## Plot spectrograms separately

        plt.imshow(realSpectrogram[i], cmap='jet', origin='lower')  # Ground Truth
        # plt.title('Real Spectrogram')
        plt.savefig(out + "/spectrograms/" + name + "/original_spec_" + str(i) + ".eps")

        #plt.subplot(222)
        #plt.imshow(decoderSpectrogram[i], cmap='jet', origin='lower')
        #plt.title('Decoder Spectrogram (using ideal h)')  # D(ideal_h)

        #ax = plt.subplot(gs[1, 0])
        plt.imshow(modelSpectrogram[i], cmap='jet', origin='lower')
        if name == 'TRAIN_evaluation_data':
            np.save("spectrogram_{}".format(i),modelSpectrogram[i])
        # plt.title('Model Spectrogram: D(E(x))')  # (D(E(X)))
        plt.savefig(out + "/spectrograms/" + name + "/model_spec_" + str(i) + ".eps")

        #ax = plt.subplot(gs[1, 1])
        plt.imshow(worldSpectrogram[i], cmap='jet', origin='lower')
        # plt.title('Spectrogram from DIVA (from encoder h)')  # World(E(H))
        plt.savefig(out + "/spectrograms/" + name + "/DIVA_spec_" + str(i) + ".eps")


        # plot DIVA spectrogram separately
        # plt.imshow(worldSpectrogram[i], cmap='jet', origin='lower', aspect="auto")
        # plt.title('Spectrogram from DIVA (from encoder h)')  # World(E(H))
        # plt.savefig(out + "/spectrograms/" + name +"/diva_spectrogram_" + str(i) + ".eps")

        if not SERVER:
            plt.show()
        else:
            plt.close()


# prepare exp folder
exp_name = 'MirrorNet'
descripiton = 'train MirrorNet without ideal hiddens'
exp_new = 'tmp/'
base_dir = './'
# exp_prepare='/hdd/cong/exp_prepare_folder.sh'
# net_dir=base_dir + exp_new + '/' + exp_name+'/net'
# subprocess.call(exp_prepare+ ' ' + exp_name + ' ' + base_dir + ' ' + exp_new, shell=True)

### Set the name of the folder
out = "figs/NEW_with_only_1_loss_for_DIVA_melodies_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour)
out_dir = "./figs/NEW_with_only_1_loss_for_DIVA_melodies_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour) + "/"

model_dir = "./figs/NEW_with_only_1_loss_for_E_DATE_2021-05-13_H_12/"

# Create sub directory is dont exist
if not os.path.exists(out_dir + exp_new):
    os.makedirs(out_dir + exp_new)

# Create weights directory if don't exist
if not os.path.exists(out_dir + exp_new + 'net/'):
    os.makedirs(out_dir + exp_new + 'net/')


if not os.path.exists(base_dir + exp_new):
    os.makedirs(base_dir + exp_new)

# Create log directory if don't exist
if not os.path.exists(base_dir + exp_new + "log/"):
    os.makedirs(base_dir + exp_new + "log/")


# setting logger   NOTICE! it will overwrite current log
log_dir = base_dir + exp_new + "log/" + str(date.today()) + ".log"
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    filename=log_dir,
                    filemode='a')

# global params

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--decoder_lr', type=float, default=1e-3,  # changed from 3e-4
                    help='learning rate')
parser.add_argument('--encoder_lr', type=float, default=1e-2,  # changed from 3e-4
                    help='learning rate')
parser.add_argument('--seed', type=int, default=20190101,
                    help='random seed')
parser.add_argument('--val-save', type=str, default=base_dir + exp_new + '/' + exp_name + '/net/cv/model_weight.pt',
                    help='path to save the best model')

parser.add_argument('--train_data', type=str,
                    default=base_dir + "music_data_6params_new/train_audio_10params_v1_new.data",
                    help='path to training data')

parser.add_argument('--test_data', type=str,
                    default=base_dir + "music_data_6params_new/test_audio_10params_v1_new.data",
                    help='path to testing data')

parser.add_argument('--initialization_data', type=str,
                    default=base_dir + "music_data_6params_new/initialize_audio.data",
                    help='path to initialization data')

parser.add_argument('--train_random_data', type=str,
                    default=base_dir + "music_data_6params_new/train_audio_10notes_random_10notes.data",
                    help='path to train random data')

args, _ = parser.parse_known_args()
print(type(args))
print(args)
np.save(base_dir + exp_new + '/model_arch', args)
args.cuda = args.cuda and torch.cuda.is_available()

np.random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA ACTIVATED")
else:
    kwargs = {}
    print("CUDA DISABLED")

training_data_path = args.train_data
validation_data_path = args.test_data
initialization_data_path = args.initialization_data
train_random_data_path = args.train_random_data

# initialization_loader = DataLoader(BoDataset(initialization_data_path),
#                           batch_size=args.batch_size,
#                           shuffle=False,
#                           **kwargs)

train_loader = DataLoader(BoDataset(training_data_path),
                          batch_size=args.batch_size,
                          shuffle=False,
                          **kwargs)
# print ('----------------------', list(train_loader))
# print ('.,.,.,.,.,.,.,.,.,.', (list(train_loader)[0]))

train_random = DataLoader(BoDataset(train_random_data_path),
                          batch_size=args.batch_size,
                          shuffle=False,
                          **kwargs)

validation_loader = DataLoader(BoDataset(validation_data_path),
                               batch_size=args.batch_size,
                               shuffle=False,
                               **kwargs)

eval_byPieces_loader = DataLoader(BoDataset(validation_data_path),
                                  batch_size=1,
                                  shuffle=False,
                                  **kwargs)

args.dataset_len = len(train_loader)
args.log_step = args.dataset_len // 2

E = encoder()
D = decoder()
# print ('. . . . . . . . . . ', list(E.parameters()))

if args.cuda:
    E.cuda()
    D.cuda()

# E.load_state_dict(torch.load('tmp/net/Gui_model_weight_E1.pt'))
# D.load_state_dict(torch.load('tmp/net/Gui_model_weight_D1.pt'))


current_lr = args.lr
E_optimizer = optim.Adam(E.parameters(), lr=args.encoder_lr)
E_scheduler = torch.optim.lr_scheduler.ExponentialLR(E_optimizer, gamma=0.2)  # changed gamma=0.5 to 0.3
# E_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(E_optimizer, mode='min', factor=0.5, patience=20, verbose=True, threshold=1e-7)

D_optimizer = optim.Adam(D.parameters(), lr=args.decoder_lr)
D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.2)  # changed gamma=0.5 to 0.1
# D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, mode='min', factor=0.2, patience=20, verbose=True, threshold=1e-7)

parameters = [p for p in E.parameters()] + [p for p in D.parameters()]
all_optimizer = optim.Adam(parameters, lr=args.lr)
all_scheduler = torch.optim.lr_scheduler.ExponentialLR(all_optimizer, gamma=0.5)  # changed gamma=0.5 to 0.2

# output info

log_print('Experiment start time is {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
log_print('-' * 70)
log_print('Discription:'.format(descripiton))
log_print('-' * 70)
log_print(args)

log_print('-' * 70)
s = 0
for param in E.parameters():
    s += np.product(param.size())
for param in D.parameters():
    s += np.product(param.size())

log_print('# of parameters: ' + str(s))

log_print('-' * 70)
log_print('Training Set is {}'.format(training_data_path))
log_print('CV set is {}'.format(validation_data_path))
log_print('-' * 70)

### Set the name of the folder
out = "figs/NEW_with_only_1_loss_for_DIVA_melodies_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour)

if not os.path.exists(out):
    os.makedirs(out)

if not os.path.exists(out + "/loss"):
    os.makedirs(out + "/loss")

print(out)

# Create music synthesier
engine, generator, parameters, rev_idx = music_syn.create_synth('toy')

# generated the spectrograms before anything to make sure there is no issue with already trained weights or something.
spec = getSpectrograms("train")
plotSpectrograms(spec, "before_training_train_data")

# '''
#         INITIALIZATION
# '''
# print("INITIALIZATION")
#
# # trainTogether_newTechnique(epochs=10, init=True, train_D=True, train_E=True, loader_eval="train", save_better_model=True)
# trainTogether_newTechnique(epochs=50, init=True, train_D=True, train_E=True, loader_eval="train",
#                            save_better_model=True)
#
# print("************************************************************************************************************")
# print("Initialization done")
# print("************************************************************************************************************")
#
# generate_figures("train_random", name="init")  ## evaluation or train
# generate_figures("train", name="init")  ## evaluation or train
#
# spec = getSpectrograms("train_random")
# plotSpectrograms(spec, "INIT_train_random_data")
#
# # spec = getSpectrograms("train")
# # plotSpectrograms(spec, "INIT_train_data")
#
# generate_figures("evaluation", name="init")  ## evaluation or train
# spec = getSpectrograms("evaluation")
# plotSpectrograms(spec, "INIT_evaluation_data")

'''
        TRAINING
'''
print("TRAINING")
for i in range(15):
    torch.cuda.empty_cache()
    print("ITERATION", str(i + 1))
    trainTogether_newTechnique(epochs=18, name="train_D_" + str(i + 1), init=False, train_D=True, train_E=False,
                               loader_eval="train", save_better_model=False)  # 20 to 5
    trainTogether_newTechnique(epochs=25, name="train_E_" + str(i + 1), init=False, train_D=False, train_E=True,
                               loader_eval="train", save_better_model=False)
    # trainTogether_newTechnique(epochs=10, name="train_D_" + str(i + 1), init=False, train_D=True, train_E=False,
    #                          loader_eval="train", save_better_model=False)  # 20 to 5

    if i % 10 == 0:
        generate_figures("train", name="still_training_train" + str(i))
        generate_figures("evaluation", name="still_training_eval" + str(i))

print("************************************************************************************************************")
print("Training done")
print("************************************************************************************************************")

generate_figures("train", name="end")  ## evaluation or train
generate_figures("evaluation", name="end")  ## evaluation or train

spec = getSpectrograms("train")
plotSpectrograms(spec, "TRAIN_train_data")

spec = getSpectrograms("evaluation")
plotSpectrograms(spec, "TRAIN_evaluation_data")

# save model with checkpoint
with open(out_dir + exp_new + '/net/music_model_weight_E1.pt', 'wb') as f:
    torch.save({'model_state_dict': E.cpu().state_dict(), 'optimizer_state_dict': E_optimizer.state_dict()}, f)
    print("We saved encoder!")

with open(out_dir + exp_new + '/net/music_model_weight_D1.pt', 'wb') as f:
    torch.save({'model_state_dict': D.cpu().state_dict(), 'optimizer_state_dict': D_optimizer.state_dict()}, f)
    print("We saved decoder!")

# '''
#         POST_TRAINING
# '''
#
# print("POST_TRAINING")
# trainTogether_newTechnique(epochs=10, init=True, train_D=True, train_E=False, loader_eval="train",
#                            save_better_model=True)  # 100 to 5 to 1
#
# print("************************************************************************************************************")
# print("post training done")
# print("************************************************************************************************************")
#
# spec = getSpectrograms("train")
# plotSpectrograms(spec, "POST_TRAIN_train_data")
#
# spec = getSpectrograms("evaluation")
# plotSpectrograms(spec, "POST_TRAIN_evaluation_data")

print("Done.")
