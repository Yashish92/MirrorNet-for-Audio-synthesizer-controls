#!/usr/bin/env python
# coding: utf-8

import matplotlib

SERVER = True  # This variable enable or disable matplotlib, set it to true when you use the server!
if SERVER:
    matplotlib.use('Agg')

import numpy as np
from scipy import signal
import os
import string
import librosa
import time
import random
import h5py
# import pyworld as pw
from tqdm import tqdm
import sys
# import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import scipy.io.wavfile
from random_generation import get_f0, get_ap, get_random_h
import nsltools as nsl
import music_synthesize_new as music_syn
import json
import re

np.set_printoptions(threshold=sys.maxsize)

def generate_data(path_audio, path_params, audio_time=2, sampling_rate=8000, random=False):   # changed audio time to 1 from 2
    # path='./data/rahil_trial_data/'
    # audio_time = 2
    # sampling_rate = 22050
    # random=True
    # data_type = 'train' or 'cv'
    # data path
    no_of_params = 7  # 16 to 128
    N = 5  # number of parameter samples across time
    spec_len = 250
    clean_path = path_audio  # read Botinhao data from this path
    directory = path_audio[:-1] + "_new.data"  # save data to this path
    if random:
        directory = path_audio[:-1] + "_random" + "_new.data"  # save data to this path

    audio_len = np.int(audio_time * sampling_rate)
    cur_sample = 0  # current sample generated
    no_files = 0

    # collect raw waveform and trim them into equal length, here is 32000 (16k * 2s)
    spk_wav_tmp = np.zeros((100000, audio_len))
    for (dirpath, dirnames, filenames) in tqdm(os.walk(clean_path)):
        for files in tqdm(filenames):
            if '.wav' in files:
                file_no = int(re.split('_|\.',files)[2])
                s_wav = clean_path + files
                s_wav, s_sr = librosa.load(s_wav, sr=sampling_rate)
                #print(file_no)

                #chunk_num = len(s_wav) // audio_len
                #extra_len = len(s_wav) % audio_len
                #if extra_len > audio_len // 2:
                #    trim_len = (chunk_num + 1) * audio_len
                #else:
                #    trim_len = chunk_num * audio_len

                #if len(s_wav) < trim_len:
                #   if chunk_num == 0:
                #        spk_wav = np.concatenate([s_wav, np.zeros(trim_len - len(s_wav))])
                #    else:
                #        spk_wav = np.concatenate([s_wav[:chunk_num * audio_len], s_wav[-audio_len:]])

                #elif len(s_wav) > trim_len:
                #   spk_wav = s_wav[:trim_len]

                spk_wav = s_wav[:audio_len]
                #print(spk_wav.shape)

                spk_wav = np.array(spk_wav).reshape(-1, audio_len)
                #print(spk_wav.shape)
                spk_wav_tmp[file_no] = spk_wav
                #cur_sample = cur_sample + spk_wav.shape[0]
                no_files = no_files + 1


    spk_wav_tmp = spk_wav_tmp[:no_files, :]
    # spk_wav_tmp.shape= (Num of examples (N), 32000)
    print(spk_wav_tmp.shape) 
    #print(spk_wav_tmp[3])
    print('trim finished')  # spk_wav_tmp.shape[0]=Number of files in data directory or Num of examples= N

    param_array = np.zeros(7)  # for music synthesizer preset
    spk_tmp = np.zeros((spk_wav_tmp.shape[0], spk_wav_tmp.shape[1]))  # raw speech with normalized power
    h_tmp = np.zeros((spk_tmp.shape[0], no_of_params, N))  # ideal hiddens from world
    spec_tmp_513 = np.zeros(
        (spk_wav_tmp.shape[0], 128, spec_len))  # dimensions of the AudSpec - needs to be softcoded for scalability
    spec_tmp_513_pw = np.zeros((spk_wav_tmp.shape[0], 128,
                                spec_len))  # dimensions of the Reconstructed AudSpec - needs to be softcoded for scalability
    print(spec_tmp_513_pw.shape, 'spec_tmp_513_pw')

    # create an array with parameters audio files
    i = 0
    for (dirpath, dirnames, filenames) in tqdm(os.walk(path_params)):
        for files in tqdm(filenames):
            if '.npy' in files:
                if random:
                        file_no = int(re.split('_|\.',files)[3])
                else:
                        file_no = int(re.split('_|\.',files)[2])
                # print(dirnames)
                param_file = path_params + files
                # print(param_file)
                loaded_params = np.load(param_file, allow_pickle=True)
                # loaded_params, loaded_chars, loaded_audio = loaded["param"], loaded["chars"], loaded["audio"]
                # print(loaded_params)
                # param_array = gen_json_list(loaded_params)
                # print(param_array)
                # param_array = np.reshape(param_array, (6, 1))  # 16 to 128
                # extended_arr = np.tile(param_array, (1, 126))
                # print(extended_arr.shape)
                # print(extended_arr)
                h_tmp[file_no, :, :] = loaded_params.copy(order='C')
                # h_tmp[i, :, :] = extended_arr   # 16 to 128
                i += 1

    #print(h_tmp[2, :, :])
    '''Parameters for AudSpectrogram'''
    frmlen = 8
    tc = 8
    paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000.0)]
    # print(sampling_rate)

    pad = 40
    for i in tqdm(range(spk_wav_tmp.shape[0])):
        # print("No of data: ", spk_wav_tmp.shape[0])
        if i % 100 == 0:
            print(i)
        wav = spk_wav_tmp[i, :].copy().astype('float64')
        wav = wav.reshape(-1)
        # wav=nsl.unitseq(wav)    #THis here causes the problem: RuntimeWarning: overflow encountered in exp

        wav = wav / np.sqrt(np.sum(wav ** 2))  # power normalization

        wav = nsl.unitseq(wav)  # THis here causes the problem: RuntimeWarning: overflow encountered in exp

        # #wav.shape=(32000,)
        # if not random:
        spk_tmp[i, :] = wav  # this is saved

        #spec513 = np.sqrt(nsl.wav2aud(wav, paras_c))  # audSpec
        spec513 = nsl.wav2aud(wav, paras_c)  # audSpec
        # print (spec513.shape, 'spec513--line 116')
        spec_tmp_513[i, :, 0:spec_len] = spec513.T  # AudSpec
        # print (spec_tmp_513[i,:,:])

    dset = h5py.File(directory, 'w')
    print(spk_tmp.shape)
    spk_set = dset.create_dataset('speaker', shape=(spk_tmp.shape[0], spk_tmp.shape[1]), dtype=np.float64)
    hid_set = dset.create_dataset('hidden', shape=(spk_tmp.shape[0], no_of_params, N), dtype=np.float64)
    # spec513_set = dset.create_dataset('spec513', shape=(spk_tmp.shape[0], 513, 401), dtype=np.float64)
    # spec_513_pw_set = dset.create_dataset('spec513_pw', shape=(spk_tmp.shape[0], 513, 401), dtype=np.float64)

    spec513_set = dset.create_dataset('spec513', shape=(spk_tmp.shape[0], 128, spec_len), dtype=np.float64)
    spec_513_pw_set = dset.create_dataset('spec513_pw', shape=(spk_tmp.shape[0], 128, spec_len), dtype=np.float64)

    spk_set[:, :] = spk_tmp

    hid_set[:, :, :] = h_tmp

    # spec513_set[:,:,:] = d
    spec513_set = []
    spec_513_pw_set[:, :, :] = spec_tmp_513

    dset.close()
    print('finished')


if __name__ == "__main__":
    if (len(sys.argv) == 3 or len(sys.argv) == 4) and sys.argv[1] != "-h":
        if len(sys.argv) == 3:
            generate_data(sys.argv[1], sys.argv[2])
        elif sys.argv[3] == "random":
            generate_data(sys.argv[1], sys.argv[2], random=True)
        else:
            print("We did not understand the second argument.")
    else:
        print("USAGE: python3", sys.argv[0], "<path to the wav data>")

# if __name__=="__main__":
#     generate_data('./data/rahil_trial_data/')
