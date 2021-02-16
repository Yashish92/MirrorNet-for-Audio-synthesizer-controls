#!/usr/bin/env python
# coding: utf-8
'''
Date: March 19
This performs the data_process script to produce audspecs
The spectrogram and the latent variables must have matching dimensions. 
The hidden space stores the log(sp). These have to be exponentiated to put into WORLD
Spec=sqrt(spec) for better freq resolution 
Aim:
Spectrogram: 128-251
latent space ap: 513-251
latent space sp: 64-251
latent space (f0): 1-251


'''
import matplotlib

SERVER = True # This variable enable or disable matplotlib, set it to true when you use the server!
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
import pyworld as pw
from tqdm import tqdm
import sys
# import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import scipy.io.wavfile
from random_generation import get_f0, get_ap, get_random_h
import nsltools as nsl 

np.set_printoptions(threshold=sys.maxsize)

def generate_data(path, audio_time = 2, sampling_rate = 16000, random=False):
# path='./data/rahil_trial_data/'
# audio_time = 2
# sampling_rate = 16000
# random=True
# data_type = 'train' or 'cv'
# data path
    clean_path = path  # read Botinhao data from this path
    directory = path[:-1]+".data"  #save data to this path
    if random:
        directory = path[:-1] + "_random" + ".data"  #save data to this path
        
    audio_len = np.int(audio_time * sampling_rate) 
    cur_sample = 0 #current sample generated

    # collect raw waveform and trim them into equal length, here is 32000 (16k * 2s)
    spk_wav_tmp = np.zeros((100000, audio_len))
    for (dirpath, dirnames, filenames) in tqdm(os.walk(clean_path)):
        for files in tqdm(filenames):
            if '.wav' in files and '._' not in files:      
                s_wav = clean_path + files
                s_wav, s_sr = librosa.load(s_wav, sr = sampling_rate)

                chunk_num = len(s_wav) // audio_len
                extra_len = len(s_wav) % audio_len
                if extra_len > audio_len//2:
                    trim_len = (chunk_num + 1) * audio_len
                else:
                    trim_len = chunk_num * audio_len

                if len(s_wav) < trim_len:
                    if chunk_num == 0:
                        spk_wav = np.concatenate([s_wav, np.zeros(trim_len-len(s_wav))])
                    else:
                        spk_wav = np.concatenate([s_wav[:chunk_num * audio_len], s_wav[-audio_len:]])

                elif len(s_wav) > trim_len:
                    spk_wav = s_wav[:trim_len]

                spk_wav = np.array(spk_wav).reshape(-1, audio_len)
                spk_wav_tmp[cur_sample:cur_sample+spk_wav.shape[0]] = spk_wav
                cur_sample = cur_sample + spk_wav.shape[0]
                
    spk_wav_tmp = spk_wav_tmp[:cur_sample, :]
    #spk_wav_tmp.shape= (Num of examples (N), 32000)

    print('trim finished')              #spk_wav_tmp.shape[0]=Number of files in data directory or Num of examples= N

    spk_tmp = np.zeros((spk_wav_tmp.shape[0], spk_wav_tmp.shape[1]))   # raw speech with normalized power
    h_tmp = np.zeros((spk_tmp.shape[0], 1+513*2, 251))                 # ideal hiddens from world
    spec_tmp_513 = np.zeros((spk_wav_tmp.shape[0], 128, 251)) #dimensions of the AudSpec - needs to be softcoded for scalability 
    spec_tmp_513_pw = np.zeros((spk_wav_tmp.shape[0], 128, 251)) #dimensions of the Reconstructed AudSpec - needs to be softcoded for scalability 
    print(spec_tmp_513_pw.shape, 'spec_tmp_513_pw')

    '''Parameters for AudSpectrogram'''
    frmlen = 8
    tc = 8
    paras_c = [frmlen, tc, -2, np.log2(sampling_rate/16000)]
    print (sampling_rate)


    pad = 40
    for i in tqdm(range(spk_wav_tmp.shape[0])):
        if i%100 == 0:
            print(i)
        wav = spk_wav_tmp[i, :].copy().astype('float64')
        wav = wav.reshape(-1)
        # wav=nsl.unitseq(wav)    #THis here causes the problem: RuntimeWarning: overflow encountered in exp

        wav = wav/np.sqrt(np.sum(wav**2))  # power normalization

        wav=nsl.unitseq(wav)    #THis here causes the problem: RuntimeWarning: overflow encountered in exp

        # #wav.shape=(32000,)
        if not random:
            spk_tmp[i, :] = wav #this is saved

        spec513=nsl.wav2aud(wav,paras_c) #audSpec
        # print (spec513.shape, 'spec513--line 116')
        spec_tmp_513[i,:,0:250]=spec513.T # AudSpec
        # print (spec_tmp_513[i,:,:])

        # WORLD to extract ideal hiddens
        f0, sp, ap = pw.wav2world(wav, sampling_rate, frame_period=8.0)    
        #f0 shape: (401,)
        #ap.shape: (401, 513)
        #sp.shape: (401, 513)
        sp=np.log(sp)
        # print ('sp:',sp)
        # print ('mean(sp):', np.mean(sp))
        # print ('max(sp)', np.max(sp))
        # print ('min(sp)', np.min(sp))

            
        if random:
            f0_ideal = f0
            ap_ideal = ap
            #f0, ap = get_f0(f0.shape), get_ap(ap.shape)
            # print (f0.shape, 'f0.shape')
            # print (ap.shape, 'ap_shape')
            f0, ap = get_random_h(f0.shape, ap.shape)
            f0 = f0.copy(order='C')
            ap = ap.copy(order='C')
            # sp=np.flipud(sp)
            # sp = sp.copy(order='C')

        # h_tmp[i,0,:] = f0.copy(order='C')
        # h_tmp[i,1:258, :] = sp.T.copy(order='C')
        # h_tmp[i,258:,:] = ap.T.copy(order='C')
        h_tmp[i,0,:] = f0.copy(order='C')   #H_tmp is saved!
        h_tmp[i,1:514, :] = sp.T.copy(order='C')    #we're saving the log(sp)
        h_tmp[i,514:,:] = ap.T.copy(order='C')

        # reconstructed speech
        sp = np.exp(sp) # To hava the same scale as D(x)
        # sp = np.log(sp) # To hava the same scale as D(x)

        y = pw.synthesize(f0, sp, ap, sampling_rate, frame_period=8.0)
        # print (len(y), 'len(y)')
        # y = y[pad:pad+32000/2]
        y = y[pad:pad+32000]

        y = y/np.sqrt(np.sum(y**2))       # power normalization
        y=nsl.unitseq(y)

        if random:
            # reconstructed speech
            y_ideal = pw.synthesize(f0_ideal, sp, ap_ideal, sampling_rate, frame_period=8.0)
            y_ideal = y_ideal[pad:pad+32000]
            # y_ideal = y_ideal[pad:pad+32000]

            y_ideal = y_ideal/np.sqrt(np.sum(y**2))       # power normalization
            # spec_ideal = np.abs(librosa.core.stft(y_ideal, n_fft=1024, hop_length = 80))
            spec_ideal = nsl.wav2aud(nsl.unitseq(y_ideal), paras_c)
            #scipy.io.wavfile.write("test.wav", sampling_rate, y)

        if random:
            spk_tmp[i,:] = y
        # y=nsl.unitseq(y)


        spec513_pw=nsl.wav2aud(y, paras_c)
        spec513_pw=np.sqrt(spec513_pw.T)
        # print (spec513_pw)
        spec_tmp_513_pw[i,:,0:250] = spec513_pw
        # print (spec_tmp_513_pw)

        if random:
            spec_tmp_513[i,:,:] = spec_tmp_513_pw[i,:,:]


    #         # plt.subplot(221)
    #         # plt.title("RANDOM")
    #         # plt.plot(f0)
    #         # plt.subplot(222)
    #         # plt.imshow(sp)
    #         # plt.subplot(223)
    #         # plt.imshow(ap)
    #         # plt.subplot(224)
    #         # plt.plot(y)
    #         # plt.show()

    #         # plt.subplot(221)
    #         # plt.title("IDEAL")
    #         # plt.plot(f0_ideal)
    #         # plt.subplot(222)
    #         # plt.imshow(sp)
    #         # plt.subplot(223)
    #         # plt.imshow(ap_ideal)
    #         # plt.subplot(224)
    #         # plt.plot(y_ideal)
    #         # plt.show()

    #         # plt.subplot(121)
    #         # plt.title("Original")
    #         # plt.imshow(spec_ideal)
    #         # plt.colorbar()
    #         # plt.subplot(122)
    #         # plt.title("Generated")
    #         # plt.imshow(spec_tmp_513_pw[i,:,:])
    #         # plt.colorbar()
    #         # plt.show()

    #         # plt.imshow(spec_tmp_513_pw[i,:,:])
    #         # plt.title("GENERATED SPECTROGRAM")
    #         # plt.show()

        # write data
    # print (spec_tmp_513)
    dset = h5py.File(directory, 'w')
    print (spk_tmp.shape)
    spk_set = dset.create_dataset('speaker', shape=(spk_tmp.shape[0], spk_tmp.shape[1]), dtype=np.float64)
    hid_set = dset.create_dataset('hidden', shape=(spk_tmp.shape[0], 1+513*2, 251), dtype=np.float64)
    # spec513_set = dset.create_dataset('spec513', shape=(spk_tmp.shape[0], 513, 401), dtype=np.float64)
    # spec_513_pw_set = dset.create_dataset('spec513_pw', shape=(spk_tmp.shape[0], 513, 401), dtype=np.float64)

    spec513_set = dset.create_dataset('spec513', shape=(spk_tmp.shape[0], 128, 251), dtype=np.float64)
    spec_513_pw_set = dset.create_dataset('spec513_pw', shape=(spk_tmp.shape[0], 128, 251), dtype=np.float64)


    spk_set[:,:] = spk_tmp
    hid_set[:,:,:] = h_tmp
    # spec513_set[:,:,:] = d
    spec513_set = []
    spec_513_pw_set[:,:,:] = spec_tmp_513_pw

    dset.close()
    print('finished')

if __name__ == "__main__":
    if (len(sys.argv) == 2 or len(sys.argv) == 3) and sys.argv[1] != "-h":
        if len(sys.argv) == 2:
            generate_data(sys.argv[1])
        elif sys.argv[2] == "random":
            generate_data(sys.argv[1], random=True)
        else:
            print("We did not understand the second argument.")
    else:
        print("USAGE: python3", sys.argv[0], "<path to the wav data>")

# if __name__=="__main__":
#     generate_data('./data/rahil_trial_data/')

