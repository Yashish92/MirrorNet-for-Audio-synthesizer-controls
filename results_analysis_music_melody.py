from __future__ import print_function
import argparse

import matplotlib

SERVER = True  # This variable enable or disable matplotlib, set it to true when you use the server!
if SERVER:
    matplotlib.use('Agg')
import sys
import os
import numpy as np
# import h5py
import time
import subprocess
import logging
# from IPython.display import Audio
# import pyworld as pw
import music_synthesize_new as music_syn
import librosa
from datetime import date
import matplotlib.pyplot as plt
import pickle as pkl
import datetime
import scipy
from scipy.io.wavfile import write
import inspect
import nsltools as nsl
import scipy.io as scio

# speaker_index = 0
# print(speaker_index, '-->Speaker Index')
analyze_sp = False
analyze_sectrograms = False
recon_from_latent = True
frmlen = 8
tc = 8
sampling_rate = 4000
paras_c = [frmlen, tc, 0.5, np.log2(sampling_rate / 16000)]
parasout = paras_c + [40, 0, 0]
# xt, s_sr = librosa.load('./debug_female_voice/Speaker_wav{}.wav'.format(speaker_index), sr=sampling_rate)
fs = sampling_rate
figs_folder = 'NEW_with_only_1_loss_for_E_DATE_2021-02-07_H_1'
# name = figs_folder + 'Speaker%d_results/' % (speaker_index)

# Create music synthesier
engine, generator, parameters, rev_idx= music_syn.create_synth('toy')

# if not os.path.exists(name):
#     os.makedirs(name)

# if analyze_sp:
#     print('Analyzing SP')
#     with open('./figs/' + figs_folder + '/end_evaluation/lentent_space/sp_hat.pkl', 'rb') as f:
#         sp_hat = pkl.load(f)
#
#     with open('./figs/' + figs_folder + '/end_evaluation/lentent_space/sp_ideal.pkl', 'rb') as f:
#         sp_ideal = pkl.load(f)
#
#     sp_hat = sp_hat[speaker_index, :, :]
#
#     sp_ideal = sp_ideal[speaker_index, :, :]
#
#     # sp_hat=(np.abs(sp_hat))
#     # sp_ideal=(np.abs(sp_ideal))
#     # # sp_hat[sp_hat<0]=10**-15
#     plt.title("Sp predicted")
#     # plt.subplot(121)
#     # plt.title("from encoder (E(x))")
#     plt.imshow((sp_hat[:, :]), cmap=plt.cm.BuPu_r)
#     plt.colorbar()
#     plt.savefig(name + 'sp_hat.eps')
#     # plt.subplot(122)
#     # plt.title("from world (ideal h)")
#     plt.figure()
#     plt.imshow((sp_ideal[:, :]), cmap=plt.cm.BuPu_r)
#     plt.title("Sp real")
#     plt.colorbar()
#     plt.savefig(name + 'sp_ideal.eps')
#
# # sp_diff=(sp_ideal-sp_hat)
# # plt.title('(sp_ideal)-(sp_hat)')
# # plt.imshow(sp_diff, cmap=plt.cm.BuPu_r)
# # plt.colorbar()
# # plt.savefig('SP Diff')
#
# if analyze_sectrograms:
#     print('Analysing Spectrograms')
#     with open('./figs/' + figs_folder + '/spectrograms/decoderSpectrogram{}.pkl'.format(speaker_index), 'rb') as f:
#         decoderSpectrogram = pkl.load(f)
#
#     with open('./figs/' + figs_folder + '/spectrograms/modelSpectrogram{}.pkl'.format(speaker_index), 'rb') as f:
#         modelSpectrogram = pkl.load(f)
#
#     with open('./figs/' + figs_folder + '/spectrograms/RealSpectrogram{}.pkl'.format(speaker_index), 'rb') as f:
#         realSpectrogram = pkl.load(f)
#
#     with open('./figs/' + figs_folder + '/spectrograms/worldSpectrogram{}.pkl'.format(speaker_index), 'rb') as f:
#         worldSpectrogram = pkl.load(f)
#
#     plt.subplot(221)
#     plt.imshow(realSpectrogram, cmap='jet', origin='lower')  # Ground Truth
#     plt.title('Real Spectrogram')
#     plt.subplot(222)
#     plt.imshow(decoderSpectrogram, cmap='jet', origin='lower')
#     plt.title('Decoder Spectrogram (using ideal h)')  # D(ideal_h)
#     plt.subplot(223)
#     plt.imshow(modelSpectrogram, cmap='jet', origin='lower')
#     plt.title('Model Spectrogram: D(E(x))')  # (D(E(X)))
#     plt.subplot(224)
#     plt.imshow(worldSpectrogram, cmap='jet', origin='lower')
#     plt.title('World Spectrogram (from encoder h)')  # World(E(H))
#     plt.savefig(name + 'Spectrograms.eps')
#
#     # scio.savemat('modelSpectrogram.mat',{'modelSpectrogram':modelSpectrogram})
#     # scio.savemat('decoderSpectrogram.mat',{'decoderSpectrogram':decoderSpectrogram})
#     # scio.savemat('realSpectrogram.mat',{'realSpectrogram':realSpectrogram})
#     # scio.savemat('worldSpectrogram.mat',{'worldSpectrogram':worldSpectrogram})
#
#     modelSpectrogram = nsl.aud_fix(modelSpectrogram.T)
#     modelSpectrogram_wav = nsl.unitseq(nsl.aud2wav(modelSpectrogram, xt, parasout)[1])
#     # modelSpectrogram_wav=( nsl.aud2wav(modelSpectrogram,xt,parasout)[1])
#     scipy.io.wavfile.write(name + 'modelSpectrogram_wav.wav', 16000, modelSpectrogram_wav)
#     # scipy.io.wavfile.write('y_ground_Truth_wav.wav' ,16000,xt)
#
#     worldSpectrogram = nsl.aud_fix(worldSpectrogram.T)
#     worldSpectrogram_wav = nsl.unitseq(nsl.aud2wav(worldSpectrogram, xt, parasout)[1])
#     # worldSpectrogram_wav=( nsl.aud2wav(worldSpectrogram,xt,parasout)[1])
#     scipy.io.wavfile.write(name + 'worldSpectrogram_wav.wav', 16000, worldSpectrogram_wav)
#
#     realSpectrogram = nsl.aud_fix(realSpectrogram.T)
#     realSpectrogram_wav = nsl.unitseq(nsl.aud2wav(realSpectrogram, xt, parasout)[1])
#     scipy.io.wavfile.write(name + 'realSpectrogram_wav.wav', 16000, realSpectrogram_wav)
#
# # xt, s_sr = librosa.load('Speaker_wav0.wav', sr = sampling_rate)
# # xt=nsl.unitseq(xt)
# # yc=nsl.wav2aud(xt, paras_c).T
# # plt.title('Speaker 0 Spec')
# # plt.imshow(yc, cmap='jet', origin='lower')
# # plt.savefig('Speaker_0_Spec.eps')
#
# # f0, sp, ap = pw.wav2world(xt, sampling_rate, frame_period=8.0)
if recon_from_latent:
    print('recon from latent')
    # with open('./figs/' + figs_folder + '/init_evaluation/lentent_space/sp_hat.pkl', 'rb') as f:
    #     sp_hat = pkl.load(f)
    #
    # with open('./figs/' + figs_folder + '/init_evaluation/lentent_space/sp_ideal.pkl', 'rb') as f:
    #     sp_ideal = pkl.load(f)

    # with open('./figs/' + figs_folder + '/end_evaluation/lentent_space/ap_ideal.pkl', 'rb') as f:
    #     ap_ideal = pkl.load(f)
    #
    # with open('./figs/' + figs_folder + '/end_evaluation/lentent_space/ap_hat.pkl', 'rb') as f:
    #     ap_hat = pkl.load(f)

    # with open('./figs/' + figs_folder + '/init_evaluation/lentent_space/f0_ideal.pkl', 'rb') as f:
    #     f0_ideal = pkl.load(f)
    #
    # with open('./figs/' + figs_folder + '/init_evaluation/lentent_space/f0_hat.pkl', 'rb') as f:
    #     f0_hat = pkl.load(f)
    # sp_ideal_c=sp_ideal.copy(order='C')
    # ap_ideal_c=ap_ideal.copy(order='C')
    # f0_ideal_c=f0_ideal.copy(order='C')
    # np.ascontiguousarray(sp_ideal)
    # np.ascontiguousarray(ap_ideal)
    # np.ascontiguousarray(f0_ideal)



    # f0_ideal = np.squeeze(f0_ideal[speaker_index, :, :])
    # sp_ideal = np.squeeze(sp_ideal[speaker_index, :, :])
    for speaker_index in range(0, 10):
        name = figs_folder + '/Speaker%i_results/' % (speaker_index)
        if not os.path.exists(name):
            os.makedirs(name)

        with open('./figs/' + figs_folder + '/end_evaluation/lentent_space/ap_ideal.pkl', 'rb') as f:
            ap_ideal = pkl.load(f)

        with open('./figs/' + figs_folder + '/end_evaluation/lentent_space/ap_hat.pkl', 'rb') as f:
            ap_hat = pkl.load(f)

        ap_ideal = np.squeeze(ap_ideal[speaker_index, :, :])
        music_ideal = ap_ideal[:, :].copy(order='C')  # 406, 513

        # f0_hat = np.squeeze(f0_hat[speaker_index, :, :])
        ap_hat = np.squeeze(ap_hat[speaker_index, :, :])
        music_hat = ap_hat[:, :].copy(order='C')  # 406, 513
        # sp_hat = np.squeeze(sp_hat[speaker_index, :, :])

        # y_ideal = pw.synthesize(f0_ideal, (np.ascontiguousarray(sp_ideal.T)), np.ascontiguousarray(ap_ideal.T),
        #                         sampling_rate, frame_period=8.0)
        #print(music_ideal.shape, music_hat.shape)
        y_ideal = music_syn.generate(music_ideal, engine, generator, parameters, rev_idx, pad=0)
        y_ideal = music_syn.resample(y_ideal, 44100, fs)  # resampling
        # y_ideal = y_ideal[pad:pad + 32000]
        y_ideal = y_ideal / np.sqrt(np.sum(y_ideal ** 2))
        y_ideal = nsl.unitseq(y_ideal)
        scipy.io.wavfile.write(name + 'y_ideal_latent_recon.wav', 4000, y_ideal)

        y_hat = music_syn.generate(music_hat, engine, generator, parameters, rev_idx, pad=0)
        y_hat = music_syn.resample(y_hat, 44100, fs)  # resampling
        # y_ideal = y_ideal[pad:pad + 32000]
        y_hat = y_hat / np.sqrt(np.sum(y_hat ** 2))
        y_hat = nsl.unitseq(y_hat)
        scipy.io.wavfile.write(name + 'y_hat_latent_recon.wav', 4000, y_hat)

        # np.ascontiguousarray(sp_hat)
        # np.ascontiguousarray(ap_hat)
        # np.ascontiguousarray(f0_hat)
        # y_hat = pw.synthesize(f0_hat, (np.ascontiguousarray(sp_hat.T)), np.ascontiguousarray(ap_hat.T), sampling_rate,
        #                       frame_period=8.0)
        # y_hat = y_hat / np.sqrt(np.sum(y_hat ** 2))
        # y_hat = nsl.unitseq(y_hat)
        # scipy.io.wavfile.write(name + 'y_hat_latent_recon.wav', 16000, y_hat)
