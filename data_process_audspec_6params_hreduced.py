#!/usr/bin/env python
# coding: utf-8
'''
Date: March 19
This performs the data_process script to produce audspecs
The spectrogram and the latent variables must have matching dimensions.
The hidden space stores the log(sp). These have to be exponentiated to put into WORLD
Spec=sqrt(spec) for better freq resolution
te: March 19
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
import music_synthesize as music_syn
import json

np.set_printoptions(threshold=sys.maxsize)


def gen_json_list(param_array):
    cu_val = np.empty(6)
    param_array = param_array.tolist()
    # with open("param_nomod.json") as f:
    #     param_list = json.load(f)
    #     # print(param_list)
    final_params = ['OSC: Volume3', 'OSC: Volume2', 'OSC: Tune3',
                    'OSC: Tune2', 'OSC: Shape1', 'OSC: Shape2']
    # final_params = ['ENV1: Decay', 'VCF1: FilterFM', 'OSC: Vibrato', 'OSC: FM',
    #                 'VCF1: Feedback', 'ENV1: Attack', 'ENV1: Sustain',
    #                 'OSC: Volume3', 'OSC: Volume2', 'OSC: OscMix',
    #                 'VCF1: Resonance', 'VCF1: Frequency', 'OSC: Tune3',
    #                 'OSC: Tune2', 'OSC: Shape1', 'OSC: Shape2']
    # final_params = ['Chrs1: Depth','Chrs1: Rate','Chrs1: Wet','Chrs2: Depth','Chrs2: Rate','Chrs2: Wet','Delay1: Center Delay','Delay1: Center Vol',
    #                 'Delay1: Dry','Delay1: Feedback','Delay1: HP','Delay1: LP','Delay1: Right Delay','Delay1: Side Vol',
    #                 'Delay1: Wow','Delay2: Center Delay','Delay2: Center Vol','Delay2: Dry','Delay2: Feedback','Delay2: HP',
    #                 'Delay2: LP','Delay2: Right Delay','Delay2: Side Vol','Delay2: Wow','ENV1: Attack','ENV1: Decay','ENV1: KeyFollow','ENV1: Release','ENV1: Sustain',
    #                 'ENV1: Velocity','ENV2: Attack','ENV2: Decay','ENV2: KeyFollow','ENV2: Release','ENV2: Sustain','ENV2: Velocity',
    #                 'HPF: FreqModDepth','HPF: Frequency','HPF: Resonance','LFO1: Delay','LFO1: DepthMod Dpt1','LFO1: Phase','LFO1: Rate','LFO2: Delay',
    #                 'LFO2: DepthMod Dpt1','LFO2: FreqMod Dpt','LFO2: Phase','LFO2: Rate','MOD: Quantise','MOD: Slew Rate','OSC: DigitalShape2',
    #                 'OSC: DigitalShape3','OSC: DigitalShape4','OSC: Drift','OSC: FM','OSC: FmModDepth','OSC: NoiseVol','OSC: NoiseVolModDepth',
    #                 'OSC: OscMix','OSC: PWModDepth','OSC: PulseWidth','OSC: Shape1','OSC: Shape2','OSC: Shape3','OSC: ShapeDepth',
    #                 'OSC: Tune1','OSC: Tune1ModDepth','OSC: Tune2','OSC: Tune2ModDepth','OSC: Tune3','OSC: Vibrato','OSC: Volume1',
    #                 'OSC: Volume2','OSC: Volume3','Plate1: Diffusion','Phase1: Feedback','Phase1: Phase','Phase1: Rate','Phase1: Stereo',
    #                 'Phase1: Wet','Phase2: Feedback','Phase2: Phase','Phase2: Rate','Phase2: Stereo','Phase2: Wet','Plate1: Damp',
    #                 'Plate1: Decay','Plate1: Diffusion','Plate1: Dry','Plate1: PreDelay','Plate1: Size','Plate1: Wet','Plate2: Damp',
    #                 'Plate2: Decay','Plate2: Diffusion','Plate2: Dry','Plate2: PreDelay','Plate2: Size','Plate2: Wet','Rtary1: Balance',
    #                 'Rtary1: Drive','Rtary1: Fast','Rtary1: Mix','Rtary1: Out','Rtary1: RiseTime','Rtary1: Slow','Rtary1: Stereo',
    #                 'Rtary2: Balance','Rtary2: Drive','Rtary2: Fast','Rtary2: Mix','Rtary2: Slow','Rtary2: Stereo','VCA1: ModDepth','VCA1: PanModDepth','VCA1: Volume',
    #                 'VCF1: Feedback','VCF1: FeedbackModDepth','VCF1: FilterFM','VCF1: FmAmountModDepth','VCF1: FreqMod2Depth',
    #                 'VCF1: FreqModDepth','VCF1: Frequency','VCF1: KeyFollow','VCF1: ResModDepth','VCF1: Resonance','VCF1: ShapeMix','VCF1: ShapeModDepth']
    for i in range(0, 6):
        cu_val[i] = param_array[final_params[i]]
        # param_list[final_params[i]] = param_array[i]
    # print(cu_val)
    return cu_val


def generate_data(path_audio, path_params, audio_time=1, sampling_rate=4000, random=False):   # changed audio time to 1 from 2
    # path='./data/rahil_trial_data/'
    # audio_time = 2
    # sampling_rate = 22050
    # random=True
    # data_type = 'train' or 'cv'
    # data path
    no_of_params = 6  # 16 to 128
    clean_path = path_audio  # read Botinhao data from this path
    directory = path_audio[:-1] + "_1s_hreduced.data"  # save data to this path
    if random:
        directory = path_audio[:-1] + "_random" + "_1s_hreduced.data"  # save data to this path

    audio_len = np.int(audio_time * sampling_rate)
    cur_sample = 0  # current sample generated
    no_files = 0

    # collect raw waveform and trim them into equal length, here is 32000 (16k * 2s)
    spk_wav_tmp = np.zeros((100000, audio_len))
    for (dirpath, dirnames, filenames) in tqdm(os.walk(clean_path)):
        for files in tqdm(filenames):
            if '.wav' in files and '._' not in files:
                s_wav = clean_path + files
                s_wav, s_sr = librosa.load(s_wav, sr=sampling_rate)

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
                spk_wav_tmp[no_files] = spk_wav
                #cur_sample = cur_sample + spk_wav.shape[0]
                no_files = no_files + 1


    spk_wav_tmp = spk_wav_tmp[:no_files, :]
    # spk_wav_tmp.shape= (Num of examples (N), 32000)
    print(spk_wav_tmp.shape)
    #print(spk_wav_tmp[3])
    print('trim finished')  # spk_wav_tmp.shape[0]=Number of files in data directory or Num of examples= N

    param_array = np.zeros(6)  # for music synthesizer preset
    spk_tmp = np.zeros((spk_wav_tmp.shape[0], spk_wav_tmp.shape[1]))  # raw speech with normalized power
    h_tmp = np.zeros((spk_tmp.shape[0], no_of_params, 1))  # ideal hiddens from world
    spec_tmp_513 = np.zeros(
        (spk_wav_tmp.shape[0], 128, 126))  # dimensions of the AudSpec - needs to be softcoded for scalability
    spec_tmp_513_pw = np.zeros((spk_wav_tmp.shape[0], 128,
                                126))  # dimensions of the Reconstructed AudSpec - needs to be softcoded for scalability
    print(spec_tmp_513_pw.shape, 'spec_tmp_513_pw')

    # create an array with parameters audio files
    i = 0
    for (dirpath, dirnames, filenames) in tqdm(os.walk(path_params)):
        for files in tqdm(filenames):
            if '.npz' in files:
                # print(dirnames)
                param_file = path_params + files
                # print(param_file)
                loaded = np.load(param_file, allow_pickle=True)
                loaded_params, loaded_chars, loaded_audio = loaded["param"], loaded["chars"], loaded["audio"]
                # print(loaded_params)
                param_array = gen_json_list(loaded_params)
                # print(param_array)
                param_array = np.reshape(param_array, (6, 1))  # 16 to 128
                # extended_arr = np.tile(param_array, (1, 126))
                # print(extended_arr.shape)
                # print(extended_arr)
                h_tmp[i, :, :] = param_array.copy(order='C')
                # h_tmp[i, :, :] = extended_arr   # 16 to 128
                i += 1

    # print(h_tmp[2, :, :])
    '''Parameters for AudSpectrogram'''
    frmlen = 8
    tc = 8
    paras_c = [frmlen, tc, 0.5, np.log2(sampling_rate / 16000.0)]
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

        spec513 = np.sqrt(nsl.wav2aud(wav, paras_c))  # audSpec
        # print (spec513.shape, 'spec513--line 116')
        spec_tmp_513[i, :, 0:125] = spec513.T  # AudSpec
        # print (spec_tmp_513[i,:,:])

        # # WORLD to extract ideal hiddens
        # f0, sp, ap = pw.wav2world(wav, sampling_rate, frame_period=8.0)
        # # f0 shape: (401,)
        # # ap.shape: (401, 513)
        # # sp.shape: (401, 513)
        # sp = np.log(sp)
        # # print ('sp:',sp)
        # # print ('mean(sp):', np.mean(sp))
        # # print ('max(sp)', np.max(sp))
        # # print ('min(sp)', np.min(sp))

        #
        #     ## use random combinations of parameter values
        #     print('for initialization')
        # f0_ideal = f0
        # ap_ideal = ap
        # # f0, ap = get_f0(f0.shape), get_ap(ap.shape)
        # # print (f0.shape, 'f0.shape')
        # # print (ap.shape, 'ap_shape')
        # f0, ap = get_random_h(f0.shape, ap.shape)
        # f0 = f0.copy(order='C')
        # ap = ap.copy(order='C')
        # sp=np.flipud(sp)
        # sp = sp.copy(order='C')

        # # h_tmp[i,0,:] = f0.copy(order='C')
        # # h_tmp[i,1:258, :] = sp.T.copy(order='C')
        # # h_tmp[i,258:,:] = ap.T.copy(order='C')
        # h_tmp[i, 0, :] = f0.copy(order='C')  # H_tmp is saved!
        # h_tmp[i, 1:514, :] = sp.T.copy(order='C')  # we're saving the log(sp)
        # h_tmp[i, 514:, :] = ap.T.copy(order='C')

        # # # reconstructed speech
        # # sp = np.exp(sp)  # To hava the same scale as D(x)
        # # # sp = np.log(sp) # To hava the same scale as D(x)
        #
        # # y = pw.synthesize(f0, sp, ap, sampling_rate, frame_period=8.0)
        # y = music_syn.music_synthesize(param_array, sampling_rate)
        # # print (len(y), 'len(y)')
        # # y = y[pad:pad+32000/2]
        # y = y[pad:pad + 32000]
        #
        # y = y / np.sqrt(np.sum(y ** 2))  # power normalization
        # y = nsl.unitseq(y)

        # if random:
        #     # reconstructed speech
        #     # y_ideal = pw.synthesize(f0_ideal, sp, ap_ideal, sampling_rate, frame_period=8.0)
        #     y_ideal = music_syn.music_synthesize(param_array_ideal, sampling_rate)
        #     y_ideal = y_ideal[pad:pad + 32000]
        #     # y_ideal = y_ideal[pad:pad+32000]
        #
        #     y_ideal = y_ideal / np.sqrt(np.sum(y ** 2))  # power normalization
        #     # spec_ideal = np.abs(librosa.core.stft(y_ideal, n_fft=1024, hop_length = 80))
        #     spec_ideal = nsl.wav2aud(nsl.unitseq(y_ideal), paras_c)
        #     # scipy.io.wavfile.write("test.wav", sampling_rate, y)

        # if random:
        #     print('for initialization')
        #     y = music_syn.music_synthesize(param_array, sampling_rate)
        #     # print (len(y), 'len(y)')
        #     # y = y[pad:pad+32000/2]
        #     y = y[pad:pad + 32000]
        #     y = y / np.sqrt(np.sum(y ** 2))  # power normalization
        #     y = nsl.unitseq(y)
        #     spk_tmp[i, :] = y
        #
        #     spec513_pw = nsl.wav2aud(y, paras_c)
        #     spec513_pw = np.sqrt(spec513_pw.T)
        #     # print (spec513_pw)
        #     spec_tmp_513_pw[i, :, 0:250] = spec513_pw
        #
        #     spec_tmp_513[i, :, :] = spec_tmp_513_pw[i, :, :]
        # y=nsl.unitseq(y)

        # spec513_pw = nsl.wav2aud(y, paras_c)
        # spec513_pw = np.sqrt(spec513_pw.T)
        # # print (spec513_pw)
        # spec_tmp_513_pw[i, :, 0:250] = spec513_pw
        # print (spec_tmp_513_pw)

        # if random:
        #     spec_tmp_513[i, :, :] = spec_tmp_513_pw[i, :, :]

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

    # plt.imshow(spec_tmp_513_pw[i,:,:])
    # plt.title("GENERATED SPECTROGRAM")
    # plt.show()

    dset = h5py.File(directory, 'w')
    print(spk_tmp.shape)
    spk_set = dset.create_dataset('speaker', shape=(spk_tmp.shape[0], spk_tmp.shape[1]), dtype=np.float64)
    hid_set = dset.create_dataset('hidden', shape=(spk_tmp.shape[0], no_of_params, 1), dtype=np.float64)
    # spec513_set = dset.create_dataset('spec513', shape=(spk_tmp.shape[0], 513, 401), dtype=np.float64)
    # spec_513_pw_set = dset.create_dataset('spec513_pw', shape=(spk_tmp.shape[0], 513, 401), dtype=np.float64)

    spec513_set = dset.create_dataset('spec513', shape=(spk_tmp.shape[0], 128, 126), dtype=np.float64)
    spec_513_pw_set = dset.create_dataset('spec513_pw', shape=(spk_tmp.shape[0], 128, 126), dtype=np.float64)

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


