'''
Compute the distance between the real spectrogram and the WORLD and decoder spectrograms to evaluate how well the network works
Date: 04-20-2020 
BLAZE IT! 
'''
from __future__ import print_function
import argparse

import matplotlib

SERVER = True # This variable enable or disable matplotlib, set it to true when you use the server!
if SERVER:
    matplotlib.use('Agg')

import sys
import os
import numpy as np
import h5py
import time
import subprocess
import logging
#from IPython.display import Audio
# import pyworld as pw
# import librosa
from datetime import date
import matplotlib.pyplot as plt
import pickle
import datetime
# from scipy.io.wavfile import write
# from random_generation import get_f0, get_ap
# import inspect
# import nsltools as nsl 
import scipy.io.wavfile
import scipy.io as scio
import scipy.optimize as so
import soundfile as sf
import matplotlib.image as image
import pickle as pkl
import scipy
from scipy import stats
import nsltools as nsl

def L2_distance(spec1, spec2):
	distance= np.linalg.norm(spec1-spec2)
	return distance

def L1_distance(spec1, spec2):
	distance=np.abs(spec1-spec2)
	# distance= np.linalg.norm(spec1-spec2, ord=1)
	return distance.sum()

def variance_difference(spec1, spec2):
	normalized_spec1=stats.zscore(spec1, axis=0, ddof=1)
	normalized_spec2=stats.zscore(spec2, axis=0, ddof=1)
	diff=normalized_spec1-normalized_spec2
	return np.var(diff, ddof=1)


def distance_metric(spec1, spec2, name=L2_distance):
	return name(spec1, spec2)

distance_between_WORLD_and_real_ar=[]
distance_between_decoder_and_real_ar=[]
distance_between_WORLD_and_decoder_ar=[]

metric_list=[L2_distance, L1_distance, variance_difference]
total_speakers=10 #number of spectrograms I'm saving
square_root_of_spectrograms=True #since we use the sqrt(spectrograms) as an input into the network
reconstruct_wavefiles=True #Flag for converting spectrograms to wave files

figs_folder='NEW_with_only_1_loss_for_E_DATE_2020-05-15_H_19_50%_Init'	#figs folder

for metric in metric_list:

	for speaker_index in range(total_speakers):

		with open('./figs/'+figs_folder+'/spectrograms/modelSpectrogram{}.pkl'.format(speaker_index), 'rb') as f:
			modelSpectrogram=pkl.load(f)

		with open('./figs/'+figs_folder+'/spectrograms/RealSpectrogram{}.pkl'.format(speaker_index), 'rb') as f:
			realSpectrogram=pkl.load(f)

		with open('./figs/'+figs_folder+'/spectrograms/worldSpectrogram{}.pkl'.format(speaker_index), 'rb') as f:
			worldSpectrogram=pkl.load(f)

		if square_root_of_spectrograms:
			modelSpectrogram=np.power(modelSpectrogram,2)
			realSpectrogram=np.power(realSpectrogram,2)
			worldSpectrogram=np.power(worldSpectrogram,2)

		dist_WORLD_and_real=distance_metric(realSpectrogram.T, worldSpectrogram.T, metric)
		dist_decoder_and_real=distance_metric(modelSpectrogram.T, realSpectrogram.T, metric)
		dist_WORLD_and_decoder=distance_metric(worldSpectrogram.T, modelSpectrogram.T, metric)


		# print ('--'*100)
		# print (speaker_index)
		# print (L2_distance(worldSpectrogram, realSpectrogram), 'L2_distance(worldSpectrogram, realSpectrogram)')
		# print (L2_distance(modelSpectrogram, realSpectrogram), 'L2_distance(modelSpectrogram, realSpectrogram)')
		# print (L2_distance(worldSpectrogram, modelSpectrogram), 'L2_distance(worldSpectrogram, modelSpectrogram)')


		distance_between_WORLD_and_real_ar.append(dist_WORLD_and_real)
		distance_between_decoder_and_real_ar.append(dist_decoder_and_real)
		distance_between_WORLD_and_decoder_ar.append(dist_WORLD_and_decoder)

	distance_between_WORLD_and_real=np.mean(distance_between_WORLD_and_real_ar)
	distance_between_decoder_and_real=np.mean(distance_between_decoder_and_real_ar)
	distance_between_WORLD_and_decoder=np.mean(distance_between_WORLD_and_decoder_ar)

	std_between_WORLD_and_real=np.std(distance_between_WORLD_and_real_ar)
	std_between_decoder_and_real=np.std(distance_between_decoder_and_real_ar)
	std_between_WORLD_and_decoder=np.std(distance_between_WORLD_and_decoder_ar)


	# print (distance_between_WORLD_and_real, 'distance_between_WORLD_and_real')
	# print (distance_between_decoder_and_real, 'distance_between_decoder_and_real')
	# print (distance_between_WORLD_and_decoder, 'distance_between_WORLD_and_decoder')
	# print (metric)

	fn = open('./figs/'+figs_folder+"/"+"eval_{}.txt".format(metric.__name__), 'w')

	print ('EVALUATION DATASET', file=fn)
	print ('DISTANCE METRIC- %s' %(metric.__name__), file=fn)
	print ('distance_between_WORLD_and_real- %f' %(distance_between_WORLD_and_real), file=fn)
	print ('std_between_WORLD_and_real- %f' %(std_between_WORLD_and_real), file=fn)

	print ('distance_between_decoder_and_real- %f' %(distance_between_decoder_and_real), file=fn)
	print ('std_between_decoder_and_real- %f' %(std_between_decoder_and_real), file=fn)

	print ('distance_between_WORLD_and_decoder- %f' %(distance_between_WORLD_and_decoder), file=fn)
	print ('std_between_WORLD_and_decoder- %f' %(std_between_WORLD_and_decoder), file=fn)
	fn.close()


if reconstruct_wavefiles:
	fs=16000;
	paras = [8, 8, -2, np.log2(fs/16000)]
	nitr=50
	parasout = paras + [nitr, 0, 0]
	x=np.random.rand(32000)

	x_real= nsl.aud2wav(realSpectrogram.T,x,parasout)[1]
	x_model= nsl.aud2wav(modelSpectrogram.T,x,parasout)[1]
	x_world= nsl.aud2wav(worldSpectrogram.T,x,parasout)[1]

	scipy.io.wavfile.write('./figs/'+figs_folder+"/"+'real.wav',fs,x_real)
	scipy.io.wavfile.write('./figs/'+figs_folder+"/"+'model_recon.wav',fs,x_model)
	scipy.io.wavfile.write('./figs/'+figs_folder+"/"+'world_recon.wav',fs,x_world)
