# %%
import argparse
import librenderman as rm
import numpy as np
import json, ast
import scipy.io.wavfile
import librosa
import glob, os
import time

my_path = os.path.dirname(os.path.abspath(__file__))

def resample(y, orig_sr, target_sr):
	if orig_sr == target_sr:
		return y
	ratio = float(target_sr) / orig_sr
	n_samples = int(np.ceil(y.shape[-1] * ratio))
	y_hat = scipy.signal.resample(y, n_samples, axis=-1)  # maybe resampy is better?
	# y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)
	return np.ascontiguousarray(y_hat, dtype=y.dtype)


def play_patch(engine, patch_gen, midiNote, midiVelocity, noteLength, renderLength, patch=None):
	if patch is None:
		patch = patch_gen.get_random_patch()
	engine.set_patch(patch)
	# Settings to play a note and extract data from the synth.
	engine.render_patch(midiNote, midiVelocity, noteLength, renderLength)
	# engine.render_patch(midiNote, midiVelocity, noteLength, renderLength, True)
	# engine.render_patch(midiNote, midiVelocity, noteLength, renderLength) #render twice to get rid of blip
	audio = engine.get_audio_frames()
	return np.array(audio), patch


def midiname2num(patch, rev_diva_midi_desc):
	"""
	converts param dict {param_name: value,...} to librenderman patch [(param no., value),..]
	"""
	return [(rev_diva_midi_desc[k], float(v)) for k, v in patch.items()]


def create_synth(dataset, path='Diva.64.so'):
	with open("diva_params.txt") as f:
		diva_midi_desc = ast.literal_eval(f.read())
	rev_idx = {diva_midi_desc[key]: key for key in diva_midi_desc}
	if dataset == "toy":
		with open("param_nomod.json") as f:
			param_defaults = json.load(f)
	else:
		with open("param_default_32.json") as f:
			param_defaults = json.load(f)
	engine = rm.RenderEngine(44100, 512, 512)
	engine.load_plugin(path)
	generator = rm.PatchGenerator(engine)
	return engine, generator, param_defaults, rev_idx


def synthesize_audio(params, engine, generator, rev_idx, midiNote, midiVelocity, noteLength, renderLength):
	# Replace param_defaults with whatever preset to play
	patch = midiname2num(params, rev_idx)
	audio, patch = play_patch(engine, generator, midiNote, midiVelocity, noteLength, renderLength, patch=patch)
	return audio


def music_synthesize(passed_params, file_name, engine, generator, rev_idx, midiNote, midiVelocity, noteLength, renderLength=4):
	# print(passed_params)

	#loaded_params = gen_json_list(passed_params)
	# passed_params = passed_params.tolist()
	print('[Synthesize Music]')
	final_audio = synthesize_audio(passed_params, engine, generator, rev_idx, midiNote, midiVelocity, noteLength, renderLength)
	write_to_wav(final_audio, file_name)
	return final_audio

def gen_json_list(param_passed):
	# cu_val = np.empty(16)
	with open("param_nomod.json") as f:
		param_list = json.load(f)
	# final_params = ['ENV1: Decay', 'VCF1: FilterFM', 'OSC: Vibrato', 'OSC: FM',
	#                 'VCF1: Feedback', 'ENV1: Attack', 'ENV1: Sustain',
	#                 'OSC: Volume3', 'OSC: Volume2', 'OSC: OscMix',
	#                 'VCF1: Resonance', 'VCF1: Frequency', 'OSC: Tune3',
	#                 'OSC: Tune2', 'OSC: Shape1', 'OSC: Shape2']
	final_params = ['OSC: Volume3', 'OSC: Volume2', 'OSC: Tune3',
					'OSC: Tune2', 'OSC: Shape1', 'OSC: Shape2']
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
	# print(param_list)
	# param_passed_list = param_passed.tolist()
	# print(len(final_params))
	for i in range(0,6):
		# cu_val = param_list[final_params[i]]
		param_list[final_params[i]] = param_passed[i]
	# print(param_list)
	return param_list

def write_to_wav(audio, file_name):
	#final_audio = resample(audio, 44100, 4000)
	audio = audio / np.sqrt(np.sum(np.array(audio) ** 2))
	scipy.io.wavfile.write(my_path +'/music_data_6params_new/train_audio_new3/' + file_name + '.wav', 44100, (np.iinfo(np.int16).max*audio/np.max(audio)).astype(np.int16))

def generate(params, outName, engine, generator, parameters, rev_idx, pad=0):
	'''
	params: midiPitch, MidiValocity, midiDuration, CutOff Frequency
	'''

	midiNote, volume2, midiDuration, VCF1 = params

	#engine, generator, parameters, rev_idx = create_synth('toy')
	my_params = {
	#"OSC: Volume2": 1,
	"OSC: Volume3": 0, 
	"OSC: Vibrato": 0,
	"ENV1: Sustain": .6,
	"main: Output": 1,
	"OSC: Volume1": 0,
	"OSC: NoiseVol": 0,
	"OSC: Noise1On": 0,
	"VCF1: Feedback":0,
	"OSC: NoiseVolModSrc": 0,
	"ENV1: Attack": 0, 
	"ENV2: Attack": 0,
	"ENV2: Decay":0,
	"ENV2: Sustain":0,
	"ENV2: Release":0,
	"Chrs1: Wet": 0, 
	"Phase1: Wet": 0,
	"Plate1: Wet": 0, 
	"Chrs2: Wet": 0, 
	"Phase2: Wet": 0,
	"Plate2: Wet": 0, 
	"OPT: EnvrateSlop": 0, 
	"VCF1: FreqModDepth":.6,
	"VCF1: FreqMod2Depth":.5,
	"VCF1: FreqModSrc": .5,
	"VCF1: FilterFM": .5,
	"VCF1: Resonance": 0,
	"ENV1: Velocity": 1, 
	"ENV2: Velocity": 1,
	}

	audio = []

	for i in range(len(midiNote)):

		for param in parameters:
			if param in my_params:
				parameters[param] = my_params[param]
			elif param == "VCF1: Frequency":
				parameters[param] = VCF1[i]/150.0
			elif param == "OSC: Volume2":
				parameters[param] = volume2[i]


		audio.extend(music_synthesize(parameters, outName, engine, generator, rev_idx, midiNote[i], 127, midiDuration[i], renderLength=midiDuration[i]+pad))

	#write_to_wav(audio, outName)
	return audio



if __name__ == "__main__":

	engine, generator, parameters, rev_idx = create_synth('toy')
	N = 5 # number of notes
	rangePitch = [62, 64, 65, 67, 69, 71, 72, 74]
	rangeVolume = [.7, 1]
	rangeDuration = [.4, .9]
	rangeFilter = [50, 100]
        sigmoid_pitch = [0,0.16666666,0.25,0.41666666, 0.58333333,0.75,0.83333333,1]
        sigmoid_r = [0, 1]
	pad = 0.1
        seg_duration = 2
	for i in range(400):
		pitch = np.random.choice(sigmoid_pitch, N)
		volume = np.random.uniform(sigmoid_r[0], sigmoid_r[1], size=N)
		duration = np.random.uniform(sigmoid_r[0], sigmoid_r[1], size=N)
		filterCutOff = np.random.uniform(sigmoid_r[0], sigmoid_r[1], size=N)
                np.save(my_path + '/music_data_6params_new/train_random_params_new3/'+ 'train_random_params_' + str(i), np.array([pitch, volume, duration, filterCutOff]))

                pitch_tr = (np.rint((pitch * 12) + 62)).astype(int)
                volume_tr = (volume * 0.3) + 0.7
                duration_tr = (duration * 0.5) + 0.4
                filterCutOff_tr = (filterCutOff * 50) + 50.0

		duration_full = seg_duration*np.array(duration_tr)/np.sum(duration_tr) - pad
		#generate([pitch_tr, volume_tr, duration_full, filterCutOff_tr], "train_audio_"+str(i), engine, generator, parameters, rev_idx, pad=pad)
		#pitch_tr = (pitch - 62)/12.0
		#volume_tr = (volume-0.7)/0.3
		#duration_tr = (duration-0.4)/0.5
		#filterCutOff_tr = (filterCutOff - 50)/50.0 
                #np.save(my_path + '/music_data_6params_new/test_params_new2/'+ 'test_params_' + str(i), np.array([pitch_tr, volume_tr, duration_tr, filterCutOff_tr]))
		#print(pitch_tr)
                #print(duration_tr)








