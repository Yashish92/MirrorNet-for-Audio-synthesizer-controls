"""
@Author : Yashish
"""
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


#def resample(y, orig_sr, target_sr):
#    if orig_sr == target_sr:
#        return y
#    ratio = float(target_sr) / orig_sr
#    n_samples = int(np.ceil(y.shape[-1] * ratio))
#    y_hat = scipy.signal.resample(y, n_samples, axis=-1)  # maybe resampy is better?
#    # y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)
#    return np.ascontiguousarray(y_hat, dtype=y.dtype)
#
#
#def play_patch(engine, patch_gen, midiNote, midiVelocity, noteLength, renderLength, patch=None):
#    if patch is None:
#        patch = patch_gen.get_random_patch()
#    engine.set_patch(patch)
#    # Settings to play a note and extract data from the synth.
#    engine.render_patch(midiNote, midiVelocity, noteLength, renderLength)
#    # engine.render_patch(midiNote, midiVelocity, noteLength, renderLength, True)
#    # engine.render_patch(midiNote, midiVelocity, noteLength, renderLength) #render twice to get rid of blip
#    audio = engine.get_audio_frames()
#    return np.array(audio), patch
#

def resample(y, orig_sr, target_sr):
    y = np.array(y)
    #print(y)
    if orig_sr == target_sr:
        return y
    ratio = float(target_sr) / orig_sr
    n_samples = int(np.ceil(y.shape[-1] * ratio))
    y_hat = scipy.signal.resample(y, n_samples, axis=-1)  # maybe resampy is better?
    #np.array(y_hat)
    # y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)
    return np.ascontiguousarray(y_hat, dtype=y.dtype)


def play_patch(engine, patch_gen, midiNote, midiVelocity, noteLength, renderLength, patch=None):
    if patch is None:
        patch = patch_gen.get_random_patch()
    engine.set_patch(patch)
    # Settings to play a note and extract data from the synth.
    engine.render_patch(int(midiNote), int(midiVelocity), float(noteLength), float(renderLength))
    #engine.render_patch(midiNote, midiVelocity, noteLength, renderLength)
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


def music_synthesize(passed_params, engine, generator, rev_idx, midiNote, midiVelocity, noteLength,
                     renderLength=4):
    # print(passed_params)

    # loaded_params = gen_json_list(passed_params)
    # passed_params = passed_params.tolist()
    # print('[Synthesize Music]')
    final_audio = synthesize_audio(passed_params, engine, generator, rev_idx, midiNote, midiVelocity, noteLength,
                                   renderLength)
    # write_to_wav(final_audio, file_name)
    return final_audio


def gen_json_list(param_array):
    cu_val = np.empty(16)
    with open("param_nomod.json") as f:
        param_list = json.load(f)
    final_params = ['ENV1: Decay', 'VCF1: FilterFM', 'OSC: Vibrato', 'OSC: FM',
                    'VCF1: Feedback', 'ENV1: Attack', 'ENV1: Sustain',
                    'OSC: Volume3', 'OSC: Volume2', 'OSC: OscMix',
                    'VCF1: Resonance', 'VCF1: Frequency', 'OSC: Tune3',
                    'OSC: Tune2', 'OSC: Shape1', 'OSC: Shape2']
    for i in range(0,16):
        # cu_val[i] = param_list[final_params[i]]
        param_list[final_params[i]] = param_array[i]
    # print(param_list)
    return param_list


def write_to_wav(audio, file_name):
    # final_audio = resample(audio, 44100, 4000)
    audio = audio / np.sqrt(np.sum(np.array(audio) ** 2))
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
    scipy.io.wavfile.write(audio_path + '/' + file_name + '.wav', 44100,
                           (np.iinfo(np.int16).max * audio / np.max(audio)).astype(np.int16))


def generate(params, outName, engine, generator, parameters, rev_idx, pad=0):
    '''
    params: midiPitch, MidiValocity, midiDuration, CutOff Frequency
    '''
    params_dic = {}
    # diva_parameters = ["VCF1: Frequency", "OSC: Volume2", "VCF1: Resonance", "ENV1: Attack", "ENV1: Decay"]
    diva_parameters = ["OSC: Volume2", "VCF1: Frequency", "VCF1: Resonance", "ENV1: Attack", "ENV1: Decay",		# to generate 10 params meldoies
                       "LFO1: Rate", "OSC: Vibrato", "LFO1: Phase"]
    #diva_parameters = ["VCF1: Frequency", "OSC: Volume2", "VCF1: Resonance", "ENV1: Attack" ]				# to generate 7 params meldoies
    my_params = {
        # "OSC: Volume2": 1,
        "OSC: Volume3": 0,
        # "OSC: Vibrato": 0,
        "ENV1: Sustain": 0,  # changed back to 0.6
        # "ENV1: Decay": 0.4,
        "main: Output": 1,
        "OSC: Volume1": 0,
        "OSC: NoiseVol": 0,
        "OSC: Noise1On": 0,
        "VCF1: Feedback": 0,
        "OSC: NoiseVolModSrc": 0,
        # "ENV1: Attack": 0,
        "ENV2: Attack": 0,
        "ENV2: Decay": 0,
        "ENV2: Sustain": 0,
        "ENV2: Release": 0,
        "Chrs1: Wet": 0,
        "Phase1: Wet": 0,
        "Plate1: Wet": 0,
        "Chrs2: Wet": 0,
        "Phase2: Wet": 0,
        "Plate2: Wet": 0,
        "OPT: EnvrateSlop": 0,
        "VCF1: FreqModDepth": 0.5,  # changed from 0.6
        "VCF1: FreqMod2Depth": 1,  # changed from 0.5
        "VCF1: FreqModSrc": .5,
        "VCF1: FilterFM": .5,
        "ENV1: Velocity": 1,
        "ENV2: Velocity": 1,
        "VCF1: SvfMode": 1,
        "VCF1: Model": 0.5,
        "LFO1: DepthMod Dpt1": 0,
        # "LFO1: Sync": 0,
        "LFO1: Waveform": 0,
    }

    melody_dur = 2
    midiNote = (params[0] * 12) + 62
    midiNote = np.rint(midiNote).astype(int)
    #volume2  = (params[1]*0.3) + 0.7
    #midiDuration = 2.1 * params[2] / np.sum(params[2])
    midi_dur = (params[1]*0.5) + 0.4
    midiDuration = melody_dur * midi_dur/ np.sum(midi_dur) - pad
    #mean_freqmod = 0.5
    #std_freqmod = 0.07

    i = 2    # change i accordingly to set the remaining parameters after setting first 4 parameters
    for p in diva_parameters:
        # if p == "VCF1: Frequency":
        #     params_dic[p] = ((params[2] * 25) + 55) / 150.0  # changed filter range from 55 to 80
        # elif p == "OSC: Volume2":
        #     params_dic[p] = (params[3]*0.3) + 0.7
        #elif p == "VCF1: FreqMod2Depth":
            #m_data = np.mean(params[4])
            #std_data = np.std(params[4])
            #trans_data = mean_freqmod + (params[4]-m_data)*(std_freqmod/std_data)
        #    params_dic[p] = params[4]
        #elif p == "LFO1: Rate":
        #   params_dic[p] = (params[4]*0.4)
        params_dic[p] = params[i]    # setting all parameters, be careful with their order btw functions!!!!
        i+=1

    audio = []

    for i in range(len(midiNote)):
        for param in parameters:
            if param in params_dic:
                parameters[param] = params_dic[param][i]
            elif param in my_params:
                parameters[param] = my_params[param]
        #            elif param == "VCF1: Frequency":
        #                parameters[param] = VCF1[i] / 150.0
        #            elif param == "OSC: Volume2":
        #                parameters[param] = volume2[i]
        #
        audio.extend(
            music_synthesize(parameters, engine, generator, rev_idx, midiNote[i], 127, midiDuration[i],
                             renderLength=midiDuration[i] + pad))

    # write_to_wav(audio, outName)
    final_audio = np.array(audio)

    write_to_wav(final_audio, outName)
    return final_audio


if __name__ == "__main__":
    audio_path = my_path + '/music_data_6params_new/train_audio_10params_v4'
    param_path = my_path + '/music_data_6params_new/train_params_10params_v4'
    engine, generator, parameters, rev_idx = create_synth('toy')
    N = 5  # number of notes
    seg_length = 2
    pad = 0.1

    param_names = ["pitch", "duration", "volume", "filterCutOff", "filterReso", "env_attack", "env_decay", "vibrato_rate", "vibrato_intesity", "vibrato_phase"]

    # Set here the parameter ranges
    rangePitch = [0, 0.16666666, 0.25, 0.41666666, 0.58333333, 0.75, 0.83333333, 1]#! Should be set here and not in generate()
    rangeDuration = [0, 1]
    rangeVolume = [.6, 1]
    rangeFilter = [.3, .6]
    rangeFilter_reso = [.1, .4]   #increasing beyond this causes some distortion in notes
    rangeEnv_attack = [0.2, .4]
    rangeEnv_decay = [0.3, .5]
    rangeVibrato_rate = [0.3, .5]
    rangeVibrato_intensity = [0.3, .7]
    rangeVibrato_phase = [0, .3]

    # rangePitch = [62, 64, 65, 67, 69, 71, 72, 74]
    # rangeVolume = [.7, 1]
    # rangeDuration = [.4, .9]
    # rangeFilter = [50, 100]
    # rangeFilter_reso = [0, 0.4]
    # rangeEnv_attack = [0.15, 0.3]
    # rangeEnv_decay = [0.3, 0.5]
    # sigmoid_pitch = [0, 0.16666666, 0.25, 0.41666666, 0.58333333, 0.75, 0.83333333, 1]
    # sigmoid_r = [0, 1]

    for i in range(400):
        pitch = np.random.choice(rangePitch, N)
        volume = np.random.uniform(rangeVolume[0], rangeVolume[1], size=N)
        duration = np.random.uniform(rangeDuration[0], rangeDuration[1], size=N)
        filterCutOff = np.random.uniform(rangeFilter[0], rangeFilter[1], size=N)
        filterReso = np.random.uniform(rangeFilter_reso[0], rangeFilter_reso[1], size=N)
        env_attack = np.random.uniform(rangeEnv_attack[0], rangeEnv_attack[1], size=N)
        env_decay = np.random.uniform(rangeEnv_decay[0], rangeEnv_decay[1], size=N)
        vibrato_rate = np.random.uniform(rangeVibrato_rate[0], rangeVibrato_rate[1], size=N)
        Vibrato_intensity = np.random.uniform(rangeVibrato_intensity[0], rangeVibrato_intensity[1], size=N)
        vibrato_phase = np.random.uniform(rangeVibrato_phase[0], rangeVibrato_phase[1], size=N)
        # pitch = np.random.choice(sigmoid_pitch, N)
        # # volume = np.random.uniform(rangeVolume[0], rangeVolume[1], size=N)
        # volume = np.random.uniform(sigmoid_r[0], sigmoid_r[1], size=N)
        # duration = np.random.uniform(sigmoid_r[0], sigmoid_r[1], size=N)
        # filterCutOff = np.random.uniform(sigmoid_r[0], sigmoid_r[1], size=N)
        # filterReso = np.random.uniform(rangeFilter_reso[0], rangeFilter_reso[1], size=N)
        # env_attack = np.random.uniform(rangeEnv_attack[0], rangeEnv_attack[1], size=N)
        # env_decay = np.random.uniform(rangeEnv_decay[0], rangeEnv_decay[1], size=N)

        params_diva = [pitch, duration, volume, filterCutOff, filterReso, env_attack, env_decay, vibrato_rate, Vibrato_intensity, vibrato_phase]

        if not os.path.exists(param_path):
            os.makedirs(param_path)
        np.save(param_path + '/' + 'train_params_' + str(i), np.array(params_diva))
        #np.save(param_path + '/' + 'test_params_' + str(i), np.array([pitch, volume, duration, filterCutOff, filterReso, env_attack]))



        #pitch_tr = (np.rint((pitch * 12) + 62)).astype(int)
        #volume_tr = (volume * 0.3) + 0.7
        #duration_tr = (duration * 0.5) + 0.4
        #filterCutOff_tr = (filterCutOff * 50) + 50.0

        #duration_final = seg_length * np.array(duration_tr) / np.sum(duration_tr) - pad
        generate(params_diva, "train_audio_" + str(i), engine, generator, parameters, rev_idx, pad=pad)
        #generate([pitch, volume, duration, filterCutOff, filterReso, env_attack], "test_audio_" + str(i), engine, generator, parameters, rev_idx, pad=pad)






