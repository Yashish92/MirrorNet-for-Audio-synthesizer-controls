# %%

# Author: Yashish Maduwantha 

import argparse
import librenderman as rm
import numpy as np
import json, ast
import librosa
import scipy
import os

my_path = os.path.dirname(os.path.abspath(__file__))

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
    #final_audio = final_audio / np.sqrt(np.sum(np.array(final_audio) ** 2))
    #final_audio = (np.iinfo(np.int16).max * final_audio / np.max(final_audio)).astype(np.int16)
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
    # audio = audio / np.sqrt(np.sum(np.array(audio) ** 2))
    scipy.io.wavfile.write(my_path + '/music_data_6params_new/test_audio_new/' + file_name + '.wav', 44100,
                           (np.iinfo(np.int16).max * audio / np.max(audio)).astype(np.int16))


## changed function template by adding drive_params argument
def generate(params, engine, generator, parameters, rev_idx, pad=0, drive_params=5):
    '''
    params: midiPitch, MidiValocity, midiDuration, CutOff Frequency
    '''
    params_dic = {}
    #diva_parameters = ["VCF1: Frequency","OSC: Volume2","VCF1: FreqMod2Depth","LFO1: Rate", "OSC: Vibrato", "LFO1: Phase", "LFO2: Rate", "LFO2: Phase", "VCF1: Resonance", "ENV1: Attack"]
    #diva_parameters = ["VCF1: Frequency","OSC: Volume2", "LFO1: Rate", "OSC: Vibrato", "LFO1: Phase", "VCF1: Resonance", "ENV1: Attack"]
    #diva_parameters = ["VCF1: Frequency","OSC: Volume2", "VCF1: Resonance", "ENV1: Attack"]

    if drive_params == 5:
        diva_parameters = ["VCF1: Frequency","OSC: Volume2", "VCF1: Resonance"]
        my_params = {
          # "OSC: Volume2": 1,
          "OSC: Volume3": 0,
          "OSC: Vibrato": 0,
          "ENV1: Sustain": 0,  # changed from 0 as initially suggested
          "ENV1: Decay": 0.4,
          "main: Output": 1,
          "OSC: Volume1": 0,
          "OSC: NoiseVol": 0,
          "OSC: Noise1On": 0,
          "VCF1: Feedback": 0,
          "OSC: NoiseVolModSrc": 0,
          "ENV1: Attack": 0,
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
          "VCF1: FreqModDepth": 0.5,
          "VCF1: FreqMod2Depth": 0.5,
          "VCF1: FreqModSrc": .5,
          "VCF1: FilterFM": .5,
          #"VCF1: Resonance": 0,                  # uncomment if filter resonance not used as an encoder out
          "ENV1: Velocity": 1,
          "ENV2: Velocity": 1,
          "VCF1: SvfMode": 1,
          "VCF1: Model": 0.5
        }

    elif drive_params == 6:
        diva_parameters = ["VCF1: Frequency","OSC: Volume2", "VCF1: Resonance", "ENV1: Attack"]
        my_params = {
          # "OSC: Volume2": 1,
          "OSC: Volume3": 0,
          "OSC: Vibrato": 0,
          "ENV1: Sustain": 0,
          "ENV1: Decay": 0.4,
          "main: Output": 1,
          "OSC: Volume1": 0,
          "OSC: NoiseVol": 0,
          "OSC: Noise1On": 0,
          "VCF1: Feedback":0,
          "OSC: NoiseVolModSrc": 0,
          #"ENV1: Attack": 0,
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
          "VCF1: FreqModDepth":0.5,
          "VCF1: FreqMod2Depth": 0.5,
          "VCF1: FreqModSrc": .5,
          "VCF1: FilterFM": .5,
          "ENV1: Velocity": 1,
          "ENV2: Velocity": 1,
          "VCF1: SvfMode": 1,
          "VCF1: Model": 0.5
         # "LFO1: DepthMod Dpt1": 0,
          #"LFO1: Sync": 0,
          #"LFO1: Waveform": 0,
        }

    elif drive_params ==7:
        diva_parameters = ["OSC: Volume2", "VCF1: Frequency", "VCF1: Resonance", "ENV1: Attack", "ENV1: Decay"]
        # diva_parameters = ["VCF1: Frequency", "OSC: Volume2", "VCF1: Resonance", "ENV1: Attack" ]
        my_params = {
            # "OSC: Volume2": 1,
            "OSC: Volume3": 0,
            "OSC: Vibrato": 0,
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
            # "LFO1: DepthMod Dpt1": 0,
            # "LFO1: Sync": 0,
            # "LFO1: Waveform": 0,
        }

        melody_dur = 2
        midiNote = (params[0] * 12) + 62
        midiNote = np.rint(midiNote).astype(int)
        # volume2  = (params[1]*0.3) + 0.7
        # midiDuration = 2.1 * params[2] / np.sum(params[2])
        midi_dur = (params[1] * 0.5) + 0.4
        midiDuration = melody_dur * midi_dur / np.sum(midi_dur) - pad
        # mean_freqmod = 0.5
        # std_freqmod = 0.07

        i = 2  # change i accordingly to set the remaining parameters after setting first 4 parameters
        for p in diva_parameters:
            # if p == "VCF1: Frequency":
            #     params_dic[p] = ((params[2] * 25) + 55) / 150.0  # changed filter range from 55 to 80
            # elif p == "OSC: Volume2":
            #     params_dic[p] = (params[3]*0.3) + 0.7
            # elif p == "VCF1: FreqMod2Depth":
            # m_data = np.mean(params[4])
            # std_data = np.std(params[4])
            # trans_data = mean_freqmod + (params[4]-m_data)*(std_freqmod/std_data)
            #    params_dic[p] = params[4]
            # elif p == "LFO1: Rate":
            #   params_dic[p] = (params[4]*0.4)
            params_dic[p] = params[i]  # setting all parameters, be careful with their order btw functions!!!!
            i += 1

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
            music_synthesize(parameters, engine, generator, rev_idx, midiNote[i], 127, noteLength=midiDuration[i], renderLength=midiDuration[i] + pad))

    # write_to_wav(audio, outName)
    final_audio = np.array(audio)
    return final_audio


if __name__ == "__main__":
    """
    Sample program, generates default preset
    """
    param_array = [0.5, 0.5, 0, 0, 0, 0.01, 1, 1, 1, 0.5, 0, 0.58333333, 0.5, 0.5, 0.5, 0.5]
    music_synthesize(param_array, sampling_rate=22050)
    # # Define arguments
    # parser = argparse.ArgumentParser()
    # # Data arguments
    # parser.add_argument('--path', type=str, default='/home/yashish/Music_Synthesizer/diva_raw/raw/', help='')
    # parser.add_argument('--output', type=str, default='outputs', help='')
    # parser.add_argument('--dataset', type=str, default='toy', help='')
    # parser.add_argument('--data', type=str, default='mel', help='')
    # args = parser.parse_args()
    # print('[Load the dataset]')
    # # Take fixed batch
    # loaded = np.load('Test_batch/0a1fa34a01aa41b6d380216df012d458_60_100.npz')
    # # print(loaded)
    # # print(loaded['param'])
    # loaded_params, loaded_chars, loaded_audio = loaded["param"], loaded["chars"], loaded["audio"]
    # # dic_params = dict(np.ndenumerate(loaded_params))
    # # print(loaded_params)
    # # print(fixed_params)
    # print('[Create synth rendering]')
    # final_params = ['ENV1: Decay', 'VCF1: FilterFM', 'OSC: Vibrato', 'OSC: FM',
    #                 'VCF1: Feedback', 'ENV1: Attack', 'ENV1: Sustain',
    #                 'OSC: Volume3', 'OSC: Volume2', 'OSC: OscMix',
    #                 'VCF1: Resonance', 'VCF1: Frequency', 'OSC: Tune3',
    #                 'OSC: Tune2', 'OSC: Shape1', 'OSC: Shape2']
    # # Create synth rendering system
    # engine, generator, param_defaults, rev_idx = create_synth('toy')
    # print('[Synthesize batch]')
    # # final_audio = synthesize_audio(param_defaults, engine, generator, rev_idx)
    # final_audio = synthesize_audio(loaded_params.tolist(), engine, generator, rev_idx)
    # final_audio = resample(final_audio, 44100, 22050)
    #
    # librosa.output.write_wav("synthesized_audio" + '.wav', final_audio, 22050)
    # # Generate the test batch for comparison
    # # audio = synthesize_batch(fixed_params, final_params, engine, generator, param_defaults, rev_idx, orig_wave=fixed_audio, name='check')


