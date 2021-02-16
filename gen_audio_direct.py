# %%
import argparse
import librenderman as rm
import numpy as np
import json, ast
import librosa
import scipy
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


def play_patch(engine, patch_gen, patch=None):
    if patch is None:
        patch = patch_gen.get_random_patch()
    engine.set_patch(patch)
    # Settings to play a note and extract data from the synth.
    midiNote = 60
    midiVelocity = 127  # 100
    noteLength = 4.0
    renderLength = 4.0
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


def synthesize_audio(params, engine, generator, rev_idx):
    # Replace param_defaults with whatever preset to play
    patch = midiname2num(params, rev_idx)
    # patch = midiname2num(params, rev_idx)
    audio, patch = play_patch(engine, generator, patch)
    return audio


def music_synthesize(passed_params, file_name, engine, generator, rev_idx):
    # print(passed_params)
    loaded_params = gen_json_list(passed_params)
    # passed_params = passed_params.tolist()
    print('[Synthesize Music]')
    # final_audio = synthesize_audio(param_defaults, engine, generator, rev_idx)
    final_audio = synthesize_audio(loaded_params, engine, generator, rev_idx)
    # final_audio = final_audio / np.sqrt(np.sum(final_audio ** 2))  # normalize the power of waveform to 1
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
    final_audio = resample(audio, 44100, 4000)
    librosa.output.write_wav(my_path + '/' + file_name + '.wav', final_audio, 4000)

if __name__ == "__main__":
    """
    Sample program, generates default preset
    """
    param_array = [0.11517011,0.12100891,0.42454311,0.30183814,0.38960454,0.39801821]#[0.14839333,0.20469569,0.29110135,0.28382395,0.27051648,0.24941227]
    engine, generator, param_defaults, rev_idx = create_synth('toy')
    # for file in os.listdir("raw_original"):
    #     if file.endswith(".npz"):
    #         loaded = np.load("raw_original/"+file)
    #         loaded_params, loaded_chars, loaded_audio = loaded["param"], loaded["chars"], loaded["audio"]
    music_synthesize(param_array, 'syn_audio', engine, generator, rev_idx)



    # os.chdir("/raw")
    # for file in glob.glob("*.npz"):
    #     # print(file)
    #     loaded = np.load(file)
    #     loaded_params, loaded_chars, loaded_audio = loaded["param"], loaded["chars"], loaded["audio"]
    # music_synthesize(loaded_params,file)

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























# # import librosa
# # import scipy
# import numpy as np
#
# def resample(y, orig_sr, target_sr):
#     if orig_sr == target_sr:
#         return y
#     # ratio = float(target_sr) / orig_sr
#     # n_samples = int(np.ceil(y.shape[-1] * ratio))
#     # y_hat = scipy.signal.resample(y, n_samples, axis=-1)  # maybe resampy is better?
#     # y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)
#     return np.ascontiguousarray(y_hat, dtype=y.dtype)
#
# loaded = np.load('raw/0a1fa34a01aa41b6d380216df012d458_60_100.npz')
# # print(loaded)
# # print(loaded['param'])
# loaded_params, loaded_chars, loaded_audio = loaded["param"], loaded["chars"], loaded["audio"]
#
# # samplerate = 22050
# # write("example.wav", samplerate, loaded_audio)
# final_audio = resample(loaded_audio, 44100, 22050)
# # librosa.output.write_wav("synthesized_audio" + '.wav', final_audio, 22050)
# #
