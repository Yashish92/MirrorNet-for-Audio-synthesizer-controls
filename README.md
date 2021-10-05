# MirrorNet : Sensorimotor Interaction Inspired Learning for Audio Synthesizer Controls

This repository hosts code and additional results for the paper "MIRRORNET : SENSORIMOTOR INTERACTION INSPIRED LEARNING FOR AUDIO SYNTHESIZER CONTROLS"

![Model architecture](model_archi_v3.png)

## Webpage

For more results on the project, please **visit the corresponding [supporting website](Yashish92.github.io)**.

#### Python

Code has been developed with `Python 3.6`. We also use a number of off the shelf libraries which are listed in [`requirements.txt`](requirements.txt). They can be installed with

```bash
$ pip install -r requirements.txt
```

We also trained all the pyTorch models in GPUs and expect anyone who is trying to have CUDA installed. 

#### Generating dataset

The DIVA melodies needed for the Exp1 and Exp2 can be generated by using [`gen_audio_direct_new_DIVA.py`](gen_audio_direct_new_DIVA.py). The final train and test sets with auditory spectrograms can be generated with [`data_process_audspec_7params_bpass_filter.py`](data_process_audspec_7params_bpass_filter.py) by passing the folders with melodies and ground truth parameters as two arguments. 

#### RenderMan

We use [RenderMan](https://github.com/fedden/RenderMan) library to batch generate audio output from synthesizer presets. For anyone who needs to use the DIVA synthesizer with the MirrorNet, we strongly recommend checking out the documentation for this library. 

