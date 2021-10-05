# MirrorNet : Sensorimotor Interaction Inspired Learning for Audio Synthesizer Controls

This repository hosts code and additional results for the paper "MIRRORNET : SENSORIMOTOR INTERACTION INSPIRED LEARNING FOR AUDIO SYNTHESIZER CONTROLS"

#### Python

Code has been developed with `Python 3.6`. We also use a number of off the shelf libraries which are listed in [`requirements.txt`](requirements.txt). They can be installed with

```bash
$ pip install -r requirements.txt
```

We also trained all the pyTorch models in GPUs and expect anyone who is trying to have CUDA installed. 


#### RenderMan

We use [RenderMan](https://github.com/fedden/RenderMan) library to batch generate audio output from synthesizer presets. For anyone who needs to use the DIVA synthesizer with the MirrorNet, we strongly recommend checking out the documentation for this library. 

