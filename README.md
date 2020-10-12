# RobinASR

This repository contains Robin's Automatic Speech Recognition (RobinASR) for the Romanian language based on the [DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf) architecture, together with a [KenLM](https://kheafield.com/papers/avenue/kenlm.pdf) language model to imporve the transcriptions. 

The pretrained text-to-speech model can be downloaded from [here](http://relate.racai.ro/resources/robinasr/deepspeech_final.pth.gz) and the pretrained KenLM can be downloaded from [here](http://relate.racai.ro/resources/robinasr/corola_5gram.arpa.gz).

Also, make sure to visit:
- A demo of the ASR system available in the RELATE platform: https://relate.racai.ro/index.php?path=robin/asr
- A post-processing web service allowing hyphenation and basic capitalization restoration: https://github.com/racai-ai/RobinASRHyphenationCorrection


## Installation

1) You must have Python 3.6+ and PyTorch 1.5.1+ installed in your system. Also. Cuda 10.1+ is required if you want to use the (recommended) GPU version.

2) Clone the repository and install its dependencies:

```
git clone https://github.com/racai-ai/RobinASR.git
cd RobinASR
pip3 install -r requirements.txt
pip3 install -e .
```

3) Install Nvidia Apex:

```
git clone --recursive https://github.com/NVIDIA/apex.git
cd apex && pip install .
```

4) If you want to use Beam Search and the KenLM language model, you must install CTCDecode:

```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

## Inference Server

Firstly, take a look at the configuration file in `deepspeech_pytorch/configs/inference_config.py` and make sure that the configuration meets your requirements. Then, run the following command:

```
python3 server.py
```

## Acknowledgments

We would like to thank [Sean Narnen](https://github.com/SeanNaren) for making his DeepSpeech2 implementation publicly-available. We used a lot of his code in our implementation. 
