# RobinASR

This repository contains Robin's Automatic Speech Recognition (RobinASR) for the Romanian language based on the [DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf) architecture, together with a [KenLM](https://kheafield.com/papers/avenue/kenlm.pdf) language model to imporve the transcriptions. 

The pretrained text-to-speech model can be downloaded from [here](https://relate.racai.ro/resources/robinasr/deepspeech_final.pth.gz) and the pretrained KenLM can be downloaded from [here](https://relate.racai.ro/resources/robinasr/corola_5gram.arpa.gz).

Also, make sure to visit:
- A demo of the ASR system available in the RELATE platform: https://relate.racai.ro/index.php?path=robin/asr
- A post-processing web service allowing hyphenation and basic capitalization restoration: https://github.com/racai-ai/RobinASRHyphenationCorrection


## Installation

### Docker

We offer two docker containers that are available on dockerhub and that provide the RobinASR out of the box:
- for running on GPU: 
```
docker pull racai/robinasr:gpu
docker run --gpus all -p 8888:8888 --net=host --ipc=host racai/robinasr:gpu
```
- for running on CPU:
```
docker pull racai/robinasr:cpu
docker run -p 8888:8888 --net=host --ipc=host racai/robinasr:cpu
```

You can also create your own docker image by following these steps:

1) Download the pretrained text-to-speech model and the pretrained KenLM at the above links, and copy them in a `models` directory inside this repository.

2) Build the docker image using the `Dockerfile`. Make sure that `deepspeech_pytorch/configs/inference_config.py` has the desired configuration.

```
docker build --tag RobinASR .
```

3) Run the docker image.

```
docker run --gpus all -p 8888:8888 --net=host --ipc=host RobinASR
```

### From Source

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

## Train a New Model

You must create 3 csv manifest files (train, valid and test) that contain on each line the the path to a wav file and the path to its corresponding transcription, separated by commas:

```
path_to_wav1,path_to_txt1
path_to_wav2,path_to_txt2
path_to_wav3,path_to_txt3
...
```

Then you must modify correspondingly with your configuration the file located at `deepspeech_pytorch/configs/train_config.py` and start training with:
```
python train.py
```

## Acknowledgments

We would like to thank [Sean Narnen](https://github.com/SeanNaren) for making his DeepSpeech2 implementation publicly-available. We used a lot of his code in our implementation. 

## Cite

If you are using this repository, please cite the following [paper](https://academiaromana.ro/sectii2002/proceedings/doc2020-4/11-Avram_Tufis.pdf) as a thank you to the authors:

```
Avram, A.M., Păiș, V. and Tufis, D., 2020, October. Towards a Romanian end-to-end automatic speech recognition based on Deepspeech2. In Proc. Rom. Acad. Ser. A (Vol. 21, pp. 395-402).
```

or in BibTeX format:

```
@inproceedings{avram2020towards,
  title={Towards a Romanian end-to-end automatic speech recognition based on Deepspeech2},
  author={Avram, Andrei-Marius and Păiș, Vasile and Tufiș, Dan},
  booktitle={Proceedings of the Romanian Academy, Series A},
  pages={395--402},
  year={2020}
}
```
