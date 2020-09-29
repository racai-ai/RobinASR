import pyaudio
import webrtcvad
import argparse
import json
import os
import numpy as np
import librosa
from scipy.io import wavfile

import torch

from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.inference import transcribe_online
from deepspeech_pytorch.opts import add_decoder_args, add_inference_args
from deepspeech_pytorch.utils import load_model, load_decoder


def decode_results(decoded_output, decoded_offsets):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(args.model_path)
            },
            "language_model": {
                "name": os.path.basename(args.lm_path) if args.lm_path else None,
            },
            "decoder": {
                "lm": args.lm_path is not None,
                "alpha": args.alpha if args.lm_path is not None else None,
                "beta": args.beta if args.lm_path is not None else None,
                "type": args.decoder,
            }
        }
    }

    for b in range(len(decoded_output)):
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if args.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


def do_transcribe(frames):
    ss = b''.join(frames)
    wave = []
    for i in range(0, len(ss), 2):
        wave.append(int.from_bytes(ss[i:i + 2], "little", signed=True) / 32767)

    audio = np.array(wave, dtype=np.float32)
    
    wavfile.write("output.wav", sample_rate, (audio * 32767).astype(np.int16))

    decoded_output, decoded_offsets = transcribe_online(audio=audio,
                                                        spect_parser=spect_parser,
                                                        model=model,
                                                        decoder=decoder,
                                                        device=device,
                                                        use_half=args.half)

    transcription = decode_results(decoded_output, decoded_offsets)["output"][0]["transcription"]

    print(transcription)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    arg_parser = add_inference_args(arg_parser)
    arg_parser.add_argument('--offsets',
                            dest='offsets',
                            action='store_true',
                            help='Returns time offset information')
    arg_parser = add_decoder_args(arg_parser)
    args = arg_parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.half)

    decoder = load_decoder(decoder_type=args.decoder,
                           labels=model.labels,
                           lm_path=args.lm_path,
                           alpha=args.alpha,
                           beta=args.beta,
                           cutoff_top_n=args.cutoff_top_n,
                           cutoff_prob=args.cutoff_prob,
                           beam_width=args.beam_width,
                           lm_workers=args.lm_workers)

    spect_parser = SpectrogramParser(audio_conf=model.audio_conf,
                                     normalize=True)

    vad = webrtcvad.Vad()
    vad.set_mode(3)

    chunk = 320
    data_format = pyaudio.paInt16
    channels = 1
    sample_rate = 16000
    record_seconds = 10000

    p = pyaudio.PyAudio()

    stream = p.open(format=data_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    frames = []
    false_count = 0

    print("Start speaking...")

    for i in range(0, int(sample_rate / chunk * record_seconds)):
        frame = stream.read(chunk)
        

        if vad.is_speech(frame, sample_rate):
            frames.append(frame)
            false_count = 0
        elif len(frames) > 40 and false_count < 25:
            frames.append(frame)
            false_count += 1
        elif len(frames) > 40:
            do_transcribe(frames)
            frames = frames[-10:]
        else:
            false_count += 1

            if false_count > 10 and len(frames) < 20:
                frames = frames[-10:]
                
        # print(false_count, len(frames))
