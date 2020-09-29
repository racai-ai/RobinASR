import json
from typing import List

import torch
import time

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.decoder import Decoder
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.utils import load_decoder, load_model
from deepspeech_pytorch.enums import DecoderType


def decode_results(decoded_output: List,
                   decoded_offsets: List,
                   cfg: TranscribeConfig):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "path": cfg.model.model_path
            },
            "language_model": {
                "path": cfg.lm.lm_path
            },
            "decoder": {
                "alpha": cfg.lm.alpha,
                "beta": cfg.lm.beta,
                "type": cfg.lm.decoder_type.value,
            }
        }
    }

    for b in range(len(decoded_output)):
        for pi in range(min(cfg.lm.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if cfg.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


def transcribe(cfg: TranscribeConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(device=device,
                       model_path=cfg.model.model_path,
                       use_half=cfg.model.use_half)

    decoder = load_decoder("beam" if cfg.lm.decoder_type == DecoderType.beam else "greedy",
                           model.labels,
                           cfg.lm.lm_path,
                           cfg.lm.alpha,
                           cfg.lm.beta,
                           cfg.lm.cutoff_top_n,
                           cfg.lm.cutoff_prob,
                           cfg.lm.beam_width,
                           cfg.lm.lm_workers)

    spect_parser = SpectrogramParser(audio_conf=model.audio_conf,
                                     normalize=True)

    start = time.time()
    decoded_output, decoded_offsets = run_transcribe(audio_path=cfg.audio_path,
                                                     spect_parser=spect_parser,
                                                     model=model,
                                                     decoder=decoder,
                                                     device=device,
                                                     use_half=cfg.model.use_half)
    results = decode_results(decoded_output=decoded_output,
                             decoded_offsets=decoded_offsets,
                             cfg=cfg)
    end = time.time()

    print("Time taken: {}".format(end -start))
    print(json.dumps(results, ensure_ascii=False))


def run_transcribe(audio_path: str,
                   spect_parser: SpectrogramParser,
                   model: DeepSpeech,
                   decoder: Decoder,
                   device: torch.device,
                   use_half: bool):
    spect = spect_parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    if use_half:
        spect = spect.half()
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets


def transcribe_online(audio, spect_parser, model, decoder, device, use_half):
    spect = spect_parser.parse_audio_online(audio).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    if use_half:
        spect = spect.half()
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets
