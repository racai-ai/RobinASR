import logging
import os
from tempfile import NamedTemporaryFile

import hydra
import torch
from flask import Flask, request, jsonify
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.inference_config import ServerConfig
from deepspeech_pytorch.inference import run_transcribe
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.utils import load_model, load_decoder
from waitress import serve
from deepspeech_pytorch.enums import DecoderType

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

ALLOWED_EXTENSIONS = {'.wav'}

cs = ConfigStore.instance()
cs.store(name="config", node=ServerConfig)


@app.route('/transcribe', methods=['POST'])
def transcribe_file():
    if request.method == 'POST':
        try:
            res = {}
            if 'file' not in request.files:
                res['status'] = "error"
                res['message'] = "audio file should be passed for the transcription"
                return jsonify(res)
            file = request.files['file']
            filename = file.filename
            _, file_extension = os.path.splitext(filename)
            if file_extension.lower() not in ALLOWED_EXTENSIONS:
                res['status'] = "error"
                res['message'] = "{} is not supported format.".format(file_extension)
                return jsonify(res)
            with NamedTemporaryFile(suffix=file_extension) as tmp_saved_audio_file:
                file.save(tmp_saved_audio_file.name)
                logging.info('Transcribing file...')
                transcription, _ = run_transcribe(audio_path=tmp_saved_audio_file,
                                                  spect_parser=spect_parser,
                                                  model=model,
                                                  decoder=decoder,
                                                  device=device,
                                                  use_half=config.model.use_half)
                logging.info('File transcribed')
                res['status'] = "OK"
                res['transcription'] = transcription[0][0]
                return jsonify(res)
        except Exception as e:
            logging.error(e)
            res['status'] = "error"
            res['message'] = "The server encountered an internal error and was unable to complete your request."
            return jsonify(res)


@hydra.main(config_name="config")
def main(cfg: ServerConfig):
    global model, spect_parser, decoder, config, device
    config = cfg
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info('Setting up server...')
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

    spect_parser = SpectrogramParser(model.audio_conf, normalize=True)
    logging.info('Server initialised')

    serve(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
