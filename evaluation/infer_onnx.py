#!/usr/bin/env python3
import sys

import nemo.collections.asr as nemo_asr
import numpy as np
import onnx
import onnxruntime
import soundfile as sf
import torch
from omegaconf import DictConfig
from ruamel.yaml import YAML


# Define vocabulary params
vocab = {item: str(item) for item in range(10)}
vocab[10] = '*'
blank_id = 11


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def infer(model_path, audio_path, config_params):

    global vocab, blank_id

    # Input signal
    x, sr = sf.read(audio_path)
    x = torch.from_numpy(x).to(torch.float)
    x = x.unsqueeze(0)

    # extract features
    model_params = DictConfig(config_params['model'])
    preprocessor = nemo_asr.models.EncDecCTCModel.from_config_dict(model_params.preprocessor)
    feats, feats_length = preprocessor(
        input_signal=x, length=torch.Tensor([x.shape[1]]).to(torch.int),
    )

    ort_session = onnxruntime.InferenceSession(model_path)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(feats)}
    log_probs = ort_session.run(None, ort_inputs)[0]  # logits
    log_probs = torch.from_numpy(log_probs)
    
    # get greedy predictions
    greedy_prediction = log_probs.argmax(dim=-1, keepdim=False).detach().numpy().tolist()[0]

    # CTC-decoding
    decoded_prediction = []
    previous = blank_id
    for p in greedy_prediction:
        if (p != previous or previous == blank_id) and p != blank_id:
            decoded_prediction.append(p)
        previous = p

    text = ''.join([vocab[token] for token in decoded_prediction])
    print(text)


def main(model_path, config_params):

    filenames = [
        'test-example/209c6fb213.wav',
        'test-example/36687b45a7.wav',
        'test-example/c1d8ef6242.wav',
        'test-example/c244c3dc1b.wav'
    ]

    for filename in filenames:
        infer(model_path, filename, config_params)
    
    print('Success!')


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print('Usage: python3 infer_onnx.py <onnx_model_path>.{onnx|quant.onnx} <model_config>.yaml')
        sys.exit()

    model_path = sys.argv[1]
    model_params_config_path = sys.argv[2]

    yaml = YAML(typ='safe')
    with open(model_params_config_path) as f:
        config_params = yaml.load(f)

    main(model_path, config_params)
