#!/usr/bin/env python3
import sys

import nemo.collections.asr as nemo_asr


def main(artefact_path):

    # load model
    model = nemo_asr.models.EncDecCTCModel.restore_from(
        artefact_path
    )

    # run testing inference
    print(model.transcribe(
        paths2audio_files=[
            'test-example/209c6fb213.wav',
            'test-example/36687b45a7.wav',
            'test-example/c1d8ef6242.wav',
            'test-example/c244c3dc1b.wav'
        ],
        batch_size=4)
    )
    print('Success!')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 infer_nemo.py <path_to_model>.nemo')
        sys.exit()

    artefact_path = sys.argv[1]
    main(artefact_path)
