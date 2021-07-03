## Description

* `convert_train_meta_to_manifest.py` - creates a training data manifest required for model training

* `split_manifest.sh` - splits train file into local train/val/test splits

* `noise_folder_to_manifest.py` - creates `noise` augmentations and `impulse` manifests required for realtime augs (in separate runs)


## Pipeline

* Please run mentioned above scripts in specified order to prepare training data and noises. Resutling trainig manifest example:
```
{"audio_filepath": "data/numbers/train/af53de7964.wav", "duration": 2.372708333333333, "text": "613*903"}
{"audio_filepath": "data/numbers/train/cb211eb75d.wav", "duration": 3.170916666666667, "text": "872*960"}
...
```

* Note, that `noise_folder_to_manifest.py` would require you to load, unpack and convert to wav 16k noise datasets (see resources in [README.md](../README.md#augmentations-used)). Prepare one manifest for `noise` and one for `impulse` classes. Only wav files are required, no specific labeling requirements. `impulse` manifest example:
```
{"audio_filepath": "/home/anton/data/noise/real_rirs/default/ConferenceRoom2-0.wav", "duration": 1.0, "label": "impulse", "text": "_", "offset": 0}
{"audio_filepath": "/home/anton/data/noise/real_rirs/default/ConferenceRoom2-1.wav", "duration": 1.0, "label": "impulse", "text": "_", "offset": 0}
...
```

