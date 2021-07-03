
## Environment installation

* Install pre-requisites
```bash
apt-get update && apt-get upgrade
apt-get install sox libsndfile1 ffmpeg
```

* Install requirements
```bash
# using pip
python3 -m pip install -r requirements

# using conda
conda env create -f environment.yml
```


## Train model
* Install environment as shown above

* Go to `scripts/` and prepare train/val/test datasets manifests, noise datasets manifests (see [readme](scripts/readme.md))

* Configure `conf/quartznet_5x1_aug.yaml` with generated manifests paths, experiments folder and your desired training params

* Train the model
```bash
python3 train_quartznet.py conf/quartznet_5x1_aug.yaml
```

* To view training logs
```bash
tensorboard --logdir <exp_path>
```


## Experiments and results

|                                 | Model params, M | Est. fp32 model size, Mb |   Four testing sentences decoding output   |
|:-------------------------------:|:---------------:|:------------------------:|:------------------------------------------:|
|        quartznet5x3.yaml        |       4.6       |           18.49          |       ['89786', '33\*6', '35', '183']       |
|     quartznet5x3_nvidia.yaml    |       6.4       |           25.56          |    ['89\*86', '33\*760', '39\*75', '18\*0']    |
|        quartznet_5x1.yaml       |       2.0       |           7.97           |    ['897\*688', '332\*6', '309\*', '183\*0']   |
|      quartznet_5x1_aug.yaml     |       2.0       |           7.97           |    ['897\*8', '32\*7', '309\*75', '183\*0']    |
| quartznet_5x1_aug.yaml (tuned*) |       2.0       |           7.97           | ['897\*868', '332\*6', '309\*755', '183\*003'] |
|       oracle transcription      |        -        |             -            |  ['896867', '332763', '309758', '183037']  |

* tuned - expanded val/test sets, limited minimal lr for scheduler, trained for more epochs, higher probs of augmentations. See `conf/quartznet_5x1_aug.yaml`


## Quantization

* Best performed fp32 model (7.97 Mb) could be compressed 4x times to int8 (2Mb) using dynamic range quantization. Please refer to [readme](quantization/readme.md) for the details


## Evaluating the model

* For model evaluation details please see [readme](evaluation/readme.md)


## Augmentations used
For full description please see `conf/quartznet_5x1_aug.yaml`

* `speed` - speed perturbation (changes voice of a speaker)

* `impulse` (2015 files 1-second length):
    * [BUT_RIRs](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database) - real room impulse responses

* `noise` (10120 files 2-seconds length):
    * [DCASE](http://dcase.community/challenge2017/task-sound-event-detection-in-real-life-audio) - audio events dataset
    * [DEMAND](https://zenodo.org/record/1227121#.YOCiOzpRVH4) - db of acoustic noises

* `gain` - volume perturbation

* `white_noise`

* `transcode_aug` - codecs augmentation 


## TODO:
- [ ] Macch fp32 and int8 models outputs


## References used
* https://developer.nvidia.com/blog/develop-smaller-speech-recognition-models-with-nvidias-nemo-framework/
* https://github.com/NVIDIA/NeMo
* https://www.onnxruntime.ai/docs/how-to/quantization.html


## Acks
* Done in 24h