#!/usr/bin/env python3
import sys

import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from nemo.utils import exp_manager, logging
from omegaconf import DictConfig, OmegaConf

try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML


def main(params):

    # init trainer and training config
    trainer = pl.Trainer(**params['trainer'])
    tc = DictConfig(params['model'])

    # init logger
    log_config = exp_manager.ExpManagerConfig(**params['exp_manager'])
    log_config.checkpoint_callback_params.monitor = 'val_wer'
    log_config.checkpoint_callback_params.save_best_model = True
    log_config = OmegaConf.structured(log_config)
    logdir = exp_manager.exp_manager(trainer, log_config)

    model = nemo_asr.models.EncDecCTCModel(cfg=tc, trainer=trainer)

    # start training
    trainer.fit(model)

    print('Training completed!')


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: train_quartznet.py <path_to_training_conf>.yaml')
        sys.exit()
    
    config_path = sys.argv[1]

    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)

    main(params)
