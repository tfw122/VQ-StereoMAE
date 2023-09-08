# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
import os
from typing import Any, Dict, List
from fileio import FileIOClient, get_current_time_string
import omegaconf
from imports.registry import registry
from trainers.base_trainer import BaseTrainer
from utils import *
from builder import build_callbacks
#from src.utils.logger import TensorboardLogger, setup_output_folder
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from trainers.callbacks import CheckpointEveryNSteps

logger = logging.getLogger(__name__)

@registry.register_trainer("lightning")
class LightningTrainer(BaseTrainer):
    def __init__(self, config, fileio, data_module, model):
        super().__init__()
        self.config = config
        self.user_config = self.config.user_config

        self.fileio = fileio

        self.trainer = None
        self.trainer_config = self.config.trainer.params

        self.log_dir = self.fileio.output_dir

        if os.path.exists(self.log_dir)!=True:
            os.makedirs(self.log_dir)
            print('directory for saving items made: {}'.format(self.log_dir))

        # for saving tensorboard logs;
        self.log_dir_tb = self.log_dir

        print("This version of the code only accomodates saving locally, not on remote cloud servers. For this version contact the maintainer")
        
        # save config and create a directory to save output ckpts;
        self.fileio.save_config(self.config, self.log_dir)

        self.resume_from_checkpoint = None

        self.data_module = data_module

        self.data_module = data_module[0]
        # define your loaders; data_module, train_loder, val_loader, test_loader
        self.train_loader= data_module[1]
        self.val_loader= data_module[2]
        self.test_loader= data_module[3]

        self.model = model

        self.configure_callbacks

        self.load()
        
    def load(self):
        #self.load_datasets()
        self._calculate_max_updates()
        self.load_loggers()
        self.load_checkpoints()
        self.load_callbacks()
        self.load_trainer()

    def load_trainer(self):
        lightning_params = self.trainer_config
        
        with omegaconf.open_dict(lightning_params):
            lightning_params.pop("max_steps")
            lightning_params.pop("max_epochs")
            #lightning_params.pop("resume_from_checkpoint")

        lightning_params_dict = OmegaConf.to_container(lightning_params, resolve=True)
        # max epochs specified in trainer_config;
        self.trainer = Trainer(resume_from_checkpoint=self.resume_from_checkpoint,
                                default_root_dir=self.log_dir,
                                logger= self.tb_writer,
                                callbacks= [self.checkpoint_callback, self.batch_checkpoint_callback], #[[]]+self.callbacks_list][0]
                                **lightning_params_dict,
                            )

    def load_checkpoints(self) -> None:
        iter_save_frequence = self.filter_callback(self.config.training.callbacks)

        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        self.checkpoint_callback = ModelCheckpoint(
            monitor=self.config.model_config.ckpt_monitor,
            save_top_k=-1,
            dirpath=self.log_dir,
            filename="sample-{epoch:03d}-{val_loss:.2f}",
            every_n_epochs=1,
            save_on_train_epoch_end=True,
        )
        self.batch_checkpoint_callback= CheckpointEveryNSteps(self.config)

    def filter_callback(self, callback_list):
        for i in callback_list:
            if i['type']=='CheckpointEveryNSteps':
                return i 
            else:
                pass

    def load_callbacks(self) -> None:
        self.callbacks = build_callbacks(self.config)
        self.callbacks_list=[]
        for callbacks_name, _ in self.callbacks.items():
            #metric_val = self.metrics[metric_name](logits, y)
            self.callbacks_list.append(self.callbacks[callbacks_name])


    def load_loggers(self) -> None:
        # TODO: @sash PL logger upgrade
        if self.user_config.save_locally:
            self.tb_writer = TensorBoardLogger('{}/tensorboard_logs/'.format(self.log_dir_tb), name=self.config.user_config.experiment_name)
        else:
            self.tb_writer = TensorBoardLogger('{}/tensorboard_logs/'.format(self.log_dir), name=self.config.user_config.experiment_name)

    #def load_datasets(self) -> None:
    #    logger.info("Loading Datasets")
    #    self.train_loader = self.data_module.train_dataloader()
    #    self.test_loader = self.data_module.test_dataloader()
    #    self.val_loader = self.data_module.val_dataloader()

    def train(self) -> None:
        logger.info("===== Model =====")
        #logger.info(self.model)
        print_model_parameters(self.model)

        logger.info("Starting training...")            

        if self.config.model_config.load_checkpoint!=None:
            self.prev_checkpoint_path=  self.config.model_config.load_checkpoint
            print('loading previous checkpoint in the trainer: {}'.format(self.prev_checkpoint_path))
            self.trainer.fit(self.model, self.train_loader, self.val_loader, ckpt_path= self.prev_checkpoint_path)
        else:
            print('no checkpoint to be loaded: {}'.format(self.config.model_config.load_checkpoint))
            self.trainer.fit(self.model, self.train_loader, self.val_loader)
        # perform final validation step at the end of training;
        self.val()

        logger.info("Finished training!")

    def val(self) -> None:
        # Don't run if current iteration is divisble by
        # val check interval as it will just be a repeat
        
        logger.info("Stepping into final validation check")
        self.trainer.validate(self.model, self.val_loader)

    def test(self) -> None:
        logger.info("===== Model =====")
        logger.info(self.model)
        print_model_parameters(self.model)

        logger.info("Starting Testing...")            

        if self.config.model_config.load_checkpoint!=None:
            self.prev_checkpoint_path=  self.config.model_config.load_checkpoint
            print('loading previous checkpoint in the trainer: {}'.format(self.prev_checkpoint_path))
            self.trainer.test(self.model, self.test_loader, ckpt_path= self.prev_checkpoint_path)
        else:
            print('no checkpoint to be loaded: {}'.format(self.config.model_config.load_checkpoint))
            print('since None checkpoint path, current ongoing training / randomly initiated model will be used')
            self.trainer.test(self.model, self.test_loader, ckpt_path=None)

        logger.info("Finished testing!")

    def _calculate_max_updates(self) -> None:
        self._max_updates = self.trainer_config.max_steps
        self._max_epochs = self.trainer_config.max_epochs
        if self._max_updates is None and self._max_epochs is None:
            raise ValueError("Neither max_updates nor max_epochs is specified.")

        self._max_updates, max_epochs = get_max_updates(
            self._max_updates,
            self._max_epochs,
            self.train_loader,
            self.trainer_config.accumulate_grad_batches,
        )
        self._max_epochs = math.ceil(max_epochs)
        return self._max_updates

    def configure_device(self) -> None:
        pass

    def configure_seed(self) -> None:
        seed = self.config.training.seed
        seed_everything(seed)