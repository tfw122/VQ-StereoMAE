# from: https://github.com/tractableai/resnet-sandbox/blob/977550df22b4a520a75de389329f28fdb43aaaff/utils/aws_client.py#L123
import tempfile 
import os
import logging
import time 
from datetime import datetime, timedelta
from omegaconf import OmegaConf


log = logging.getLogger(__name__)

# File IO
class FileIOClient():
    def __init__(self, config):
        self.config = config
        print("THIS VERSION OF THE CODE SUPPORTS LOCAL FILE I/O ONLY! FOR CLOUD CLIENTS CONTACT THE MAINTAINER")
        output_dir_frozen = self.create_output_save_path(self.config)
        self.output_dir = output_dir_frozen

    def save_config(self, config, local_path=None):
        if local_path==None:
            with tempfile.NamedTemporaryFile(mode='w+b') as f:
                OmegaConf.save(config=config, f=f.name)
        else:
            # OmegaConf.save can also accept a `str` or `pathlib.Path` instance:
            OmegaConf.save(config, "{}/config.yaml".format(local_path))
            print("config saved to: {}".format(local_path))

    def create_output_save_path(self, config):
        root_dir = config.user_config.save_root_dir
        username_prefix = config.user_config.username_prefix
        task_type = config.user_config.task_type
        experiment_name = config.user_config.experiment_name
        timestamp = get_current_time_string()
        #Outputs to s3://s3bucket_name/{s3key_prefix}/{task_type}/{experiment_name}/{timestamp}/
        output_dir = '{}/{}/{}/{}/{}/train_outputs'.format(root_dir, 
                                            username_prefix,
                                            task_type,
                                            experiment_name,
                                            timestamp)
        
        
        if os.path.exists(output_dir)!=True:
            os.makedirs(output_dir)
            print('output dir for saving files: "{}" created!'.format(output_dir))
        else:
            print('{} path already exists'.format(output_dir))

        return output_dir

# ----------------------------------------------- AWS utils -----------------------------------------------
def get_current_time_string():
    t = time.time()
    t = datetime.fromtimestamp(int(t))
    t = t + timedelta(hours=1)
    return datetime.strftime(t, "%y%m%d-%H%M%S")