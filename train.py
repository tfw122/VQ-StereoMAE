from utils import *
from builder import *
from fileio import *
from arguments import args
from trainers.lightning_trainer import * 
from datasets.dataset_zoo.mim.stereo_vision_mim_builder import StereoVisionMaskedImageModellingDatasetModule
from models.masked_vision_model import MultiScaleMaskedImageAutoEncoder

from datasets.transforms.vision_transforms import VisionTransforms

def main():
    setup_imports()
    
    parser= args.get_parser()
    opts = parser.parse_args()

    config = build_config(opts)
    print(config.model_config)

    fileio_client = FileIOClient(config)

    #start training:
    if config.model_config.get("load_ckpt_model_only") is not None:
        ckpt_path = config.model_config.load_ckpt_model_only
        
    else:
        ckpt_path = None
        
    print(ckpt_path)
    trainer = build_trainer(config, fileio_client, ckpt_path)
    trainer.train()

if __name__=='__main__':
    main()
