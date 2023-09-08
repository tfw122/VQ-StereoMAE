import logging
import warnings
from typing import List 
import collections
from ast import literal_eval
from imports.registry import registry
import pytorch_lightning as pl
import torch 
import torchmetrics
import torch.nn as nn
from utils import *
from imports.registry import registry
from omegaconf import DictConfig, OmegaConf, errors as OCErrors
import fairscale.optim.oss as OSS
logger = logging.getLogger(__name__)

def build_config(args):

    # build default trainer config;
    default_config = load_yaml(args.default_config_path)
    # add opts so that some of the overrides for the defaults
    # from command line required for setup can be honored
    default_config = _merge_with_dotlist(
        default_config, args.opts, skip_missing=True, log_info=False
    )
    model_config = load_yaml(args.model_config_path)
    dataset_config = load_yaml(args.dataset_config_path)
    user_config = load_yaml(args.user_config_path)

    config = OmegaConf.merge(default_config, model_config, dataset_config, user_config)
    
    # The above function only merges and overrides the default config; 
    # the following now allows you to change any other configs too; i.e. model config, dataset config etc.
    config = _merge_with_dotlist(config, args.opts)
    _update_specific(config)
    _upgrade_config(config)

    return config


def build_optimizer(model_parameters, config):
    optimizer_config = config.optimizer
    if "type" not in optimizer_config:
        raise ValueError(
            "Optimizer attributes must have a 'type' key "
            "specifying the type of optimizer. "
            "(Custom or PyTorch, e.g. 'adam_w' or 'SGD')"
        )
    optimizer_type = optimizer_config.type

    if "params" not in optimizer_config:
        warnings.warn("optimizer attributes has no params defined, defaulting to {}.")

    params = optimizer_config.get("params", {})

    # get pytorch default optimizer;
    if hasattr(torch.optim, optimizer_type):
        optimizer_class = getattr(torch.optim, optimizer_type)
    # get custom optimizer via the registry;
    else:
        optimizer_class = registry.get_optimizer_class(optimizer_type)
        if optimizer_class is None:
            raise ValueError(
                "No optimizer class of type {} present in "
                "either torch or registered to registry"
            )

    # load model parameters to optimise;
    if optimizer_config.get("enable_state_sharding", False):
        # TODO(vedanuj): Remove once OSS is moved to PT upstream
        try:
            from fairscale.optim.oss import OSS
        except ImportError:
            print(
                "Optimizer state sharding requires fairscale. "
                + "Install using pip install fairscale."
            )
            raise

        assert (
            dist.is_initialized()
        ), "Optimizer state sharding can only be used in distributed mode."

        is_fp16 = config.get("training", {}).get("fp16", False)
        optimizer = OSS(
            params=model_parameters, optim=optimizer_class, broadcast_fp16=is_fp16, **params
        )
    else:
        if params!=None:
            optimizer = optimizer_class(model_parameters, **params)
        else:
            # initiate using default settings:
            optimizer = optimizer_class(model_parameters)
    return optimizer

def build_scheduler(optimizer, config):
    scheduler_config = config.get("scheduler", {})

    if "type" not in scheduler_config:
        warnings.warn(
            "No type for scheduler specified even though lr_scheduler is True, "
            "specify the type of scheduler in config. "
            "(Custom or PyTorch, e.g. 'step' or 'ReduceLROnPlateau')"
        )
    scheduler_type = scheduler_config.type

    if "params" not in scheduler_config:
        warnings.warn("scheduler attributes has no params defined, defaulting to {}.")
    
    params = scheduler_config.get("params", {})
    # get pytorch default optimizer;
    if scheduler_type!=None:
        if hasattr(torch.optim.lr_scheduler, scheduler_type):
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
        # get custom optimizer via the registry;
        else:
            scheduler_class = registry.get_scheduler_class(scheduler_type)
            if scheduler_class is None:
                raise ValueError(
                    "No scheduler class of type {} present in "
                    "either torch or registered to registry"
                )
        scheduler = scheduler_class(optimizer, **params)
    else:
        scheduler = None
        warnings.warn("no scheduler loaded, since scheduler type in config is {}.".format(scheduler_type))

    return scheduler

def build_dataset(config):
    try:
        dataset_key= config.dataset_config.dataset_mode.dataset_classes
    except:
        dataset_key= config.dataset_config.dataset_classes
        
    dataset_classes={}

    for dataset_ in dataset_key:
        dataset_class = registry.get_dataset_class(dataset_)
        if dataset_class is None:
            raise RuntimeError(f"No dataset registered for name: {dataset_}")
        dataset_classes["{}".format(dataset_)]=dataset_class

    return dataset_classes
    # builder_instance: pl.LightningDataModule = dataset_builder(config)
    # return builder_instance

def build_datamodule(config) -> pl.LightningDataModule:
    print(config)
    print(config.dataset_config.dataset_builder)
    dataset_key= config.dataset_config.dataset_builder
    dataset_builder = registry.get_builder_class(dataset_key)
    
    assert dataset_builder, (
        f"Key {dataset_key} doesn't have a registered " + "dataset builder"
    )
    builder_instance: pl.LightningDataModule = dataset_builder(config)
    return builder_instance

def build_model(config, ckpt_path=None):
    model_name = config.model_config.name
    model_class = registry.get_model_class(model_name)
    if model_class is None:
        raise RuntimeError(f"No model registered for name: {model_name}")

    if ckpt_path==None:
        model = model_class(config=config)
    else:
        model = model_class.load_from_checkpoint(checkpoint_path=ckpt_path, config=config)
        print('model ckpt weights loaded from: {}'.format(ckpt_path))

    return model

def build_loss(config):
    loss_list = config.model_config.loss
    #loss_name = OmegaConf.to_object(loss_name)
    losses= nn.ModuleDict()

    for loss_ in loss_list:
        loss_name = loss_['type']
        # get pytorch default loss function;
        if hasattr(torch.nn, loss_name):
            loss_class = getattr(torch.nn, loss_name)
            # any specific args;
            if "params" in loss_:
                params = loss_['params']
                loss_fnc = loss_class(**params)
            else:
                # initiate loss with default args
                loss_fnc = loss_class()
        # get custom loss function via the registry;
        else:
            loss_class = registry.get_loss_class(loss_)
            if loss_class is None:
                raise RuntimeError(f"No loss registered for name: {loss_}")
            loss_fnc = loss_class(config=config)

        losses[loss_['type']]=loss_fnc

    return losses

def build_trainer(config, fileio_client, ckpt_path=None):
    """Builder function for creating a trainer class. Trainer class name
    is picked from the config.
    Args:
        config (DictConfig): Configuration that will be used to create
            the trainer.
        fileio_client: the fileio class that has info about file storage;
    Returns:
        (BaseTrainer): A trainer instance
    """
    trainer_type = config.training.trainer
    trainer_cls = registry.get_trainer_class(trainer_type)
    data_builder = build_datamodule(config)
    tokenizer = get_visual_tokenizer(config)
    model = build_model(config, ckpt_path)
    # get the dataloaders;
    dataset_loaders= load_datasets(data_builder)
    # initiate trainer;
    trainer_obj = trainer_cls(config, fileio_client, dataset_loaders, model, tokenizer)

    return trainer_obj

def load_datasets(data_module) -> None:
    print("Loading Datasets")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    return data_module, train_loader, val_loader, test_loader

def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = get_model(
            args.tokenizer_model,
            pretrained=True,
            pretrained_weight=args.tokenizer_weight,
            as_tokenzer=True,
            n_code=args.codebook_size, 
            code_dim=args.codebook_dim,
        ).eval()
    return model

def build_callbacks(config):
    callbacks= {}
    callback_list = config.training.callbacks

    for callback_ in callback_list:
        callback_name = callback_['type']
        # get pytorch_lightning default callback function;
        if hasattr(pl.callbacks, callback_name):
            callback_class = getattr(pl.callbacks, callback_name)
            # initiate it with chosen args;
            if "params" in callback_:
                params = callback_['params']
                callback_fnc = callback_class(**params)
            else:
                # initiate callback with default args
                callback_fnc = callback_class()
        
        # get custom callback function via the registry;
        else:
            callback_class = registry.get_callback_class(callback_name)
            if callback_class is None:
                raise RuntimeError(f"No callback registered for name: {callback_name}")
            callback_fnc = callback_class(config=config)
        
        callbacks[callback_['type']]=callback_fnc

    return callbacks

def build_metrics(config):
    metrics= nn.ModuleDict()
    metric_list = config.training.metrics
    for metric_ in metric_list:
        metric_name = metric_['type']
        # get pytorch metrics default metrics function;
        if hasattr(torchmetrics, metric_name):
            metric_class = getattr(torchmetrics, metric_name)
            if "params" in metric_:
                params = metric_['params']
                metric_fnc = metric_class(**params)
            else:
                metric_fnc = metric_class()
        
        # get metric function via the registry;
        else:
            metric_class = registry.get_metric_class(metric_name)
            if metric_class is None:
                raise RuntimeError(f"No metric registered for name: {metric_name}")
            metric_fnc = metric_class(config=config)
        
        metrics[metric_['type']]=metric_fnc

    return metrics

# ---------------------------------------------------- Helper Functions ---------------------------------------------

def build_cli_opt_list(opts):
    opts_dot_list = _convert_to_dot_list(opts)
    return OmegaConf.from_dotlist(opts_dot_list)

def _convert_to_dot_list(opts):
    if opts is None:
        opts = []

    if len(opts) == 0:
        return opts

    # Support equal e.g. model=visual_bert for better future hydra support
    has_equal = opts[0].find("=") != -1

    if has_equal:
        return opts

    return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

def _merge_with_dotlist(
    config: DictConfig,
    opts: List[str],
    skip_missing: bool = False,
    log_info: bool = True,
):
    # TODO: To remove technical debt, a possible solution is to use
    # struct mode to update with dotlist OmegaConf node. Look into this
    # in next iteration
    # TODO: Simplify this function
    if opts is None:
        opts = []

    if len(opts) == 0:
        return config

    # Support equal e.g. model=visual_bert for better future hydra support
    has_equal = opts[0].find("=") != -1
    if has_equal:
        opt_values = [opt.split("=", maxsplit=1) for opt in opts]
        if not all(len(opt) == 2 for opt in opt_values):
            for opt in opt_values:
                assert len(opt) == 2, f"{opt} has no value"
    else:
        assert len(opts) % 2 == 0, "Number of opts should be multiple of 2"
        opt_values = zip(opts[0::2], opts[1::2])

    for opt, value in opt_values:
        if opt == "dataset":
            opt = "datasets"

        splits = opt.split(".")
        current = config
        for idx, field in enumerate(splits):
            array_index = -1
            if field.find("[") != -1 and field.find("]") != -1:
                stripped_field = field[: field.find("[")]
                array_index = int(field[field.find("[") + 1 : field.find("]")])
            else:
                stripped_field = field
            if stripped_field not in current:
                if skip_missing is True:
                    break
                raise AttributeError(
                    "While updating configuration"
                    " option {} is missing from"
                    " configuration at field {}".format(opt, stripped_field)
                )
            if isinstance(current[stripped_field], collections.abc.Mapping):
                current = current[stripped_field]
            elif (
                isinstance(current[stripped_field], collections.abc.Sequence)
                and array_index != -1
            ):
                try:
                    current_value = current[stripped_field][array_index]
                except OCErrors.ConfigIndexError:
                    if skip_missing:
                        break
                    raise

                # Case where array element to be updated is last element
                if (
                    not isinstance(
                        current_value,
                        (collections.abc.Mapping, collections.abc.Sequence),
                    )
                    or idx == len(splits) - 1
                ):
                    if log_info:
                        logger.info(f"Overriding option {opt} to {value}")
                    current[stripped_field][array_index] = _decode_value(value)
                else:
                    # Otherwise move on down the chain
                    current = current_value
            else:
                if idx == len(splits) - 1:
                    if log_info:
                        logger.info(f"Overriding option {opt} to {value}")
                    current[stripped_field] = _decode_value(value)
                else:
                    if skip_missing:
                        break

                    raise AttributeError(
                        "While updating configuration",
                        "option {} is not present "
                        "after field {}".format(opt, stripped_field),
                    )

    return config

def _decode_value(value):
    # https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L400
    if not isinstance(value, str):
        return value

    if value == "None":
        value = None

    try:
        value = literal_eval(value)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return value

def _update_specific(config):
    # tp = self.config.training

    # if args["seed"] is not None or tp['seed'] is not None:
    #     print(
    #         "You have chosen to seed the training. This will turn on CUDNN "
    #         "deterministic setting which can slow down your training "
    #         "considerably! You may see unexpected behavior when restarting "
    #         "from checkpoints."
    #     )

    # if args["seed"] == -1:
    #     self.config["training"]["seed"] = random.randint(1, 1000000)

    if (
        "learning_rate" in config
        and "optimizer" in config
        and "params" in config.optimizer
    ):
        lr = config.learning_rate
        config.optimizer.params.lr = lr

    # TODO: Correct the following issue
    # This check is triggered before the config override from
    # commandline is effective even after setting
    # training.device = 'xla', it gets triggered.
    if not torch.cuda.is_available() and "cuda" in config.training.device:
        warnings.warn(
            "Device specified is 'cuda' but cuda is not present. "
            + "Switching to CPU version."
        )
        config.training.device = "cpu"

    return config

def _upgrade_config(config):
    mapping = {
        "training.resume_file": "checkpoint.resume_file",
        "training.resume": "checkpoint.resume",
        "training.resume_best": "checkpoint.resume_best",
        "training.load_pretrained": "checkpoint.resume_pretrained",
        "training.pretrained_state_mapping": "checkpoint.pretrained_state_mapping",
        "training.run_type": "run_type",
    }

    for old, new in mapping.items():
        value = OmegaConf.select(config, old)
        if value:
            OmegaConf.update(config, new, value)

def get_optimizer_parameters(model, config):
    parameters = model.parameters()

    has_custom = hasattr(model, "get_optimizer_parameters")
    
    if has_custom:
        parameters = model.get_optimizer_parameters(config)

    is_parallel = isinstance(model, nn.DataParallel) or isinstance(
        model = nn.parallel.DistributedDataParallel
    )

    if is_parallel and hasattr(model.module, "get_optimizer_parameters"):
        parameters = model.module.get_optimizer_parameters(config)

    # If parameters are a generator, convert to a list first
    parameters = list(parameters)

    if len(parameters) == 0:
        raise ValueError("optimizer got an empty parameter list")

    # If parameters are in format of list, instead of grouped params
    # convert them to grouped params form
    if not isinstance(parameters[0], dict):
        parameters = [{"params": parameters}]

    for group in parameters:
        group["params"] = list(group["params"])

    check_unused_parameters(parameters, model, config)

    return parameters

def check_unused_parameters(parameters, model, config):
    optimizer_param_set = {p for group in parameters for p in group["params"]}
    unused_param_names = []
    for n, p in model.named_parameters():
        if p.requires_grad and p not in optimizer_param_set:
            unused_param_names.append(n)
    if len(unused_param_names) > 0:
        logger.info(
            "Model parameters not used by optimizer: {}".format(
                " ".join(unused_param_names)
            )
        )
        if not config.optimizer.allow_unused_parameters:
            raise Exception(
                "Found model parameters not used by optimizer. Please check the "
                "model's get_optimizer_parameters and add all parameters. If this "
                "is intended, set optimizer.allow_unused_parameters to True to "
                "ignore it."
            )