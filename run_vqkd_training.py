# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on BEiT, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from timm.models import create_model
from timm.optim import create_optimizer as create_optimizer
import torch.nn as nn
import torch.distributed as dist
#from timm.optim.optim_factory import create_optimizer

from vqkd.datasets import build_vqkd_dataset

from vqkd.engine_for_vqkd import evaluate, train_one_epoch, calculate_codebook_usage
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import vqkd.modeling_vqkd as modeling_vqkd

def get_args():
    parser = argparse.ArgumentParser('BEiT pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    # Model parameters
    parser.add_argument('--model', default='vqkd_encoder_base_decoder_3x768x12_clip', type=str, metavar='MODEL',  help='Name of model to train')  

    parser.add_argument('--rec_loss_type', default='cosine', type=str, metavar='MODEL',
                        help='type of loss to calculate reconstruction distance')

    parser.add_argument('--codebook_n_emd', default=8192, type=int, metavar='MODEL',
                        help='number of codebook')
    parser.add_argument('--codebook_emd_dim', default=32, type=int, metavar='MODEL',
                        help='number of codebook')
    parser.add_argument('--ema_decay', default=0.99, type=float, metavar='MODEL', help='ema decay for quantizer')
    parser.add_argument('--quantize_kmeans_init', action='store_true', help='enable kmeans_init for quantizer')

    parser.add_argument('--process_type', default='default', type=str, choices=['default', 'dall-e', 'imagenet_norm'],
                        help='Image process type (default, dall-e)')
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')

    parser.add_argument('--gpu', type=int, default =1, help='GPU ID to use.')
    # regress feature
    parser.add_argument('--teacher_model_type', default='clip', type=str, help='teacher_model_type during training')
    parser.add_argument('--teacher_input_size', default=224, type=int, help='teacher_input_size for clip-large p14')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0., metavar='PCT',
                        help='Color jitter factor (default: 0.)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic, lanczos default: "bicubic")')
    parser.add_argument('--min_crop_scale', type=float, default=0.08, metavar='PCT',
                        help='min_crop_scale (default: 0.08)')

    # Dataset parameters
    parser.add_argument('--data_path', default='../data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='', type=str, help='dataset path')
    parser.add_argument('--data_set', default='image_folder', type=str, help='dataset path')
 
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--output_dir', default='../data/tokeniser_output/beit-2',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--dist_eval', action='store_true', default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', action='store_true', default=False)
    
    parser.add_argument('--eval', action='store_true', default=False, help="Perform evaluation only")
    parser.add_argument('--calculate_codebook_usage', action='store_true', default=False)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--rank', type=int, default=1, help='Rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', default = False, action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def get_vqkd_model(args, **kwargs):
       
    model = create_model(
        args.model,
        pretrained=False,
        as_tokenzer=False,
        n_code=args.codebook_n_emd,
        code_dim=args.codebook_emd_dim,
        img_size=args.input_size,
        rec_loss_type=args.rec_loss_type,
        teacher_model_type=args.teacher_model_type,
        teacher_input_size=args.teacher_input_size,
        decay=args.ema_decay,
        quantize_kmeans_init=args.quantize_kmeans_init,
        process_type=args.process_type
    )
    return model


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    model = get_vqkd_model(args).to(device)

    model = torch.nn.DataParallel(model)

    # get dataset
    dataset_train = build_vqkd_dataset(is_train=True, args=args)
    if args.disable_eval:
        dataset_val = None
    else:
        dataset_val = build_vqkd_dataset(is_train=False, args=args)

    
    if args.rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if dataset_val:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    model_without_ddp = model
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size 
    if not args.eval:
        print("Model = %s" % str(model_without_ddp))
    for part in ['encoder', 'decoder']:

        if isinstance(model, torch.nn.DataParallel):

            model_part = eval(f"model.module.{part}")
        else:
            model_part = eval(f"model.{part}")

        n_learnable_parameters = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
        n_fix_parameters = sum(p.numel() for p in model_part.parameters() if not p.requires_grad)
        print(f'number of learnable params in model.{part}: {n_learnable_parameters / 1e6} M')
        print(f'number of fixed params in model.{part}: {n_fix_parameters / 1e6} M')

    n_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fix_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'total number of learnable params: {n_learnable_parameters / 1e6} M')
    print(f'total number of fixed params in : {n_fix_parameters / 1e6} M')

    total_batch_size = args.batch_size 
    args.lr = total_batch_size / 128 * args.lr
    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    print("Weigth Decay = %.8f" % args.weight_decay)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    #if args.distributed:

    #    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #    model_without_ddp = model.module
    #else:
    #    pass

    print("Use step level LR & WD scheduler!")

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, len(data_loader_train),
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(args, model, optimizer, loss_scaler)
        
            
    if args.eval:
        evaluate(data_loader_val, model, device, None, 0, args)
        return
        
    if args.calculate_codebook_usage:
        calculate_codebook_usage(data_loader_val, model, device, None, 0, args)
        return
        
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
            
    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        #data_loader_train.sampler.set_epoch(epoch)

        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model, 
            data_loader_train,
            optimizer, 
            device, 
            epoch, 
            loss_scaler,
            args.clip_grad, 
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            args=args
        )
        if args.output_dir:
            # if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq)
        
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, log_writer, epoch, args=args)
            print(f"Validation loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.4f}")

            if log_writer is not None:
                log_writer.update(**test_stats, head="val/loss")
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch, 'n_parameters': n_learnable_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch, 'n_parameters': n_learnable_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args()
    main(args)

