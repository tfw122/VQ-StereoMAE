"""
This script contains the code for 2 Models:
Masked Vision Model, based on the following: https://arxiv.org/abs/2111.06377

And the base ViT model used for downstream classification.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.utils import make_grid, save_image
from pathlib import Path
#import random
#import timm
#import timm.optim.optim_factory as optim_factory
from functools import partial 
import os 
from collections import OrderedDict
import matplotlib.pyplot as plt 

from modules.pos_embeds import *
from base_model import BaseModel
from modules.image_encoder import *
from modules.masked_vision_layers import *
#from modules.discriminators import Discriminator, MSGDiscriminator
from imports.registry import registry
from modules.layer_utils import *
#from losses.image_reconstruction import MaskedImageLoss, scale_pyramid
#from datasets.vision_transforms_utils import UnNormalise
#from imports.constants import IMAGE_COLOR_MEAN, IMAGE_COLOR_STD

from modules.raft.update import BasicMultiUpdateBlock
from modules.raft.extractor import MultiBasicEncoder, ResidualBlock
from modules.raft.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from modules.raft.utils import coords_grid, upflow8, InputPadder

#from torchmetrics import Accuracy, Precision, F1Score, Recall, AUROC

# ----------------------------- ViT Model for Downstream FineTuning ------------------------------
@registry.register_model("stereo_vit_downstream")
class StereoVITEncoderDownStream(BaseModel):
    def __init__(self, config):
        super().__init__()
        print("!!!!!! BEFORE TRAINING MAKE SURE TO SET gradient_clip_val = 1.0 for STEREO DOWNSTREAM TRAINING !!!!!!")
        print("<<<<<< TRAIN WITH MIXED PRECISION FOR FASTER CONVERGENCE >>>>>>>")
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config =  self.config.dataset_config
        #self.gpu_device = self.config.trainer.params.gpu
        self.frequency_to_visualise = self.model_config.frequency_to_visualise
        self.automatic_optimization = True
        #if self.gpu_device==-1:
        #    self.device_count = torch.cuda.device_count()
        #else:
        #    self.device_count = len(self.gpu_device)
        
        self.image_out_dir= '{}/{}/{}_out_{}_{}'.format(self.config.user_config.save_root_dir, self.config.user_config.username_prefix, self.config.user_config.task_type, self.dataset_config.dataset_name, self.config.user_config.experiment_name)
        if os.path.exists(self.image_out_dir)!=True:
            os.makedirs(self.image_out_dir)
        # encoder args;
        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        num_heads = self.model_config.image_encoder.num_heads
        mlp_ratio = self.model_config.image_encoder.mlp_ratio
        depth = self.model_config.image_encoder.depth
    

        #-------------------------------------- DEFINE STEREO MAE ViT ENCODER ------------------------------------

        # if using public datasets i.e. ImageNet / CIFAR etc, then original VIT model can be used.
        if self.dataset_config.dataset_name=='imagenet' or self.dataset_config.dataset_name=='imagenet_vision':
            print('USING OG TIMM VIT ENCODER')
            self.patch_embedx = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)
            num_patches = self.patch_embedx.num_patches
            
            self.vit_model = VisionTransformer(patch_size=self.patch_size, embed_dim=embed_dim, 
                                            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                            qkv_bias=True,
                                            norm_layer=self.norm_layer)
            self.vit_model.patch_embed = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)
            self.vit_model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            
        # If using external data model is different to account for different image sizes;
        else:
            print('USING CUSTOM VIT ENCODER')
            self.vit_model = VisionTransformer_EncoderOnly(self.config, self.norm_layer, self.model_config.image_encoder.dropout_rate)
        
        self.patch_converter = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=0))
        #self.flatten = Flatten()
        #-------------------------------------- DEFINE RAFT STEREO DECODER ------------------------------------
        self.raft_hidden_dims = self.model_config.raft_decoder.hidden_dims
        self.context_dims = self.model_config.raft_decoder.hidden_dims
        # REPLACED THE FOLLOWING BY THE VIT ENCODER;
        self.cnet = MultiBasicEncoder(output_dim=[self.raft_hidden_dims, self.context_dims], norm_fn="batch", downsample=self.model_config.raft_decoder.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.model_config.raft_decoder, hidden_dims=self.model_config.raft_decoder.hidden_dims)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(self.context_dims[i], self.raft_hidden_dims[i]*3, 3, padding=3//2) for i in range(self.model_config.raft_decoder.n_gru_layers)])
        
        self.conv2 = nn.Sequential(
                ResidualBlock(1, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        #self.conv2_right = nn.Sequential(
        #        ResidualBlock(3, 128, 'instance', stride=1),
        #        nn.Conv2d(128, 256, 3, padding=1))

        
        if self.model_config.load_pretrained_mae!=None:
            self.load_pre_text_pretrained_weights()
            print('pre-text MAE model weights loaded! \n from: {}'.format(self.model_config.load_pretrained_mae))
        else:
            print('no pre-text weights added; initiating with random weights')

        if self.model_config.load_imagenet_supervised!=None:
            self.load_imagenet_weights()
            print('pre-trained OG Imagenet supervised ViT weights loaded! \n from: {}'.format(self.model_config.load_imagenet_supervised))
        else:
            if self.model_config.load_pretrained_mae!=None:
                print("loading pre-trained MAE weights")
            else:
                print('no pre-trained OG Imagenet supervised ViT weights added; initiating with random weights')

        if self.model_config.load_pretrained_raft_decoder!=None:
            self.load_pretrained_raft_decoder_weights()
            print('pre-trained RAFT decoder weights loaded! \n from: {}'.format(self.model_config.load_pretrained_raft_decoder))
        else:
            print('no pre-trained RAFT decoder weights; initiating decoder with random weights')

    def sequence_loss(self, flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
        """ Loss function defined over sequence of flow predictions """

        n_predictions = len(flow_preds)
        assert n_predictions >= 1
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()

        # exclude extremly large displacements
        valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
        assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
        assert not torch.isinf(flow_gt[valid.bool()]).any()

        for i in range(n_predictions):
            assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
            flow_loss += i_weight * i_loss[valid.bool()].mean()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1
    
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.model_config.raft_decoder.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)
    
    def load_imagenet_weights(self):
        # load the model dictionary
        # NOTE: ONLY encoder weights should be loaded; 
        pretrained_dict= torch.load(self.model_config.load_imagenet_supervised)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        model_dict = self.vit_model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.vit_model.load_state_dict(model_dict)
        # if patch embed value != pretrained dict; throw an error; to ensure patchembed is correct.
    
    def load_pre_text_pretrained_weights(self):
        # load pretrained weights (from local);
        pretrained_weights = torch.load(self.model_config.load_pretrained_mae, map_location='cpu')
        pretrained_weights = pretrained_weights['state_dict']
        # these weights will have encoder attached in front of the dict keys.
        # we will clean this up;
        new_pretrained_weights_dict = OrderedDict()
        for k, v in pretrained_weights.items():
            name = k.replace('encoder.', '') # remove `encoder.` k[8:]
            #name = 'vit_model.'+ name # add `vit_model.` to make it identical to current model.
            new_pretrained_weights_dict[name] = v
        # now that weights are identical between vit_model and pretrained encoder weights;
        model_dict = self.vit_model.state_dict()
        # 1. filter out unnecessary keys
        new_pretrained_weights_dict = {k: v for k, v in new_pretrained_weights_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_weights_dict) 
        # 3. load the new state dict
        self.vit_model.load_state_dict(model_dict)

    def weights_transfer(self, model_dict, new_pretrained_weights_dict):
        # 1. filter out unnecessary keys
        new_pretrained_weights_dict_module = {k: v for k, v in new_pretrained_weights_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_weights_dict_module) 
        return model_dict


    def load_pretrained_raft_decoder_weights(self):
        # updating; cnet, update_block, context_zqr_convs with pre-trained RAFT weights;
        pretrained_weights = torch.load(self.model_config.load_pretrained_raft_decoder, map_location='cpu')
        new_pretrained_weights_dict = OrderedDict()

        for k, v in pretrained_weights.items():
            # Raft = module.cnet.norm1.weight, StereoVITEncoderDownStream= cnet.norm1.weight
            name = k.replace('module.', '') # remove `module.`
            new_pretrained_weights_dict[name] = v

        # <<<<<< cnet Update >>>>>>
        # now that weights are identical between cnet and pretrained raft cnet weights;
        model_dict_cnet = self.cnet.state_dict()
        model_dict_cnet = self.weights_transfer(model_dict_cnet, new_pretrained_weights_dict)
        self.cnet.load_state_dict(model_dict_cnet)
        print("cnet updated with pre-trained RAFT")

        # <<<<<< update_block Update >>>>>>
        model_dict_update_block = self.update_block.state_dict()
        model_dict_update_block = self.weights_transfer(model_dict_update_block, new_pretrained_weights_dict)
        self.update_block.load_state_dict(model_dict_update_block)
        print("update_block updated with pre-trained RAFT")

        # <<<<<< context_zqr_convs Update >>>>>>
        model_dict_context_zqr_convs = self.context_zqr_convs.state_dict()
        model_dict_context_zqr_convs = self.weights_transfer(model_dict_context_zqr_convs, new_pretrained_weights_dict)
        self.context_zqr_convs.load_state_dict(model_dict_context_zqr_convs)
        print("context_zqr_convs updated with pre-trained RAFT")

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def to_patch(self, in_dim, out_dim):
        # For converting feature maps into different dims (may not be needed)
        return nn.Linear(in_dim, out_dim, bias=True)

    def reshape_feat(self, feature_map):
        # feature map > remove cls token > cvt to img (unpatchify) > reshape for raft afterwards
        # remove cls token
        feat = feature_map[:, 1:, :] # size: [batch_size, 196, 768]
        # cvt to img (unpatchify)
        #feat_img = self.unpatchify(feat) # 224 * 448 * 3
        return feat

    def forward(self, x_left, x_right, iters=12, flow_init=None, test_mode=False):
        # cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        # fmap1, fmap2 = self.fnet([image1, image2])
        
        left_embed = self.vit_model(x_left)
        right_embed = self.vit_model(x_right)
        
        left_reshaped = self.patch_converter(left_embed.unsqueeze(1))
        right_reshaped = self.patch_converter(right_embed.unsqueeze(1))

        left_reshaped = F.interpolate(left_reshaped, size=(56,112), mode='bilinear')
        right_reshaped = F.interpolate(right_reshaped, size=(56,112), mode='bilinear')

        fmap1 = self.conv2(left_reshaped) # b x 256 x 56 x 56
        fmap2 = self.conv2(right_reshaped) # b x 256 x 56 x 56
        
        cnet_list = self.cnet(x_left, num_layers=self.model_config.raft_decoder.n_gru_layers)

        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]

        # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if self.model_config.raft_decoder.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.model_config.raft_decoder.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.model_config.raft_decoder.corr_implementation == "reg_cuda": # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.model_config.raft_decoder.corr_implementation == "alt_cuda": # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(fmap1, fmap2, radius=self.model_config.raft_decoder.corr_radius, num_levels=self.model_config.raft_decoder.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            
            if self.model_config.raft_decoder.n_gru_layers == 3 and self.model_config.raft_decoder.slow_fast_gru: # Update low-res GRU
                net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
            if self.model_config.raft_decoder.n_gru_layers >= 2 and self.model_config.raft_decoder.slow_fast_gru:# Update low-res GRU and mid-res GRU
                net_list = self.update_block(net_list, inp_list, iter32=self.model_config.raft_decoder.n_gru_layers==3, iter16=True, iter08=False, update=False)
            net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.model_config.raft_decoder.n_gru_layers==3, iter16=self.model_config.raft_decoder.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions 

    def training_step(self, batch, batch_idx):
        # _ ,image1, image2, flow, valid
        _, x_left, x_right, flow, valid = batch
        
        flow_predictions = self.forward(x_left, x_right, iters=self.model_config.raft_decoder.train_iters)
        loss, metrics = self.sequence_loss(flow_predictions, flow, valid)

        # Visualise / save disparity maps for tracking updates;
        # log images;
        if self.global_step % self.frequency_to_visualise ==0:
            self.save_flow(x_left[0,:,:,:], flow_predictions, flow[0,:,:,:])

        self.log('train_loss', loss.item(), rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('train_epe', metrics['epe'], rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_1px', metrics['1px'], rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_3px', metrics['3px'], rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_5px', metrics['5px'], rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # _ ,image1, image2, flow, valid
        _, x_left, x_right, flow, valid = batch
        
        flow_predictions = self.forward(x_left, x_right, iters=self.model_config.raft_decoder.train_iters)
        loss, metrics = self.sequence_loss(flow_predictions, flow, valid)

        self.log('val_loss', loss.item(), rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_epe', metrics['epe'], rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_1px', metrics['1px'], rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_3px', metrics['3px'], rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('val_5px', metrics['5px'], rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def proxy_validation_step(self, batch, batch_idx):
        # _ ,image1, image2, flow, valid
        _, x_left, x_right, flow_gt, valid_gt = batch

        # calculate for each image in a batch;

        out_list, epe_list = [], []
        for i in range(x_left.size(0)):
            x_left_b, x_right_b, flow_gt_b, valid_gt_b = x_left[i,:,:,:], x_right[i,:,:,:], flow_gt[i,:,:,:], valid_gt[i,:,:]
            x_left_b = x_left_b[None]
            x_right_b= x_right_b[None]

            padder = InputPadder(x_left_b.shape, divis_by=32)
            x_left_b, x_right_b = padder.pad(x_left_b, x_right_b)
            
            _, flow_pr = self.forward(x_left_b, x_right_b, iters=self.model_config.raft_decoder.valid_iters, test_mode=True)

            flow_pr = padder.unpad(flow_pr).squeeze(0)
            assert flow_pr.shape == flow_gt_b.shape, (flow_pr.shape, flow_gt_b.shape)
            epe = torch.sum((flow_pr - flow_gt_b)**2, dim=0).sqrt()

            epe = epe.flatten()
            val = (valid_gt_b.flatten() >= 0.5) & (flow_gt_b.abs().flatten() < 192)

            out = (epe > 1.0)

            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())
        
        epe_list = np.array(epe_list)
        out_list = np.concatenate(out_list)

        epe = np.mean(epe_list)
        d1 = 100 * np.mean(out_list)

        self.log('val_epe', epe, rank_zero_only=True, logger=True, sync_dist=True)
        self.log('val_d1', d1, rank_zero_only=True, logger=True, sync_dist=True)

        # The following is done to a list;
        #epe = np.mean(epe)
        #d1 = 100 * np.mean(out)

        #self.log('val_epe', np.mean(epe), rank_zero_only=True, logger=True, sync_dist=True)
        #self.log('val_d1', d1, rank_zero_only=True, logger=True, sync_dist=True)

        return {'epe': epe, 'out': d1}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """ Configure and load optimizers here.
        Create the optimizer and learning rate scheduler """
        lr = 0.0002 # max learning rate.
        wdecay = .00001 # Weight decay in optimizer
        num_steps = 100000 # length of training schedule

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wdecay, eps=1e-8)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def save_flow(self, input_left, flow_predictions, flow_gt):
        flow_up = flow_predictions[-1][0].clone().detach() # the last flow map is the final output
        # convert to a grid;
        flow_up = -flow_up.cpu().numpy().squeeze()
        # np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
        input_left = input_left.squeeze().permute(1, 2, 0).cpu().numpy() # b x c x h x w
        flow_gt = flow_gt.permute(1, 2, 0).cpu().numpy() # b x c x h x w
        #save_image(output_directory / f"{file_stem}.png", -flow_up.cpu()s.numpy().squeeze(), cmap='jet')
        fig, axs = plt.subplots(3)
        fig.suptitle('Left input Image & Predicted Disparity')
        axs[0].imshow(input_left)
        axs[0].axis('off')
        axs[1].imshow(flow_up, cmap='jet')
        axs[1].axis('off')
        axs[2].imshow(flow_gt, cmap='jet')
        axs[2].axis('off')
        fig.savefig('{}/{:08d}.png'.format(self.image_out_dir, self.global_step))
        plt.clf() # closing the figure otherwise; consumes memory.
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        #p = self.patch_embed.patch_size[0]
        ph = self.vit_model.patch_embed.patch_size[0]
        pw = self.vit_model.patch_embed.patch_size[1]

        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // ph
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, ph*pw * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        #p = self.patch_embed.patch_size[0]
        ph = self.vit_model.patch_embed.patch_size[0]
        pw = self.vit_model.patch_embed.patch_size[1]
        
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3,h * ph, w * pw))
        return imgs