"""
This script contains the code for 2 Models:
Masked Vision Model, based on the following: https://arxiv.org/abs/2111.06377

And the base ViT model used for downstream classification.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional 
from torchvision.utils import make_grid, save_image
import random
import timm
import timm.optim.optim_factory as optim_factory
from functools import partial 
import os 
from collections import OrderedDict

from vqkd.modeling_vqkd import vqkd_encoder_base_decoder_3x768x12_clip
from modules.pos_embeds import *
from models.base_model import BaseModel
from modules.image_encoder import *
from modules.masked_vision_layers import *
from modules.discriminators import Discriminator, MSGDiscriminator
from imports.registry import registry
from modules.layer_utils import *
from losses.image_reconstruction import MaskedImageLoss, scale_pyramid
from datasets.vision_transforms_utils import UnNormalise
from imports.constants import IMAGE_COLOR_MEAN, IMAGE_COLOR_STD
from torchmetrics import Accuracy, Precision, F1Score, Recall, AUROC
import vqkd.modeling_vqkd 
from timm.models import create_model


@registry.register_model("masked_image_autoencoder")
class MaskedImageAutoEncoder(BaseModel):
    """
    based on the paper: https://arxiv.org/abs/2111.06377
    Masked Autoencoder with VisionTransformer backbone

    From my experiments, model works better when image inputs are NOT Normalised!
    """
    def __init__(self, config):
        super().__init__()
        print('for this work well, make sure inputs are not normalised!')
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config =  self.config.dataset_config
        self.user_config = self.config.user_config
        self.image_out_dir= '{}/{}/mae_out_test_{}_{}_{}'.format(self.config.user_config.save_root_dir, self.config.user_config.username_prefix, self.dataset_config.dataset_name, self.model_config.loss_type, self.user_config.experiment_name)
        if os.path.exists(self.image_out_dir)!=True:
            os.makedirs(self.image_out_dir)
        # patch embed args;

        self.patch_embed = PatchEmbed(
            img_size=self.dataset_config.preprocess.vision_transforms.params.Resize.size,
            patch_size=self.model_config.image_encoder.patch_size,
            in_channels=self.model_config.image_encoder.in_channels,
            embed_dim=self.model_config.image_encoder.embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.num_heads = self.model_config.image_encoder.num_heads
        #self.num_heads = self.model_config.num_heads


        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_config.image_encoder.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.model_config.image_encoder.embed_dim))    
        #self.mask_ratio = self.model_config.mask_ratio
  
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 1, self.model_config.image_encoder.embed_dim))

        self.pos_drop = nn.Dropout(p=0.1)
        self.qk_scale = self.model_config.qk_scale

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, self.model_config.drop_path_rate, self.model_config.image_encoder.depth)]  
        self.blocks = nn.ModuleList([
            Block(
                dim=self.model_config.image_encoder.embed_dim, num_heads = self.model_config.image_encoder.num_heads, mlp_ratio=self.model_config.image_encoder.mlp_ratio,
                drop=self.model_config.drop_rate, attn_drop=self.model_config.attn_drop_rate, drop_path=dpr[i],
                act_layer=self.model_config.act_layer
            )
            for i in range(self.model_config.image_encoder.depth)])
        self.norm = self.model_config.norm_layer(self.model_config.image_encoder.embed_dim)
        
        self.init_std = self.model_config.init_std
        self.lm_head = nn.Linear(self.model_config.image_encoder.embed_dim, self.model_config.vocab_size)   

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

        self.finetune_imagenet= self.model_config.finetune_imagenet
        self.num_samples_to_visualise = self.model_config.num_samples_to_visualise
        self.frequency_to_visualise = self.model_config.frequency_to_visualise

        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        self.norm_layer_arg= self.model_config.norm_layer_arg
        
        if self.norm_layer_arg=='partial':
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
            print('using partial layer norm')
        else:
            self.norm_layer = nn.LayerNorm
        
        self.patch_embed = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = MAEEncoder(config, self.patch_embed, self.norm_layer)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder = MAEDecoder(config, self.patch_embed, self.norm_layer)
        # --------------------------------------------------------------------------

        # --------- build loss ---------
        self.loss_fnc = MaskedImageLoss(config, self.patch_embed)
        # if using the GAN loss; initiate the discriminator;
        if self.model_config.loss_type=='gan':
            raise Exception('To use the GAN loss, use the following model: {}. This model implementation does not support GAN loss'.format(
                            'masked_image_autoencoder_gan_loss'
            ))
        
        if self.model_config.normalisation_params=='imagenet':
            self.unnormalise = UnNormalise(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD)
        else:
            raise Exception("the following type of normalisation has not been implemented: {}".format(self.model_config.normalisation_params))

        if self.finetune_imagenet!=None:
            self.load_imagenet_weights()
            print('OG imagenet weights loaded from: {} \n to commence finetuning'.format(self.finetune_imagenet))

    def get_visual_tokenizer(args):
        print(f"Creating visual tokenizer: {args.model_config.tokenizer_model}")
        model = create_model(
                args.model_config.tokenizer_model,
                pretrained=True,
                #pretrained_weight=args.tokenizer_weight,
                as_tokenzer=True,
                n_code=args.model_config.codebook_size, 
                code_dim=args.model_config.codebook_dim,
            ).eval()
        return model

    def load_imagenet_weights(self):
        # load the model dictionary
        # NOTE: ONLY encoder weights should be loaded; decoder has to be trained from scratch for the specific data;
        # otherwise no point in doing MAE since imagenet distribution can also reconstruct different image types.
        #pretrained_dict= torch.load(self.finetune_imagenet)
        pretrained_dict= torch.load(self.finetune_imagenet)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        model_dict = self.encoder.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.encoder.load_state_dict(model_dict)
        # if patch embed value != pretrained dict; throw an error; to ensure patchembed is correct.
        
    def forward(self, x_left, x_right, mask_ratio=None, return_all_tokens = False, return_patch_tokens = False):
        # LEFT IMAGE INFERENCE
        # run encoder;
        if mask_ratio==None:
            l_latent, l_masks, l_ids_restore = self.encoder(x_left, self.mask_ratio)
        else:
            l_latent, l_masks, l_ids_restore = self.encoder(x_left, mask_ratio)
        # run decoder;
        l_predictions = self.decoder(l_latent, l_ids_restore)  # [N, L, p*p*3]

        # SHARED WEIGHTS: RIGHT IMAGE INFERENCE
        # run encoder;
        if mask_ratio==None:
            r_latent, r_masks, r_ids_restore = self.encoder(x_right, self.mask_ratio)
        else:
            r_latent, r_masks, r_ids_restore = self.encoder(x_right, mask_ratio)
        # run decoder;
        r_predictions = self.decoder(r_latent, r_ids_restore)  # [N, L, p*p*3]
        
        return (l_predictions, r_predictions), (l_masks, r_masks)

    def training_step(self, batch, batch_idx):
        x_left, x_right = batch['left_image'], batch['right_image']
        
        out, mask = self(x_left, x_right)
        #loss=0.0 # initiate loss variable as 0 then add to it in a loop'
        
        loss = self.loss_fnc((x_left, x_right), out, mask)
        
        # clone logits for metrics (don't want gradients to pass)
        self.log('train_loss', loss.item(), rank_zero_only=True, prog_bar=True, logger=True)
        
        # log images;
        if self.global_step % self.frequency_to_visualise ==0:
            preds_left = out[0].clone().detach() # left image
            preds_right = out[1].clone().detach() # right image

            mask_left = mask[0].clone().detach() # left mask
            mask_right = mask[1].clone().detach() # right mask
            
            # [rand_batch_id,:,:].unsqueeze(0)
            orig_img_left, masked_img_left, recon_left, _= self.visualise_sample(preds_left[0:2,:,:], 
                                                                                 mask_left[0:2,:], 
                                                                                 x_left[0:2,:,:,:])

            # _, _, _, recon_with_visible_right
            orig_img_right, masked_img_right, recon_right, _= self.visualise_sample(preds_right[0:2,:,:], 
                                                                                    mask_right[0:2,:], 
                                                                                    x_right[0:2,:,:,:])
            # deifne grid;
            grid = make_grid(
                torch.cat((orig_img_left.permute(0,3,1,2), orig_img_right.permute(0,3,1,2), 
                           masked_img_left.permute(0,3,1,2), masked_img_right.permute(0,3,1,2),
                           recon_left.permute(0,3,1,2), recon_right.permute(0,3,1,2)), dim=0))

            #self.logger.experiment.add_image('train_images', grid, batch_idx, self.global_step)
            
            save_image(grid, '{}/{:08d}.png'.format(self.image_out_dir, self.global_step))
        
        return loss

    def validation_step(self, batch, batch_idx):
        x_left, x_right = batch['left_image'], batch['right_image']

        out, mask = self(x_left, x_right)
        loss=0.0 # initiate loss variable as 0 then add to it in a loop'
        
        loss += self.loss_fnc((x_left, x_right), out, mask)
        
        # clone logits for metrics (don't want gradients to pass)
        self.log('val_loss', loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        # calculate other metrics
        #preds = out>0.5
        #metrics_output = self.calculate_metrics(out, y.int().squeeze(1)) 

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    
    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        weight_decay=0.05
        blr= 1.5e-4
        min_lr = 0.
        warmup_epochs=15
        betas= (0.9, 0.95)

        # following timm: set wd as 0 for bias and norm layers
        param_groups = optim_factory.add_weight_decay(self, weight_decay) # weight_decay;
        optimizer = torch.optim.Adam(self.parameters(), lr=blr, betas=betas)
        #print(optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9) #build_scheduler(optimizer, self.config)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    # --------------- helper functions ----------------
    #def on_train_epoch_start(self):
    #    if self.current_epoch==0:
    #        sample_input= torch.randn((8,3,10,224,224))
    #        self.logger.experiment.add_graph(MaskedImageAutoEncoder(self.config),sample_input)

    def visualise_sample(self, pred, mask, img):
        y = self.unpatchify(pred)
        y = torch.einsum('nchw->nhwc', y).detach() #.cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1] *3)  # (N, H*W, p*p*3)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach() #.cpu()
        
        x = torch.einsum('nchw->nhwc', img)

        # masked image
        im_masked = x * (1 - mask)

        # model reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # return (original image, masked image, model reconstruction, fused; reconstruction + visible pixels)
        #return x[0], im_masked[0], y[0], im_paste[0]
        return x, im_masked, y, im_paste

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        #p = self.patch_embed.patch_size[0]
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]

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
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]
        
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3,h * ph, w * pw))
        return imgs

        # grid = torchvision.utils.make_grid(sample_imgs) 
        # self.logger.experiment.add_image('generated_images', grid, 0) 

    # Assuming the existing code (VQ-based model) is already in place...

    def patch_aggregation(vq_output, masked_output):
        """
        Function to aggregate and concatenate VQ-based model's output and masked image model's output.
        
        Parameters:
        - vq_output: tuple of tensors; (l_quant, r_quant) from the VQ-based model.
        - masked_output: tensor; output from the masked image model.

        Returns:
        - aggregated_output: concatenated output tensor.
        """
        
        # Flatten the output tensors for concatenation
        flattened_l_quant = vq_output[0].view(vq_output[0].size(0), -1)
        flattened_r_quant = vq_output[1].view(vq_output[1].size(0), -1)
        flattened_masked_output = masked_output.view(masked_output.size(0), -1)
        
        # Concatenate along the feature dimension
        aggregated_output = torch.cat([flattened_l_quant, flattened_r_quant, flattened_masked_output], dim=1)
        
        return aggregated_output
    







































@registry.register_model("masked_image_autoencoder_msg_gan")
class MultiScaleMaskedImageAutoEncoder(BaseModel):
    """
    Multi Scale version of the model above
    """
    def __init__(self, config):
        super().__init__()
        print('for this work well, make sure inputs are not normalised!')
        
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config =  self.config.dataset_config
        assert self.model_config.loss_type=='gan' or self.model_config.loss_type=='gan_perceptual', "GAN Loss type must be a gan to use this model!"

        self.image_out_dir= '{}/{}/mae_out_test_{}_{}_{}'.format(self.config.user_config.save_root_dir, self.config.user_config.username_prefix, self.dataset_config.dataset_name, self.model_config.loss_type, self.user_config.experiment_name)
        
        if os.path.exists(self.image_out_dir)!=True:
            os.makedirs(self.image_out_dir)
        
        # patch embed args;

        self.patch_embed = PatchEmbed(
        img_size=self.dataset_config.preprocess.vision_transforms.params.Resize.size,
        patch_size=self.model_config.image_encoder.patch_size,
        in_channels=self.model_config.image_encoder.in_channels,
        embed_dim=self.model_config.image_encoder.embed_dim,
        )

        num_patches = self.patch_embed.num_patches
        self.num_heads = self.model_config.image_encoder.num_headss


        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_config.image_encoder.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.model_config.image_encoder.embed_dim))    
        #self.mask_ratio = self.model_config.mask_ratio
  
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 1, self.model_config.image_encoder.embed_dim))

        self.pos_drop = nn.Dropout(p=0.1)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, self.model_config.drop_path_rate, self.model_config.image_encoder.depth)]  

        self.blocks = nn.ModuleList([Block(
            dim=self.model_config.image_encoder.embed_dim, num_heads=self.model_config.num_heads, mlp_ratio=self.model_config.image_encoder.mlp_ratio, qkv_bias=self.model_config.qkv_bias, qk_scale=self.model_config.qk_scale,
            drop=self.model_config.drop_rate, attn_drop=self.model_config.attn_drop_rate, drop_path=dpr[i], norm_layer=self.model_config.norm_layer,
            init_values=self.model_config.init_values, window_size=self.patch_embed.patch_shape if self.model_config.use_rel_pos_bias else None,
            attn_head_dim=self.model_config.attn_head_dim,
            )
            for i in range(self.model_config.image_encoder.depth)])   
        self.norm = self.model_config.norm_layer(self.model_config.image_encoder.embed_dim) 
        self.init_std = self.model_config.init_std  
        self.lm_head = nn.Linear(self.model_config.image_encoder.embed_dim, self.model_config.vocab_size)   

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

        self.mask_ratio = self.model_config.mask_ratio
        self.finetune_imagenet= self.model_config.finetune_imagenet
        self.num_samples_to_visualise = self.model_config.num_samples_to_visualise
        self.frequency_to_visualise = self.model_config.frequency_to_visualise
        self.scales = self.model_config.discriminator.depth

        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        self.norm_layer_arg= self.model_config.norm_layer_arg
        
        if self.norm_layer_arg=='partial':
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
            print('using partial layer norm')
        else:
            self.norm_layer = nn.LayerNorm
        
        self.patch_embed = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)

        # Define the Generator;
        self.generator = nn.ModuleDict({
            'patch_embed': PatchEmbed(img_size, self.patch_size, in_channels, embed_dim),
            'encoder': MSGMAEEncoder(config, self.patch_embed, self.norm_layer),
            'decoder': MSGMAEDecoder(config, self.patch_embed, self.norm_layer)
        })

        # --------- build loss ---------
        self.loss_fnc = MaskedImageLoss(config, self.patch_embed)

        if self.finetune_imagenet!=None:
            self.load_imagenet_weights()
            print('og imagenet weights loaded from: {} \n to commence finetuning'.format(self.finetune_imagenet))


    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)
    
    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  
        mask_token = self.mask_token.expand(batch_size, -1, -1)  

        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)
    
    
    def load_imagenet_weights(self):
        # load the model dictionary
        # NOTE: ONLY encoder weights should be loaded; decoder has to be trained from scratch for the specific data;
        # otherwise no point in doing MAE since imagenet distribution can also reconstruct different image types.
        pretrained_dict= torch.load(self.finetune_imagenet)
        model_dict = self.generator.encoder.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.generator.encoder.load_state_dict(model_dict)
        # if patch embed value != pretrained dict; throw an error; to ensure patchembed is correct.
        
    def forward(self, x_left, x_right, bool_masked_pos=None, layer_id=12, return_qkv=False, split_out_as_qkv=False, return_all_tokens=False, return_patch_tokens=False):    
        
        def process_input(x, bool_masked_pos, layer_id, return_qkv, split_out_as_qkv, return_all_tokens=False, return_patch_tokens=False):
            if bool_masked_pos is None:
                bool_masked_pos = torch.zeros((x.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(x.device)
            x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
            batch_size, seq_len, _ = x.size()

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)

            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

            x = torch.cat((cls_tokens, x), dim=1)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            x = self.pos_drop(x)

            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

            # handling intermediate layers
            if isinstance(layer_id, list):
                output_list = []
                for l, blk in enumerate(self.blocks):
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                    if l in layer_id:
                        output_list.append(x[:, 1:])
                return output_list
            elif isinstance(layer_id, int):
                for l, blk in enumerate(self.blocks):
                    if l < layer_id:
                        x = blk(x, rel_pos_bias=rel_pos_bias)
                    elif l == layer_id:
                        x = blk.norm1(x)
                    else:
                        break
                x = x[:, 1:]
            else:
                raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

            if return_qkv:
                qkv = None
                for i, blk in enumerate(self.blocks):
                    if i < len(self.blocks) - 1:
                        x = blk(x, rel_pos_bias=rel_pos_bias)
                    else:
                        x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)

                if split_out_as_qkv:
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    return x, q, k, v
                else:
                    x = self.norm(x)
                    x = self.lm_head(x[bool_masked_pos])

                return x, qkv
            elif return_all_tokens or return_patch_tokens:
                x = x[:, 1:]
                if return_patch_tokens:
                    return x
                return self.lm_head(x)
            else:
                x = self.norm(x)
                return self.lm_head(x[bool_masked_pos])
        
        # Process each input separately
        result_left = process_input(x_left, bool_masked_pos, layer_id, return_qkv, split_out_as_qkv, return_all_tokens, return_patch_tokens)
        result_right = process_input(x_right, bool_masked_pos, layer_id, return_qkv, split_out_as_qkv, return_all_tokens, return_patch_tokens)

        # Return combined results
        return result_left, result_right

    def forward_get_last_selfattention(self, x_left, x_right):
        def process_input(x):
            x = self.patch_embed(x)
            batch_size, seq_len, _ = x.size()
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            x = self.pos_drop(x)
            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                else:
                    # return attention of the last block
                    return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)

        # Process each input separately
        attention_left = process_input(x_left)
        attention_right = process_input(x_right)

        # Return combined results
        return attention_left, attention_right


    def training_step(self, batch, batch_idx, optimizer_idx):
        x_left, x_right = batch['left_image'], batch['right_image']

        x = torch.cat((x_left, x_right), dim=1)

        out, mask, scaled_images = self(x, x)
    
        for i in range(x.size(2)):
            # imgs, pred, mask
            if self.dataset_config.max_images>1:
                scaled_gt = scale_pyramid(x[:,:,i,:,:], self.scales)
                
                loss = self.loss_fnc(scaled_gt[::-1], scaled_images[i][::-1], mask[:,i,:], self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]
            else:
                scaled_gt = scale_pyramid(x[:,:,i,:,:], self.scales)
                loss = self.loss_fnc(scaled_gt[::-1], scaled_images[i][::-1], mask, self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]

        # perform back prop;
        # ------------ GENERATOR --------------
        if optimizer_idx==0:

            # clone logits for metrics (don't want gradients to pass)
            self.log('gen_loss', gen_loss.item(), rank_zero_only=True, prog_bar=True, logger=True)
            
            # log images;
            if self.global_step % self.frequency_to_visualise ==0:
                preds = out.clone().detach()
                # randomly select a batch and n images
                if self.dataset_config.max_images>1:
                    rand_img_ids = random.sample(range(0, self.dataset_config.max_images-1), self.num_samples_to_visualise)
                    rand_batch_id= random.randint(0, self.config.training.batch_size-1)
                else:
                    rand_img_ids= [0]
                    rand_batch_id= 0
                Images, Masked_Images, Recons, ReconsVisible=[],[],[],[]
                
                
                for j in rand_img_ids:
                    if self.dataset_config.max_images>1:
                        orig_img, masked_img, recon, recon_with_visible= self.visualise_sample(preds[rand_batch_id,j,:,:].unsqueeze(0), 
                                                                                            mask[rand_batch_id,j,:].unsqueeze(0), 
                                                                                            x[rand_batch_id,:,j,:,:].unsqueeze(0))
                    else:
                        # only one image;
                        orig_img, masked_img, recon, recon_with_visible= self.visualise_sample(preds[rand_batch_id,:,:].unsqueeze(0), 
                                                                                            mask[rand_batch_id,:].unsqueeze(0), 
                                                                                            x[rand_batch_id,:,j,:,:].unsqueeze(0))
                    
                    Images.append(orig_img.permute(2,0,1))
                    Masked_Images.append(masked_img.permute(2,0,1))
                    Recons.append(recon.permute(2,0,1))
                    ReconsVisible.append(recon_with_visible.permute(2,0,1))
                
                Images = torch.stack(Images)
                Masked_Images = torch.stack(Masked_Images)
                Recons = torch.stack(Recons)
                ReconsVisible = torch.stack(ReconsVisible)
                
                #print(Images.size(), Masked_Images.size(), Recons.size(), ReconsVisible.size())
                grid = make_grid(
                    torch.cat((Images, Masked_Images, 
                            Recons, ReconsVisible), dim=0))
                #grid = make_grid(
                #    torch.cat((self.unnormalise(Images), self.unnormalise(Masked_Images), 
                #               self.unnormalise(Recons), self.unnormalise(ReconsVisible)), dim=0))
                
                #self.logger.experiment.add_image('train_images', grid, batch_idx, self.global_step)
                
                save_image(grid, '{}/{}.png'.format(self.image_out_dir, self.global_step))
            
            return gen_loss
        
        # ------------ DISC Loss --------------
        if optimizer_idx==1:
            # clone logits for metrics (don't want gradients to pass)
            self.log('disc_loss', disc_loss.item(), rank_zero_only=True, prog_bar=True, logger=True)
            return disc_loss
            

    def validation_step(self, batch, batch_idx):
        x_left, x_right = batch['left_image'], batch['right_image']
        
        x = torch.cat((x_left, x_right), dim=1)  # dim=1 refers to the channel dimension in a (batch, channel, height, width) tensor

        out, mask, scaled_images = self(x, x)
    
        disc_loss, gen_loss=0.0, 0.0 # initiate loss variable as 0 then add to it in a loop'
        
        for i in range(x.size(2)):
            # imgs, pred, mask
            if self.dataset_config.max_images>1:
                scaled_gt = scale_pyramid(x[:,:,i,:,:], self.scales)
                
                loss = self.loss_fnc(scaled_gt[::-1], scaled_images[i][::-1], mask[:,i,:], self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]
            else:
                scaled_gt = scale_pyramid(x[:,:,i,:,:], self.scales)
                loss = self.loss_fnc(scaled_gt[::-1], scaled_images[i][::-1], mask, self.current_epoch)
                disc_loss += loss[0]
                gen_loss += loss[1]
        

        # clone logits for metrics (don't want gradients to pass)
        self.log('val_loss', gen_loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_disc_loss', disc_loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)
        # calculate other metrics
        #preds = out>0.5
        #metrics_output = self.calculate_metrics(out, y.int().squeeze(1)) 

        return {'val_loss': gen_loss}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    
    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        weight_decay=0.05
        blr= 1.5e-4
        min_lr = 0.
        warmup_epochs=20
        betas= (0.9, 0.95)

        # following timm: set wd as 0 for bias and norm layers
        #param_groups = optim_factory.add_weight_decay(self, weight_decay) # weight_decay;
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=blr, betas=betas)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=blr, betas=betas)
        #print(optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=30, gamma=0.9) #build_scheduler(optimizer, self.config)
        return [optimizer_gen, optimizer_disc], [scheduler]


    def visualise_sample(self, pred, mask, img):
        y = self.unpatchify(pred)
        y = torch.einsum('nchw->nhwc', y).detach() #.cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach() #.cpu()
        
        x = torch.einsum('nchw->nhwc', img)

        # masked image
        im_masked = x * (1 - mask)

        # model reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # return (original image, masked image, model reconstruction, fused; reconstruction + visible pixels)
        return x[0], im_masked[0], y[0], im_paste[0]

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        #p = self.patch_embed.patch_size[0]
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]

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
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]

        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3,h * ph, w * pw))
        return imgs








































# ----------------------------- ViT Model for Downstream FineTuning ------------------------------
@registry.register_model("vit_downstream")
class VITEncoderDownStream(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.data_type = self.model_config.data_type
        self.dataset_config =  self.config.dataset_config
        self.transformer_params= self.model_config.transformer
        self.train_task = self.model_config.train_task
        #self.gpu_device = self.config.trainer.params.gpus
        #if self.gpu_device==-1:
        #    self.device_count = torch.cuda.device_count()
        #else:
        #    self.device_count = len(self.gpu_device)
        
        self.output_dir=  '../data/{}/mae_out_test_{}_{}_{}'.format(self.config.user_config.s3key_prefix, self.dataset_config.dataset_name, self.model_config.loss_type, self.user_config.experiment_name)
        if os.path.exists(self.output_dir)!=True:
            os.makedirs(self.output_dir)
        
        # encoder args;
        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        num_heads = self.model_config.image_encoder.num_heads
        mlp_ratio = self.model_config.image_encoder.mlp_ratio
        depth = self.model_config.image_encoder.depth

        self.norm_layer_arg= self.model_config.norm_layer_arg
        
        if self.norm_layer_arg=='partial':
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
            print('using partial layer norm')
        else:
            self.norm_layer = nn.LayerNorm

        # if using public datasets i.e. ImageNet / CIFAR etc, then original VIT model can be used.
        if self.dataset_config.dataset_name=='imagenet' or self.dataset_config.dataset_name=='imagenet_vision':
            self.patch_embedx = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)
            num_patches = self.patch_embedx.num_patches
            
            self.vit_model = VisionTransformer(patch_size=self.patch_size, embed_dim=embed_dim, 
                                            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                            qkv_bias=True,
                                            norm_layer=self.norm_layer)
            self.vit_model.patch_embed = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)
            self.vit_model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            # change the final layer classifier; to desired classes!
            #self.vit_model.head = nn.Linear(in_features=768, out_features=self.model_config.classifier.num_classes, bias=True)

        # If using Tractable data classification model is different to account for multiple images;
        else:
            self.vit_model = VisionTransformer_EncoderOnly(self.config, self.norm_layer, self.transformer_params.dropout_rate)
            self.linearize = nn.Conv2d(self.vit_model.num_patches + 1, 1, 1) # add 1 to patch to account for posembed


            self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model = self.transformer_params.d_model, 
                                        nhead=self.transformer_params.nhead, 
                                        dim_feedforward=self.transformer_params.dim_feedforward, 
                                        dropout= self.transformer_params.dropout_rate,
                                        batch_first=True)
            for i in range(self.transformer_params.depth)])

            self.transformer_layers = nn.Sequential(*self.transformer_blocks)

            #self.transformer_ = nn.TransformerEncoderLayer(d_model = self.transformer_params.d_model, 
            #                                            nhead=self.transformer_params.nhead, 
            #                                            dim_feedforward=self.transformer_params.dim_feedforward, 
            #                                            dropout= self.transformer_params.dropout_rate,
            #                                            batch_first=True) #256, 8, 512,

            #self.transformer_2 = nn.TransformerEncoderLayer(d_model = self.transformer_params.d_model, 
            #                                            nhead=self.transformer_params.nhead, 
            #                                            dim_feedforward=self.transformer_params.dim_feedforward, 
            #                                            dropout= self.transformer_params.dropout_rate,
            #                                            batch_first=True) #256, 8, 512,
       
            # -------- final fully connected --------;
            self.attn_pooling = nn.Conv2d(self.config.dataset_config.max_images, 1, 1)
            self.act = nn.ReLU()
            self.output_layer = nn.Linear(self.model_config.classifier.in_dim, self.model_config.classifier.num_classes)
        
        if self.model_config.load_pretrained_mae!=None:
            self.load_pre_text_pretrained_weights()
            print('pre-text MAE model weights loaded! \n from: {}'.format(self.model_config.load_pretrained_mae))
        else:
            print('no pre-text weights added; initiating with random weights')

        # if linear probing; freeze encoder, otherwise train as normal.
        if self.train_task=='linear_probe':
            self.vit_model = self.freeze_model(self.vit_model)
            print('LINEAR PROBE SELECTED! HENCE, FREEZING VIT ENCODER MODEL')
        else:
            print('FINE TUNING! HENCE, VIT ENCODER MODEL WILL BE TRAINED AS NORMAL')

        # -------- metrics ----------
         #self.metrics = #build_metrics(self.config)
        self.Precision = Precision(threshold=0.54, average='samples')
        self.Recall = Recall(threshold=0.54, average='samples')
        self.F1 = F1Score(threshold=0.54, average='samples', mdmc_average='samplewise')
        #self.Auroc= AUROC(average='micro')
        self.Accuracy = Accuracy(threshold=0.54, average='samples')
        # -------- loss function ----------
        if self.data_type=='public':
            self.loss_fnc = nn.CrossEntropyLoss()
        else:
            self.loss_fnc = nn.BCEWithLogitsLoss()
        
        self.sigmoid = nn.Sigmoid()

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

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
        return model

    def forward(self, x, pad_mask=None):
        if self.dataset_config.dataset_name=='imagenet' or self.dataset_config.dataset_name=='imagenet_vision':
            logits = self.vit_model(x)
            return logits #self.out_activation(logits)
        else:
            embeddings, embeddings_linearised= [], []
            
            # loop over the images;
            for i in range(x.size(2)):
                imgs = x[:,:,i,:,:]
                embed1 = self.vit_model(imgs)
                embeddings.append(embed1)
            embeddings = torch.stack(embeddings).permute(1,0,2,3)

            # linearize the patches:
            for j in range(embeddings.size(1)):
                embeddings_linearised.append(self.linearize(embeddings[:,j,:,:].unsqueeze(2)))
            embeddings_linearised = torch.stack(embeddings_linearised).squeeze()
            if embeddings_linearised.dim() < 3:
                embeddings_linearised = embeddings_linearised.unsqueeze(0)
            else:
                embeddings_linearised= embeddings_linearised.permute(1,0,2)

            # pass through transformer & classifier
            #embed2 = self.transformer_(embeddings_linearised, src_key_padding_mask= pad_mask.type(torch.cuda.FloatTensor))
            #embed2 = self.transformer_2(embed2, src_key_padding_mask= pad_mask.type(torch.cuda.FloatTensor))
            embed2 = self.transformer_layers(embeddings_linearised, src_key_padding_mask= pad_mask.type(torch.cuda.FloatTensor))
            embed2 = embed2.unsqueeze(2)
            embed2 = self.act(self.attn_pooling(embed2).squeeze(1))
            logits = self.output_layer(embed2.squeeze(1))
            
            return logits

    def training_step(self, batch):

        if self.dataset_config.dataset_name == 'imagenet':
            x, y = batch[0], batch[1] # in imagenet sample[0]== image, sample[1]== class
            pad_mask=None
        else:
            x, pad_mask, y = batch['images'], batch['pad_mask'], batch['labels']
            y = y.squeeze(1)
        
        out = self.forward(x, pad_mask)
        loss = self.loss_fnc(out, y)

        #print(out.size(), y.size(), '{:.3f}'.format(loss.item()))
        # clone logits for metrics (don't want gradients to pass)
        self.log('train_loss', loss.item(), rank_zero_only=True, prog_bar=True, logger=True)
        
        # calculate metrics;
        preds = out.clone()

        f1_score = self.F1(preds, y.int())  
        precision_score = self.Precision(preds, y.int()) 
        recall_score = self.Recall(preds, y.int()) 
        accuracy_score = self.Accuracy(preds, y.int()) 
        #auroc_score = self.Auroc(preds, y.int().squeeze(1)) 

        self.log('train_{}'.format('f1'), f1_score, rank_zero_only=True)
        self.log('train_{}'.format('precision'), precision_score, rank_zero_only=True)
        self.log('train_{}'.format('recall'), recall_score, rank_zero_only=True)
        self.log('train_{}'.format('acurracy'), accuracy_score, rank_zero_only=True)
        #self.log('train_{}'.format('auroc'), auroc_score, rank_zero_only=True)

        #for metric_name, metric_val in metrics_output.items():
        #    self.log('train_{}'.format(metric_name), metric_val, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.data_type=='public':
            x, y = batch
            pad_mask=None
        else:
            x, pad_mask, y = batch['images'], batch['pad_mask'], batch['labels']
            y = y.squeeze(1)

        out = self.forward(x, pad_mask)

        loss = self.loss_fnc(out, y)

        # calculate accuracy;
        self.Accuracy(out, y.int())
        # calculate other metrics
        #preds = out>0.5
        #metrics_output = self.calculate_metrics(out, y.int().squeeze(1)) 

        f1_score = self.F1(out, y.int())  
        precision_score = self.Precision(out, y.int()) 
        recall_score = self.Recall(out, y.int()) 
        #auroc_score = self.Auroc(out, y.int().squeeze(1)) 

        # clone logits for metrics (don't want gradients to pass)
        self.log('val_loss', loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('val_{}'.format('f1'), f1_score, rank_zero_only=True, logger=True, sync_dist=True)
        self.log('val_{}'.format('precision'), precision_score, rank_zero_only=True, logger=True, sync_dist=True)
        self.log('val_{}'.format('recall'), recall_score, rank_zero_only=True, logger=True, sync_dist=True)
        #self.log('val_{}'.format('auroc'), auroc_score, rank_zero_only=True)

        # log to logger;
        #for metric_name, metric_val in metrics_output.items():
        #    self.log('val_{}'.format(metric_name), metric_val, rank_zero_only=True)

        self.log("val_acc", self.Accuracy, prog_bar=True, rank_zero_only=True, logger=True, sync_dist=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        if self.data_type=='public':
            x, y = batch
            pad_mask=None
        else:
            x, pad_mask, y = batch['images'], batch['pad_mask'], batch['labels']
            #claim_id = data['claim_id']
            #create a definition for claim_id that fixes the data error
            claim_id = batch['claim_id']
            
            y = y.squeeze(1)

        # get the output types;
        logits = self.forward(x, pad_mask)
        output = self.sigmoid(logits)
        predictions = (output>0.55).float()

        bce_loss = self.loss_fnc(logits, y)
        # clone logits for metrics (don't want gradients to pass)
        self.log('test_loss', bce_loss, rank_zero_only=True, prog_bar=True, logger=True, sync_dist=True)

        ##### Add your analytics code here, use self.log() to put it in the loading bar! (can be same as val / more)

        
        # Return dictionary (as many elements as you'd like;)
        return {'test_loss': bce_loss.item(), 'logits': logits.cpu().numpy(), 'preds': predictions.cpu().numpy(), 
                'outputs': output.cpu().numpy(), 'gt_labels': y.cpu().numpy(), 'claim_id': claim_id}

    def test_epoch_end(self, outputs):
        # this is a list of dictionaries
        logits_list = [d['logits'] for d in outputs]
        preds_list = [d['preds'] for d in outputs]
        outputs_list = [d['outputs'] for d in outputs]
        gt_labels_list = [d['gt_labels'] for d in outputs]
        claim_id_list = [d['claim_id'] for d in outputs]

        print('saving numpy arrays as backup')
        np.save('{}/pred_outputs.npy'.format(self.output_dir), np.asarray(preds_list))
        np.save('{}/gt_labels.npy'.format(self.output_dir), np.asarray(gt_labels_list))
        np.save('{}/logits.npy'.format(self.output_dir), np.asarray(logits_list))
        np.save('{}/sigmoid_outputs.npy'.format(self.output_dir), np.asarray(outputs_list))
        np.save('{}/claim_ids.npy'.format(self.output_dir), np.asarray(claim_id_list))

    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        betas= (0.9, 0.95)
        base_batch = 2 # NEED TO CONFIRM WHAT THIS IS: 
        # https://github.com/tractableai/resnet-sandbox/blob/experiment/large_models3/experiments/hugh.tomkins/2021_09_capacity_experiments/9_deployed/train_reference_deployable.py#L63
        lr = ((0.001 * self.device_count) * base_batch) / 8
        min_lr = 0.0001
        steps = self.trainer.total_training_steps
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=betas)
        optimizer = torch.optim.SGD(self.parameters(), lr = lr, momentum=0.99, nesterov=True)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9) #build_scheduler(optimizer, self.config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=min_lr)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
