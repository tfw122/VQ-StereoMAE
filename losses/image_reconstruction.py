import torch
import torch.nn as nn
import torch.nn.functional as F
from imports.registry import registry
import lpips
import copy
from torchvision import models
import timm
from timm.models.vision_transformer import PatchEmbed
from losses.dall_e.dvae import Dalle_VAE

from losses.torch_utils import training_stats
from losses.torch_utils import misc
from losses.torch_utils.ops import conv2d_gradfix

@registry.register_loss('mae_loss')
class MaskedImageLoss(nn.Module):
    def __init__(self, config, patch_embed):
        super(MaskedImageLoss, self).__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config = self.config.dataset_config
        self.image_loss_weightings=  self.model_config.image_loss_weightings
        self.gan_arch_type=  self.model_config.discriminator.gan_arch_type
    
        
        self.scales = 4 
        # will be used in future for experimenting with different reconstruction losses
        self.loss_type = self.model_config.loss_type 
        
        # std (default) loss type is MSE based loss from the original paper.
        if self.loss_type=='ssim' or self.loss_type=='ms_ssim':
            self.ssim_loss = SSIM(self.config)
        
        elif self.loss_type=='perceptual':
            self.perceptual_loss = VanillaPerceptualLoss(self.config)
        
        elif self.loss_type=='lpips':
            self.lpips_loss = LPIPS(self.config)


        self.norm_pix_loss = self.model_config.norm_pix_loss
        
        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        

        self.patch_embed = patch_embed #PatchEmbed(img_size, patch_size, in_channels, embed_dim)

    
    def forward(self, imgs, pred, mask, epoch=None):
        """
        imgs: [N, 3, H, W] >>>>>> GT
        pred: [N, L, p*p*3] >>>>>>> PREDICTIONS
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # OG Mean Square Error:
        left_img, right_img = imgs
        left_pred, right_pred = pred
        left_mask, right_mask = mask
        
        if self.loss_type=='mae':
            left_target = self.patchify(left_img)
            right_target = self.patchify(right_img)

            # implements original https://arxiv.org/abs/2111.06377 loss
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5


            loss_left = (left_pred - left_target) ** 2
            loss_right = (right_pred - right_target) ** 2
            loss_left = loss_left.mean(dim=-1)  # [N, L], mean loss per patch
            loss_right = loss_right.mean(dim=-1)  # [N, L], mean loss per patch

            loss_left = (loss_left * left_mask).sum() / left_mask.sum()  # mean loss on removed patches
            loss_right = (loss_right * right_mask).sum() / right_mask.sum()  # mean loss on removed patches

        # SSIM + L1 Loss:
        elif self.loss_type=='ssim':
            # single scale structural similarity index loss with l1
            left_pred = self.unpatchify(left_pred)
            right_pred = self.unpatchify(right_pred)
            
            loss_left = self.ssim_loss(left_pred, left_img, self.scales)
            loss_right = self.ssim_loss(right_pred, right_img, self.scales)
        
        # L1 + Style + Perceptual Loss:
        elif self.loss_type=='perceptual':
            # has to be paired with l1 or equivalent! otherwise; results will be blocky.
            
            left_pred = self.unpatchify(left_pred)
            right_pred = self.unpatchify(right_pred)

            loss_dict_left= self.perceptual_loss(left_pred, left_img)
            loss_dict_right= self.perceptual_loss(right_pred, right_img)

            # filter out dictionary with relevant keys;
            dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])

            # the keys and coeffs are the same for left and right
            perc_dict_weightings = dictfilt(self.image_loss_weightings, loss_dict_left.keys())
            # calculate loss and multiply by weightings
            loss_left, loss_right=0.0, 0.0
            for key, coef in perc_dict_weightings.items():
                value_left = coef * loss_dict_left[key]
                value_right = coef * loss_dict_right[key]
                loss_left += value_left
                loss_right += value_right
            
        # L1 + LPIPS Loss:
        elif self.loss_type=='lpips':
            left_pred = self.unpatchify(left_pred)
            right_pred = self.unpatchify(right_pred)

            loss_left = self.lpips_loss(left_pred, left_img)
            loss_right = self.lpips_loss(right_pred, right_img)
        
        loss = loss_left + loss_right

        return loss

    # ---------------- helper functions ------------------
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


@registry.register_loss('lpips')
class LPIPS(nn.Module):
    def __init__(self, config):
        super(LPIPS, self).__init__()
        """
        from: https://arxiv.org/abs/1801.03924
        """
        self.config = config
        self.loss_fn = lpips.LPIPS(net=self.config.model_config.loss_network) # best forward scores
        # i.e. vgg / alex etc.
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, gt):
        # both images should be RGB normalised to -1 to 1
        l1_loss = self.l1(pred, gt)
        lpips_loss = self.loss_fn(pred, gt)
        loss = l1_loss + lpips_loss

        return loss


@registry.register_loss('perceptual_and_style')
class VanillaPerceptualLoss(nn.Module):
    """
    from: https://arxiv.org/abs/1603.08155
    This loss gives you both Perceptual and Style Transfer loss
    as a dictionary output
    choice is yours whether to use both or just one.
    """
    def __init__(self, config):
        super(VanillaPerceptualLoss, self).__init__()
        self.config = config

        if self.config.model_config.feature_extractor=='dall_e':
            self.feat_extractor = DALLEFeatureExtractor(self.config)
            self.blocks=4
            print('using DALL-E encoder as feature extractor')
        else:
            self.feat_extractor = VGG16FeatureExtractor()
            self.blocks=3
            print('using VGG16 as feature extractor')

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        print('''FOR STYLE LOSS, TRAIN IN FULL PRECISION (FP32) NOT HALF PREDICION (FP16) \n
                 otherwise gram matrix calculation will result in inf values and loss will be nan''')

    def forward(self, pred, gt):
        losses={}

        losses['l1'] = self.l1(pred, gt)
        
        if pred.shape[1] == 3:
            feat_output = self.feat_extractor(pred)
            feat_gt = self.feat_extractor(gt)
        elif pred.shape[1] == 1:
            feat_output = self.feat_extractor(torch.cat([pred]*3, 1))
            feat_gt = self.feat_extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('input shape must be either a RGB image or GrayScale')

        # get perceptual loss;
        losses['perc'] = 0.0
        # We extracted feature maps from 3 scales of the VGG network;
        for i in range(self.blocks):
            losses['perc'] += self.l1(feat_output[i], feat_gt[i])

        losses['style'] = 0.0
        for i in range(self.blocks):
            losses['style'] += self.l1(gram_matrix(feat_output[i]),
                                       gram_matrix(feat_gt[i]))

        return losses
            
@registry.register_loss('psnr')
class PSNR(nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self, config):
        super(PSNR, self).__init__()
        self.config = config

    @staticmethod
    def __call__(pred, gt):
        mse = torch.mean((pred - gt) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

@registry.register_loss('ssim')
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, config):
        super(SSIM, self).__init__()
        self.config= config
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.ssim_w = 0.85

    def ssim(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


    def forward(self, pred, gt, scales):
        #get multi scales
        pred_pyramid = scale_pyramid(pred, scales)
        gt_pyramid = scale_pyramid(gt, scales)
        # calculate L1 loss:
        l1_loss = [torch.mean(torch.abs(pred_pyramid[i] - gt_pyramid[i])) for i in range(scales)]
        # calculate SSIM loss
        ssim_loss = [torch.mean(self.ssim(pred_pyramid[i], gt_pyramid[i])) for i in range(scales)]
        # combine SSIM and L1: (0.85 * SSIM Loss) + (0.15 * L1 Loss)
        image_loss = [self.ssim_w * ssim_loss[i] + (1 - self.ssim_w) * l1_loss[i] for i in range(scales)]
        # sum all loss tensors (For all the scales)
        image_loss = sum(image_loss)

        return image_loss


# ---------------------------- Helper Functions ----------------------------
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class DALLEFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(DALLEFeatureExtractor, self).__init__()
        self.config = config
        # loads the full dall_e model; encoder + decoder
        self.dall_e = Dalle_VAE(self.config)
        # select the encoder only for the feature extractor
        self.encoder = self.dall_e.encoder
        self.input_layer = self.encoder.blocks.input
        self.enc_1 = self.encoder.blocks.group_1
        self.enc_2 = self.encoder.blocks.group_2
        self.enc_3 = self.encoder.blocks.group_3
        self.enc_4 = self.encoder.blocks.group_4

        # fix the encoder
        for i in range(4):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [self.input_layer(image)]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    s = img.size()
    h = s[2]
    w = s[3]
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        scaled_imgs.append(nn.functional.interpolate(img,
                            size=[nh, nw], mode='bilinear',
                            align_corners=True))
    return scaled_imgs