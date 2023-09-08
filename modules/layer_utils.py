import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as torchvision_models  # vision models lib for image encoders
import timm # vision transformers lib for image encoders
# hugging face lib for language encoders
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from omegaconf import OmegaConf

def build_image_encoder(arch='resnet18', pretrained=True):
    # get all the model names in torchvision library
    torchvision_model_names = sorted(name for name in torchvision_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(torchvision_models.__dict__[name]))

    # Get all the model names in pytorch image models (timm) library
    timm_model_names = sorted(timm.list_models(pretrained=True))

    # by default timm is used, however this can be changed should you choose to use torchvision;
    if arch in timm_model_names:
        # set num_classes to 0 to get the penuiltimate layer outputs; or physically change last layer in the encoder model;
        enc_model = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        
    elif arch in torchvision_model_names:
        enc_model = torch.hub.load('pytorch/vision', arch, pretrained=pretrained)
    else:
        raise Exception('the following model architecture: {} has not been implemented'.format(arch))
    
    return enc_model


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_channels = in_chans
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


def get_named_dict(model):
    named_layers = dict(model.named_modules())
    print(named_layers)
    return named_layers


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=='__main__':
    import timm 

    model= build_image_encoder('resnet50', pretrained=True)