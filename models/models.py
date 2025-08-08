import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from .clip_text import CLIP_Text
from .peft_vit import Peft_ViT, ViT_Tuner, ImageEncoder
from .peft_rn import Peft_RN, RN_Tuner
from .peft_modules import PromptLearner
from .classifiers import *

from .peft_modules import *


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Linear(inplanes, planes)
        # self.conv11 = nn.Linear(inplanes, planes)
        # self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(planes, inplanes)
        

        self.conv_temp = nn.Linear(1000, inplanes)
        self.gate_layer = nn.Linear(planes, 1)
        # self.bn2 = norm_layer(planes)

    def forward(self, x, logits):
        # identity = x

        out = self.conv1(x) + self.conv_temp(logits)
        out = self.relu(out)
        out = F.dropout(out, training=self.training)
        gate = self.gate_layer(out)
        out = self.conv2(out)

        # out = identity
        # out = self.relu(out)

        return out, gate


class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        self.dtype = clip_model.dtype

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features

    def encode_image(self, image, return_before_proj=False):
        return self.image_encoder(image.to(self.dtype), return_before_proj)
    
    @torch.no_grad()
    def init_text_features(self, prompts):
        text_features = self.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        self.text_features = text_features
    
    def forward(self, image, return_feature=False):
        if return_feature:
            image_features = self.encode_image(image, return_before_proj=True)
            return image_features
        
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logit = self.logit_scale * F.linear(image_features, self.text_features)
        # if return_feature:
        #     return image_features
        return logit
    
    
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        
        self.head = None
        if cfg.dual_prompt:
            self.clip_prompter = PromptLearner(cfg, classnames, clip_model)
            clip_prompter = self.clip_prompter
        else:
            clip_prompter = PromptLearner(cfg, classnames, clip_model)

        # clip_prompter = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = clip_prompter.tokenized_prompts

        if cfg.dual_prompt :
            self.image_encoder = ImageEncoder(clip_model)
            self.tuner = clip_prompter
        else:
            self.image_encoder = Peft_ViT(clip_model.visual)
            self.tuner = ViT_Tuner(cfg, clip_model.visual, len(classnames))
            self.text_tuner = clip_prompter

        self.text_encoder = CLIP_Text(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg


    def forward(self, image, use_tuner=True, return_feature=False):
        if self.cfg.dual_prompt:
            visual_prompts = self.tuner.forward_vis
            image_features = self.image_encoder(image.type(self.dtype), visual_prompts, None, proj=True)
        else:
            tuner=self.tuner
            image_features = self.image_encoder(image, tuner, None, proj=True)

        if return_feature:
            return image_features
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_prompts = self.text_tuner.forward_txt()
        text_features = self.text_encoder.prompt_forward(text_prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # text_features = text_features @ self.image_encoder.proj.t()

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        return logits#, image_features, text_features


class PeftModelFromCLIP(nn.Module):
    def __init__(self, cfg, clip_model, num_classes):
        super().__init__()

        if cfg.backbone.startswith("CLIP-ViT"):
            self.text_encoder = CLIP_Text(clip_model)
            self.image_encoder = Peft_ViT(clip_model.visual)
            self.tuner = ViT_Tuner(cfg, clip_model.visual, num_classes)
        elif cfg.backbone.startswith("CLIP-RN"):
            self.text_encoder = CLIP_Text(clip_model)
            self.image_encoder = Peft_RN(clip_model.visual)
            self.tuner = RN_Tuner(cfg, clip_model.visual, num_classes)
        
        self.coarse_peft = None
        self.coarse_head = None
        dtype = self.image_encoder.dtype
        feat_dim = self.image_encoder.out_dim
        self.elimate_module = None

        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features
    
    def forward(self, image, use_tuner=True, return_feature=False, txt_feat=None):
        tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        return self.image_encoder(image, tuner, head, input_coarse_list=None, coarse_head=None)


class PeftModelFromViT(nn.Module):
    def __init__(self, cfg, vit_model, num_classes):
        super().__init__()

        if cfg.backbone.startswith("IN21K-ViT"):
            self.image_encoder = Peft_ViT(vit_model)
            self.tuner = ViT_Tuner(cfg, vit_model, num_classes)
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)

    def forward(self, image, use_tuner=True, return_feature=False):
        tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        return self.image_encoder(image, tuner, head)
