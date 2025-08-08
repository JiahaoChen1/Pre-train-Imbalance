import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224

import datasets
from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator, knn_predict
from utils.templates import ZEROSHOT_TEMPLATES

# from torchtoolbox.transform import Cutout
# from utils.evaluator import Evaluator, 

import open_clip 

def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(backbone_name, prec):
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=True).eval()

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model


def load_openclip_to_cpu(backbone_name, prec):
    if backbone_name == "CLIP-ViT-B/16":
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16')
        # model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained="/home/jiahao_chen/.cache/huggingface/hub/CLIP-ViT-B-16-CommonPool.L-s1B-b8K/open_clip_pytorch_model.bin")
    model = model.eval()
    state_dict = None
    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model

def load_metaclip_to_cpu(backbone_name, prec):
    if backbone_name == "CLIP-ViT-B/16":
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='metaclip_400m')
    model = model.eval()
    state_dict = None
    # file = model.state_dict()
    # new_state_dict = {}
    # for k, v in model.state_dict():
    #     new_k = k.replace('text_model', 'transformer')
    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


class Tester:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.build_data_loader()

        self.build_model()
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None

        

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            # Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # transforms.RandomErasing(value='random', scale=(0.1, 0.33)),
            
        ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        if cfg.tte:
            if cfg.tte_mode == "fivecrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.FiveCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "tencrop":
                transform_test = transforms.Compose([
                    transforms.Resize(resolution + expand),
                    transforms.TenCrop(resolution),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Normalize(mean, std),
                ])
            elif cfg.tte_mode == "randaug":
                _resize_and_flip = transforms.Compose([
                    transforms.RandomResizedCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.Lambda(lambda image: torch.stack([_resize_and_flip(image) for _ in range(cfg.randaug_times)])),
                    transforms.Normalize(mean, std),
                ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])

        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
            split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
        else:
            split_cls_num_list = self.cls_num_list
        self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
        self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
        self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        self.train_loader = DataLoader(train_dataset,
            batch_size=cfg.micro_batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_init_loader = DataLoader(train_init_dataset,
            batch_size=512, sampler=init_sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.train_test_loader = DataLoader(train_test_dataset,
            batch_size=512, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
            batch_size=512, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
        
        if self.cfg.zero_shot or self.cfg.test_only or self.cfg.test_train:
            if 'iNat' not in self.cfg.dataset:
                val_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test, val=True)
                self.val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        
                self.knn_loader =  self.val_loader
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size

        print("Total training points:", sum(self.cls_num_list))
        # print(self.cls_num_list)

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.backbone})")
        clip_model1 = load_openclip_to_cpu(cfg.backbone, cfg.prec)
        print('******use open clip*******')
        clip_model2 = load_metaclip_to_cpu(cfg.backbone, cfg.prec)
        print('******use meta clip*******')
        clip_model3 = load_clip_to_cpu(cfg.backbone, cfg.prec)

        self.clip_model1 = PeftModelFromCLIP(cfg, clip_model1, num_classes)
        self.clip_model2 = PeftModelFromCLIP(cfg, clip_model2, num_classes)
        self.clip_model3 = PeftModelFromCLIP(cfg, clip_model3, num_classes)

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader
        elif mode == 'val':
            print(f"Evaluate on the val set")
            data_loader = self.val_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            # if self.cfg.coarse:
            #     coarse_label = self.indices_map[label]

            if _ncrops <= 5:

                output = self.model(image)
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    output.append(self.model(image[:, k]))
                output = torch.stack(output).mean(dim=0)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]

    @torch.no_grad()
    def backdoor_adjustment(self, mode="test"):
        # if self.tuner is not None:
        #     self.tuner.eval()
        # if self.head is not None:
        #     self.head.eval()
        self.clip_model1.eval().to(self.device)
        self.clip_model2.eval().to(self.device)
        self.clip_model3.eval().to(self.device)
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader
        elif mode == 'val':
            print(f"Evaluate on the val set")
            data_loader = self.val_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            # if self.cfg.coarse:
            #     coarse_label = self.indices_map[label]

            if _ncrops <= 5:

                output1 = self.clip_model1(image)
                output2 = self.clip_model2(image)
                output3 = self.clip_model3(image)
                output = output1 + output2 + output3
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    output.append(self.model(image[:, k]))
                output = torch.stack(output).mean(dim=0)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]


    def load_model(self, directory_clip, directory_openclip, directory_metaclip):
        load_path_clip = os.path.join(directory_clip, "checkpoint.pth.tar")
        load_path_openclip = os.path.join(directory_openclip, "checkpoint.pth.tar")
        load_path_metaclip = os.path.join(directory_metaclip, "checkpoint.pth.tar")


        checkpoint_clip = torch.load(load_path_clip, map_location=self.device)
        checkpoint_openclip = torch.load(load_path_openclip, map_location=self.device)
        checkpoint_metaclip = torch.load(load_path_metaclip, map_location=self.device)


        self.clip_model3.tuner.load_state_dict(checkpoint_clip["tuner"])
        self.clip_model2.tuner.load_state_dict(checkpoint_metaclip["tuner"])
        self.clip_model1.tuner.load_state_dict(checkpoint_openclip["tuner"])

        self.clip_model3.head.load_state_dict(checkpoint_clip["head"])
        self.clip_model2.head.load_state_dict(checkpoint_metaclip["head"])
        self.clip_model1.head.load_state_dict(checkpoint_openclip["head"])
