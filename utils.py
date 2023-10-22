import  torch, os
import  numpy as np
from    torch.utils.data import DataLoader
from tqdm import tqdm

import time

from torchvision import models, transforms, datasets
import advertorch.attacks as attacks
from aRUBattack import aRUB

import pickle

from PIL import Image

attack_list=["PGD_L1", "PGD_L2", "PGD_Linf", "FGSM", "BIM_L2", "BIM_Linf", "MI-FGSM", "CnW", "DDN", "EAD", "Single_pixel", "DeepFool"]

def setAttack(str_at, net, eps, args):
    e = eps/255.
    iter = args.iter
    if str_at == "PGD_L1":
        return attacks.L1PGDAttack(net, eps=e, nb_iter=iter)
    elif str_at == "PGD_L2":
        return attacks.L2PGDAttack(net, eps=e, nb_iter=iter)
    elif str_at == "PGD_Linf":
        return attacks.LinfPGDAttack(net, eps=e, nb_iter=iter)
    elif str_at == "FGSM":
        return attacks.GradientSignAttack(net, eps=e)
    elif str_at == "BIM_L2":
        return attacks.L2BasicIterativeAttack(net, eps=e, nb_iter=iter)
    elif str_at == "BIM_Linf":
        return attacks.LinfBasicIterativeAttack(net, eps=e, nb_iter=iter)
    elif str_at == "MI-FGSM":
        return attacks.MomentumIterativeAttack(net, eps=e, nb_iter=iter) # 0.3, 40
    elif str_at == "CnW":
        return attacks.CarliniWagnerL2Attack(net, args.n_way, max_iterations=iter)
    elif str_at == "EAD":
        return attacks.ElasticNetL1Attack(net, args.n_way, max_iterations=iter)
    elif str_at == "DDN":
        return attacks.DDNL2Attack(net, nb_iter=iter)
    elif str_at == "Single_pixel":
        return attacks.SinglePixelAttack(net, max_pixels=iter)
    elif str_at == "DeepFool":
        return attacks.DeepfoolLinfAttack(net, args.n_way, eps=e, nb_iter=iter)
    elif str_at == "aRUB":
            return aRUB(net, rho=e, q=1, n_way=args.n_way, imgc=args.imgc, imgsz=args.imgsz)
    else:
        print("wrong type Attack")
        exit()

def setModel(str, n_way, imgsz, imgc, pretrained='IMAGENET1K_V1'):
    str = str.lower()
    if str=="resnet18":
        model = models.resnet18(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_way)
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet34":
        model = models.resnet34(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_way)
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet50":
        model = models.resnet50(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_way)
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet101":
        model = models.resnet101(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_way)
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet152":
        model = models.resnet152(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_way)
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="alexnet":
        model = models.alexnet(weights=pretrained)
        num_ftrs = model.classifier._modules["6"].in_features
        model.classifier._modules["6"] = torch.nn.Linear(num_ftrs, n_way)
        model.features._modules["0"] = torch.nn.Conv2d(imgc, 64, kernel_size=11, stride=4, padding=2, bias=False)
        return model
    elif str=="densenet121":
        model = models.densenet121(weights=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, n_way)
        return model
    elif str=="mobilenet_v2":
        model = models.mobilenet_v2(weights=pretrained)
        num_ftrs = model.classifier._modules["1"].in_features
        model.classifier._modules["1"] = torch.nn.Linear(num_ftrs, n_way)
        model.features._modules["0"]._modules["0"] = torch.nn.Conv2d(imgc, 32, kernel_size=3, stride=2, padding=1, bias=False)
        return model
    else:
        print("wrong model")
        print("possible models : resnet18, resnet34, resnet50, resnet101, resnet152, alexnet, densenet121, mobilenet_v2")
        exit()
