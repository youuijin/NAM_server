import  torch
import  numpy as np

from torchvision import models
import advertorch.attacks as attacks
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import random
from datetime import datetime

attack_list=["PGD_L1", "PGD_L2", "PGD_Linf", "FGSM", "BIM_L2", "BIM_Linf", "MI-FGSM", "CnW", "DDN", "EAD", "Single_pixel", "DeepFool"]

def setAttack(str_at, net, eps, args):
    e = eps/255.
    iter = args.iter
    if str_at == "PGD_L1":
        return attacks.L1PGDAttack(net, eps=e, nb_iter=iter)
    elif str_at == "PGD_L2":
        return attacks.L2PGDAttack(net, eps=e, nb_iter=iter)
    elif str_at == "PGD_Linf":
        return attacks.LinfPGDAttack(net, eps=e, nb_iter=iter, clip_max=1.0, clip_min=-1.0)
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
    else:
        print("wrong type Attack")
        exit()

def setModel(str, n_way, imgsz, imgc, pretrained='IMAGENET1K_V1'):
    str = str.lower()
    if str=="resnet18":
        model = models.resnet18(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet34":
        model = models.resnet34(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet50":
        model = models.resnet50(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(imgc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet101":
        model = models.resnet101(weights=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
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
        model.classifier._modules["1"] = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.features._modules["0"]._modules["0"] = torch.nn.Conv2d(imgc, 32, kernel_size=3, stride=2, padding=1, bias=False)
        return model
    else:
        print("wrong model")
        print("possible models : resnet18, resnet34, resnet50, resnet101, resnet152, alexnet, densenet121, mobilenet_v2")
        exit()

def set_seed(seed=706):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_optim(args, model):
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    adv_optim = torch.optim.SGD(model.parameters(), lr=args.lr * args.lr_ratio, momentum=0.9, weight_decay=5e-4)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = 100)
    # adv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adv_optim, T_max = 100)

    if args.sche == 'lambda98':
        lamb = 0.98
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: lamb**epoch)
        adv_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=adv_optim, lr_lambda=lambda epoch: lamb**epoch)
    elif args.sche == 'lambda95':
        lamb = 0.95
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: lamb**epoch)
        adv_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=adv_optim, lr_lambda=lambda epoch: lamb**epoch)
    elif args.sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(args.epoch))
        adv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adv_optim, T_max=int(args.epoch))
    elif args.sche == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(args.epoch*0.2), gamma=0.5)
        adv_scheduler = torch.optim.lr_scheduler.StepLR(adv_optim, step_size=int(args.epoch*0.2), gamma=0.5)
    elif args.sche == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[int(args.epoch*0.5), int(args.epoch*0.8)], gamma=0.1) 
        adv_scheduler = torch.optim.lr_scheduler.MultiStepLR(adv_optim, milestones=[int(args.epoch*0.5), int(args.epoch*0.8)], gamma=0.1) 
    elif args.sche == 'no':
        lamb = 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: lamb**epoch)
        adv_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=adv_optim, lr_lambda=lambda epoch: lamb**epoch)
    else:
        print("Wrong Learning rate Scheduler")
        exit()
    
    return optim, adv_optim, scheduler, adv_scheduler

def writer(args, log_dir):
    approx_writer, adv_writer = None, None
    cur = datetime.now().strftime('%m%d_%H%M%S')

    if args.train_attack!="":
        log_name = f"{args.sche}/{args.train_attack}/{args.train_eps}/{args.lr}_{args.lr_ratio}/{cur}"
        if args.train_attack == 'QAUB':
            log_name = f"{args.sche}/{args.train_attack}/step{args.step}/{args.train_eps}/{args.lr}_{args.lr_ratio}/{cur}"
        writer = SummaryWriter(f"./{log_dir}/{log_name}")
        # if args.train_attack=="aRUB" or args.train_attack=="QAUB":
        #     approx_writer = SummaryWriter(f"./{log_dir}/approx_loss")
        #     adv_writer = SummaryWriter(f"./{log_dir}/adv_loss")
    else:
        log_name = f"{args.sche}/no_attack/{args.lr}/{cur}"
        writer = SummaryWriter(f"./{log_dir}/{log_name}")

    return writer, approx_writer, adv_writer, log_name

def predict(x, y, model):
    logit = model(x)
    pred = F.softmax(logit, dim=1)
    outputs = torch.argmax(pred, dim=1)
    correct_count = (outputs == y).sum().item()
    loss = F.cross_entropy(logit, y)

    return correct_count, loss

def adv_predict(args, at_type, at_bound, x, y, model, mode='train'):
    if at_type=='aRUB' or at_type=='QAUB':
        adv_loss = approx_loss(args, x, y, model, bound=False)
    else:
        at = setAttack(at_type, model, at_bound, args)
        advx = at.perturb(x, y)
        correct_count, adv_loss = predict(advx, y, model)

    if mode=='train':
        return adv_loss
    else:
        return correct_count, adv_loss


def approx_loss(args, x, y, model, bound): 
    if args.train_attack == 'aRUB':
        at = setAttack('aRUB', model, args.train_eps, args)
        logit_adv = at.perturb(x, y)
        approx_loss = F.cross_entropy(logit_adv, y, reduction='none')
        
    elif args.train_attack == 'QAUB':
        at = QAUB(args)
        approx_loss = at.approx_loss(model, x, y)
    
    # if bound:
    return approx_loss.sum()
        
    
    

def bound_rate():
    pass # TODO

def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input