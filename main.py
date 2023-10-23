import  torch
import  numpy as np
import  random
import  argparse

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import utils
from NAMloss import NAMLoss

def main(args, set_k=None):
    seed = 706
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:'+str(args.device_num))

    if set_k!=None:
        args.NAM_k = set_k
        args.lr_ratio = 1
        # if set_k<0:
        #     args.NAM_k = -1
        #     args.lr_ratio = -1*set_k
        # else:
        #     args.NAM_k = 1
        #     args.lr_ratio = set_k


    model = utils.setModel(args.model, args.n_way, args.imgsz, args.imgc, pretrained=None)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    transform = transforms.Compose([transforms.Resize((args.imgsz, args.imgsz)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0, 0, 0), (1, 1, 1))])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    train_size = int(len(train_data) * 0.8) # 80% training data
    valid_size = len(train_data) - train_size # 20% validation data
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
    
    print(len(train_data), len(valid_data), len(test_data))
    
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    adv_optim = torch.optim.Adam(model.parameters(), lr=args.lr*args.lr_ratio)
    
    if args.NAM:
        writer = SummaryWriter(f"./loss_log/{args.model}/NAM_{round(args.NAM_k*args.lr_ratio, 1)}_{args.lr}")
    elif args.train_attack!="":
        writer = SummaryWriter(f"./loss_log/{args.model}/{args.train_attack}_{args.train_eps}_{args.lr}_{args.lr_ratio}")
    else:
        writer = SummaryWriter(f"./loss_log/{args.model}/no_attack_{args.lr}")

    best_val = 0
    best_val_adv = 0
    last_val = 0
    last_val_adv = 0
    bound_rate = [0 for i in range(args.epoch)] # epoch마다 저장
    bound_rate_elements = [0 for i in range(args.epoch)] # epoch마다 저장

    for epoch in range(args.epoch):
        db = torch.utils.data.DataLoader(train_data, batch_size=args.task_num, shuffle=True, num_workers=2)
        correct_count=0
        distance = 0
        model.train()
        train_losses = []
        train_losses_adv = []
        val_losses = []
        val_losses_adv = []
        for _, (x, y) in enumerate(tqdm(db, desc="train")):
            x = x.to(device)
            y = y.to(device)
            logit = model(x)
            pred = torch.nn.functional.softmax(logit, dim=1)
            outputs = torch.argmax(pred, dim=1)
            correct_count += (outputs == y).sum().item()
            loss = torch.nn.functional.cross_entropy(logit, y)
            train_losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            # adv_correct_count = 0
            if args.train_attack!="" or args.NAM:
                if args.NAM:
                    loss_fn = NAMLoss(model, args.NAM_k)
                    loss = loss_fn.get_loss(x, y, sum=False, reduction=True)
                else:
                    at = utils.setAttack(args.train_attack, model, args.train_eps, args)
                    if(args.train_attack=="aRUB"):
                        logit = at.perturb(x, y)
                    else:
                        advx = at.perturb(x, y)
                        logit = model(advx)
                    pred = torch.nn.functional.softmax(logit, dim=1)
                    outputs = torch.argmax(pred, dim=1)
                    # adv_correct_count += (outputs == y).sum().item()
                    loss = torch.nn.functional.cross_entropy(logit, y)
                train_losses_adv.append(loss.item())
                adv_optim.zero_grad()
                loss.backward()
                adv_optim.step()

                if(args.train_attack=="aRUB" or args.NAM):
                    origin_at = utils.setAttack(args.test_attack, model, 2.0, args)
                    advx = origin_at.perturb(x, y)
                    logit_origin = model(advx)
                    loss_origin = torch.nn.functional.cross_entropy(logit_origin, y, reduction='none')
                    if args.NAM:
                        loss_not_batched = loss_fn.get_loss(x, y, sum=True, reduction=False)
                    else:
                        loss_not_batched = torch.nn.functional.cross_entropy(logit, y, reduction='none')
                    # L2 norm
                    bound_rate_elements[epoch] += x.shape[0]
                    bound_rate[epoch] +=(loss_origin<=loss_not_batched).sum().item()

                    distance += torch.norm(loss_origin - loss_not_batched).item()

        if args.train_attack == "aRUB" or args.NAM:
            writer.add_scalar("L2 distance", distance, epoch)
            writer.add_scalar("bound_rate", round(bound_rate[epoch]/bound_rate_elements[epoch]*100,2), epoch)
        writer.add_scalar("train_acc", round(correct_count/len(train_data)*100, 2), epoch)
        # writer.add_histogram("train", np.array(train_losses), epoch)
        # writer_train.add_histogram(summary_name, np.array(train_losses), epoch)
        # if args.train_attack!="" or args.NAM:
        #     writer.add_histogram("train_adv", np.array(train_losses_adv), epoch)
            # writer_train_adv.add_histogram(summary_name, np.array(train_losses_adv), epoch)
        
        model.eval()
        db_val = torch.utils.data.DataLoader(valid_data, batch_size=args.task_num, shuffle=True, num_workers=0)
        val_correct_count=0
        for _, (x, y) in enumerate(tqdm(db_val, desc="val")):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                logit = model(x)
                pred = torch.nn.functional.softmax(logit, dim=1)
            outputs = torch.argmax(pred, dim=1)
            val_correct_count += (outputs == y).sum().item()
            loss = torch.nn.functional.cross_entropy(logit, y)
            val_losses.append(loss.item())
        val_adv_correct_count=0
        at = utils.setAttack(args.test_attack, model, args.test_eps, args)
        for _, (x, y) in enumerate(tqdm(db_val, desc="adv val")):
            x = x.to(device)
            y = y.to(device)
            advx = at.perturb(x, y)
            logit = model(advx)
            pred = torch.nn.functional.softmax(logit, dim=1)
            outputs = torch.argmax(pred, dim=1)
            val_adv_correct_count += (outputs == y).sum().item()
            loss = torch.nn.functional.cross_entropy(logit, y)
            val_losses_adv.append(loss.item())

        # writer.add_histogram("val", np.array(val_losses), epoch)
        writer.add_scalar("val_loss", np.array(val_losses).mean(), epoch)
        # if args.train_attack!="" or args.NAM:
        #     writer.add_histogram("val_adv", np.array(val_losses_adv), epoch)
        
        last_val = round(val_correct_count/len(valid_data)*100, 2)
        last_val_adv = round(val_adv_correct_count/len(valid_data)*100, 2)
        if last_val > best_val:
            best_val = last_val
        if last_val_adv > best_val_adv:
            best_val_adv = last_val_adv

        # writer.add_scalar("train_adv_acc", round(adv_correct_count/len(train_data)*100, 2), epoch)
        writer.add_scalar("val_acc", last_val, epoch)
        writer.add_scalar("val_adv_acc", last_val_adv, epoch)
        
        print("epoch: ", epoch, "\ttraining acc:", round(correct_count/len(train_data)*100, 2))
        print("val acc:", last_val, "\tval adv acc:", last_val_adv)

    if args.train_attack == "aRUB" or args.NAM:
        bound_rate_value = round(100.0* np.array(bound_rate).sum()/np.array(bound_rate_elements).sum(), 2)
    else:
        bound_rate_value = 0
    
    return last_val, last_val_adv, best_val, best_val_adv, bound_rate_value

def iter_main(k):
    argparser = argparse.ArgumentParser()

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=112)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    
    # Training options
    argparser.add_argument('--model', type=str, help='type of model to use', default="resnet18")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=256)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0005)
    # argparser.add_argument('--adv_lr', type=float, help='adversarial learning rate', default=0.0001)
    argparser.add_argument('--lr_ratio', type=float, help='adv lr ratio', default=0.2)
    argparser.add_argument('--train_attack', type=str, help='attack for adversarial training', default="")
    argparser.add_argument('--train_eps', type=float, help='training attack bound', default=6)

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--test_eps', type=float, help='attack-eps', default=6)
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=2)
    argparser.add_argument('--iter', type=int, default=10)

    argparser.add_argument('--pretrained', type=str, help='path of pretrained model', default="")

    argparser.add_argument('--NAM', action='store_true', default=True)
    argparser.add_argument('--NAM_k', type=float, default=1)
    args = argparser.parse_args()

    return main(args, set_k=k)



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=112)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    
    # Training options
    argparser.add_argument('--model', type=str, help='type of model to use', default="")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=256)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0005)
    # argparser.add_argument('--adv_lr', type=float, help='adversarial learning rate', default=0.0001)
    argparser.add_argument('--lr_ratio', type=float, help='adv lr ratio', default=0.2)
    argparser.add_argument('--train_attack', type=str, help='attack for adversarial training', default="")
    argparser.add_argument('--train_eps', type=float, help='training attack bound', default=6)

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--test_eps', type=float, help='attack-eps', default=6)
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=2)
    argparser.add_argument('--iter', type=int, default=10)

    argparser.add_argument('--pretrained', type=str, help='path of pretrained model', default="")

    argparser.add_argument('--NAM', action='store_true', default=False)
    argparser.add_argument('--NAM_k', type=float, default=1)
    args = argparser.parse_args()

    main(args)