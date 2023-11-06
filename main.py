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
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: 0.95**epoch)
    adv_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=adv_optim, lr_lambda=lambda epoch: 0.95**epoch)

    
    if args.train_attack!="":
        writer = SummaryWriter(f"./step_log/{args.train_attack}_{args.train_eps}_{args.lr}_{args.lr_ratio}")
        if args.train_attack=="aRUB":
            approx_writer = SummaryWriter("./step_log/approx_loss")
            adv_writer = SummaryWriter("./step_log/adv_loss")
    else:
        writer = SummaryWriter(f"./step_log/{args.model}/no_attack_{args.lr}")

    best_val = 0
    best_val_adv = 0
    last_val = 0
    last_val_adv = 0
    
    for epoch in range(args.epoch):
        train_db = torch.utils.data.DataLoader(train_data, batch_size=args.task_num, shuffle=True, num_workers=2)
        train_correct_count=0
        train_loss = 0
        train_loss_adv = 0
        approx_loss = 0

        bound_rate = 0
        bound_rate_elements = 0 

        model.train()

        for _, (x, y) in enumerate(tqdm(train_db, desc="train")):
            x = x.to(device)
            y = y.to(device)
            logit = model(x)
            pred = torch.nn.functional.softmax(logit, dim=1)
            outputs = torch.argmax(pred, dim=1)
            train_correct_count += (outputs == y).sum().item()
            loss = torch.nn.functional.cross_entropy(logit, y)
            train_loss += loss.item()*x.shape[0]
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            # adv_correct_count = 0
            if args.train_attack!="":
                at = utils.setAttack(args.train_attack, model, args.train_eps, args)
                if(args.train_attack=="aRUB"):
                    logit_adv = at.perturb(x, y)
                else:
                    advx = at.perturb(x, y)
                    logit_adv = model(advx)
                pred = torch.nn.functional.softmax(logit, dim=1)
                outputs = torch.argmax(pred, dim=1)
                loss = torch.nn.functional.cross_entropy(logit_adv, y, reduction='none').sum()
                # train_losses_adv.append(loss.sum().item())
                train_loss_adv += loss.item()*x.shape[0]
                adv_optim.zero_grad()
                loss.backward()
                adv_optim.step()

                if(args.train_attack=="aRUB"):
                    origin_at = utils.setAttack("PGD_Linf", model, args.train_eps, args)
                    advx = origin_at.perturb(x, y)
                    logit_origin = model(advx)
                    loss_origin = torch.nn.functional.cross_entropy(logit_origin, y, reduction='none')
                    
                    approx_loss = torch.nn.functional.cross_entropy(logit_adv, y, reduction='none')
                    approx_loss_sum = approx_loss.sum().item()
                    loss_adv_sum = loss_origin.sum().item()

                    bound_rate_elements += x.shape[0]
                    bound_rate += (loss_origin<=approx_loss).sum().item()

                    
        if args.train_attack == "aRUB":
            writer.add_scalar("bound_rate", round(bound_rate/bound_rate_elements*100,4), epoch)
            approx_writer.add_scalar(f"aRUB_{args.train_eps}_{args.lr}/", round(approx_loss_sum/len(train_data), 4), epoch)
            adv_writer.add_scalar(f"aRUB_{args.train_eps}_{args.lr}/", round(loss_adv_sum/len(train_data), 4), epoch)

        writer.add_scalar("train/acc", round(train_correct_count/len(train_data)*100, 4), epoch)
        writer.add_scalar("train/loss", round(train_loss/len(train_data)*100, 4), epoch)
        writer.add_scalar("train/approx", round(train_loss_adv/len(train_data)*100, 4), epoch)
        
        # writer.add_histogram("train", np.array(train_losses), epoch)
        # writer_train.add_histogram(summary_name, np.array(train_losses), epoch)
        # if args.train_attack!="" or args.NAM:
        #     writer.add_histogram("train_adv", np.array(train_losses_adv), epoch)
            # writer_train_adv.add_histogram(summary_name, np.array(train_losses_adv), epoch)
        
        model.eval()
        db_val = torch.utils.data.DataLoader(valid_data, batch_size=args.task_num, shuffle=True, num_workers=0)
        val_correct_count = 0
        val_adv_correct_count = 0
        val_loss = 0
        val_loss_adv = 0
        for _, (x, y) in enumerate(tqdm(db_val, desc="val")):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                logit = model(x)
                pred = torch.nn.functional.softmax(logit, dim=1)
            outputs = torch.argmax(pred, dim=1)
            val_correct_count += (outputs == y).sum().item()
            loss = torch.nn.functional.cross_entropy(logit, y)
            val_loss += loss.item()*x.shape[0]

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
            val_loss_adv += loss.item()*x.shape[0]

        writer.add_scalar("val/acc", round(val_correct_count/len(valid_data)*100, 4), epoch)
        writer.add_scalar("val/loss", round(val_loss/len(valid_data)*100, 4), epoch)
        writer.add_scalar("val/adv_acc", round(val_adv_correct_count/len(valid_data)*100, 4), epoch)
        writer.add_scalar("val/adv_loss", round(val_loss_adv/len(valid_data)*100, 4), epoch)

        scheduler.step()
        adv_scheduler.step()

        last_val = round(val_correct_count/len(valid_data)*100, 2)
        last_val_adv = round(val_adv_correct_count/len(valid_data)*100, 2)
        if last_val > best_val:
            best_val = last_val
        if last_val_adv > best_val_adv:
            best_val_adv = last_val_adv
        
        # print("epoch: ", epoch, "\ttraining acc:", round(val_correct_count/len(train_data)*100, 2))
        # print("val acc:", last_val, "\tval adv acc:", last_val_adv)

    # if args.train_attack == "aRUB" or args.NAM:
    #     bound_rate_value = round(100.0* np.array(bound_rate).sum()/np.array(bound_rate_elements).sum(), 2)
    # else:
    #     bound_rate_value = 0
    
    # return last_val, last_val_adv, best_val, best_val_adv, bound_rate_value


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=112)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    
    # Training options
    argparser.add_argument('--model', type=str, help='type of model to use', default="resnet18")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=256)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    # argparser.add_argument('--adv_lr', type=float, help='adversarial learning rate', default=0.0001)
    argparser.add_argument('--lr_ratio', type=float, help='adv lr ratio', default=0.5)
    argparser.add_argument('--train_attack', type=str, help='attack for adversarial training', default="")
    argparser.add_argument('--train_eps', type=float, help='training attack bound', default=6)

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--test_eps', type=float, help='attack-eps', default=6)
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=6)
    argparser.add_argument('--iter', type=int, default=10)

    args = argparser.parse_args()

    main(args)