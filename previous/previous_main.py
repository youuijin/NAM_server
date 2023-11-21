import  torch
import  numpy as np
import  random
import  argparse

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from tqdm import tqdm
import time

from torch.utils.tensorboard import SummaryWriter

import utils

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

    # transform = transforms.Compose([transforms.Resize((args.imgsz, args.imgsz)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize((0, 0, 0), (1, 1, 1))])
    transform = transforms.Compose([transforms.RandomCrop(args.imgsz, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([transforms.Resize((args.imgsz, args.imgsz)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    train_size = int(len(train_data) * 0.8) # 80% training data
    valid_size = len(train_data) - train_size # 20% validation data
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform_test)
    
    print(len(train_data), len(valid_data), len(test_data))



    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    adv_optim = torch.optim.SGD(model.parameters(), lr=args.lr*args.lr_ratio, momentum=0.9, weight_decay=5e-4)

    print(args.sche)
    if args.sche == 'lambda98':
        lamb = 0.98
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: lamb**epoch)
        adv_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=adv_optim, lr_lambda=lambda epoch: lamb**epoch)
    elif args.sche == 'lambda95':
        print('in')
        lamb = 0.95
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: lamb**epoch)
        adv_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=adv_optim, lr_lambda=lambda epoch: lamb**epoch)
    elif args.sche == 'cosine':
        T = 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=T)
        adv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adv_optim, T_max=T)
        args.sche += '_'+str(T)
    elif args.sche == 'step':
        size = 20
        gamm = 0.5
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=size, gamma=gamm)
        adv_scheduler = torch.optim.lr_scheduler.StepLR(adv_optim, step_size=size, gamma=gamm)
        args.sche += '_'+str(size)+"_"+str(gamm)
    elif args.sche == 'multi_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [100, 150], 0.1)
        adv_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [100, 150], 0.1)
    else:
        lamb = 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: lamb**epoch)
        adv_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=adv_optim, lr_lambda=lambda epoch: lamb**epoch)

    log_dir = "logs"
    
    if args.train_attack!="":
        writer = SummaryWriter(f"./{log_dir}/{args.train_attack}_{args.train_eps}_{args.lr}_{args.lr_ratio}_{args.sche}")
        # if args.train_attack=="aRUB":
        #     approx_writer = SummaryWriter(f"./{log_dir}/approx_loss")
        #     adv_writer = SummaryWriter(f"./{log_dir}/adv_loss")
    else:
        writer = SummaryWriter(f"./{log_dir}/no_attack_{args.lr}_{args.sche}")

    best_val = 0
    best_val_adv = 0
    last_val = 0
    last_val_adv = 0

    all_time = time.time()
    train_time = 0
    
    for epoch in range(args.epoch):
        train_st = time.time()
        train_data.dataset.transform = transform
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
                train_loss_adv += loss.item()*x.shape[0]
                adv_optim.zero_grad()
                loss.backward()
                adv_optim.step()

                # if(args.train_attack=="aRUB"):
                #     origin_at = utils.setAttack("PGD_Linf", model, args.train_eps, args)
                #     advx = origin_at.perturb(x, y)
                #     logit_origin = model(advx)
                #     loss_origin = torch.nn.functional.cross_entropy(logit_origin, y, reduction='none')
                    
                #     approx_loss = torch.nn.functional.cross_entropy(logit_adv, y, reduction='none')
                #     approx_loss_sum = approx_loss.sum().item()
                #     loss_adv_sum = loss_origin.sum().item()

                #     bound_rate_elements += x.shape[0]
                #     bound_rate += (loss_origin<=approx_loss).sum().item()

        # train_time += time.time()-train_st
                    
        # if args.train_attack == "aRUB":
        #     writer.add_scalar("bound_rate", round(bound_rate/bound_rate_elements*100,4), epoch)
        #     approx_writer.add_scalar(f"aRUB_{args.train_eps}_{args.lr}/", round(approx_loss_sum/len(train_data), 4), epoch)
        #     adv_writer.add_scalar(f"aRUB_{args.train_eps}_{args.lr}/", round(loss_adv_sum/len(train_data), 4), epoch)

        writer.add_scalar("train/acc", round(train_correct_count/len(train_data)*100, 4), epoch)
        writer.add_scalar("train/loss", round(train_loss/len(train_data)*100, 4), epoch)
        writer.add_scalar("train/approx", round(train_loss_adv/len(train_data)*100, 4), epoch)
        
        # writer.add_histogram("train", np.array(train_losses), epoch)
        # writer_train.add_histogram(summary_name, np.array(train_losses), epoch)
        # if args.train_attack!="" or args.NAM:
        #     writer.add_histogram("train_adv", np.array(train_losses_adv), epoch)
            # writer_train_adv.add_histogram(summary_name, np.array(train_losses_adv), epoch)
        if epoch%5==0:
            valid_data.dataset.transform = transform_test
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

        # writer.add_scalar('lr', optim.param_groups[0]['lr'], epoch)
        scheduler.step()
        adv_scheduler.step()

        # last_val = round(val_correct_count/len(valid_data)*100, 2)
        # last_val_adv = round(val_adv_correct_count/len(valid_data)*100, 2)
        # if last_val > best_val:
        #     best_val = last_val
        # if last_val_adv > best_val_adv:
        #     best_val_adv = last_val_adv
        
        # print("epoch: ", epoch, "\ttraining acc:", round(val_correct_count/len(train_data)*100, 2))
        # print("val acc:", last_val, "\tval adv acc:", last_val_adv)

    # if args.train_attack == "aRUB" or args.NAM:
    #     bound_rate_value = round(100.0* np.array(bound_rate).sum()/np.array(bound_rate_elements).sum(), 2)
    # else:
    #     bound_rate_value = 0
    
    # return last_val, last_val_adv, best_val, best_val_adv, bound_rate_value
    # all_time = time.time() - all_time

    # writer.add_scalar("time/all", all_time, 0)
    # writer.add_scalar("time/train", train_time, 0)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=32)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    
    # Training options
    argparser.add_argument('--model', type=str, help='type of model to use', default="resnet18")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=128)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    # argparser.add_argument('--adv_lr', type=float, help='adversarial learning rate', default=0.0001)
    argparser.add_argument('--lr_ratio', type=float, help='adv lr ratio', default=0.5)
    argparser.add_argument('--train_attack', type=str, help='attack for adversarial training', default="")
    argparser.add_argument('--train_eps', type=float, help='training attack bound', default=6)

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--test_eps', type=float, help='attack-eps', default=6)
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=6)
    argparser.add_argument('--iter', type=int, default=10)

    argparser.add_argument('--sche', type=str, default="")

    args = argparser.parse_args()

    main(args)