import  torch
import  argparse
import csv

from torch.utils.data import random_split, DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from tqdm import tqdm
import time

import utils

def main(args):
    utils.set_seed()

    device = torch.device('cuda:'+str(args.device_num))

    model = utils.setModel(args.model, args.n_way, args.imgsz, args.imgc, pretrained=None)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # norm_mean, norm_std = (0, 0, 0), (1, 1, 1)
    norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    # norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([transforms.RandomCrop(args.imgsz, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])
    transform_val = transforms.Compose([transforms.Resize((args.imgsz, args.imgsz)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    train_data = CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_size = int(len(train_data) * 0.8) # 80% training data
    valid_size = len(train_data) - train_size # 20% validation data
    train_data, valid_data = random_split(train_data, [train_size, valid_size])
    test_data = CIFAR10(root='./data', train=False, download=False, transform=transform_val)

    optim, adv_optim, scheduler, adv_scheduler = utils.set_optim(args, model)
    
    log_dir = "logs"
    writer, _, _, log_name = utils.writer(args, log_dir)
    
    train_time = 0
    attack_time = 0

    last_val = 0
    last_val_adv = 0
    
    for epoch in range(args.epoch):
        train_data.dataset.transform = transform
        train_db = DataLoader(train_data, batch_size=args.task_num, shuffle=True, num_workers=2)
        
        train_correct_count = 0
        train_loss = 0
        train_loss_adv = 0

        model.train()

        train_time_st = time.time()
        for _, (x, y) in enumerate(tqdm(train_db, desc="train")):
            x = x.to(device)
            y = y.to(device)
            correct_count, loss = utils.predict(x, y, model)
            train_correct_count += correct_count
            train_loss += loss.item()*x.shape[0]
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            if args.train_attack!="":
                attack_time_st = time.time()
                loss_adv = utils.adv_predict(args, args.train_attack, args.train_eps, x, y, model, mode='train')

                attack_time += (time.time() - attack_time_st)
                train_loss_adv += loss_adv.item()
                adv_optim.zero_grad()
                loss_adv.backward()
                adv_optim.step()

        train_time += (time.time() - train_time_st)

        writer.add_scalar("train/acc", round(train_correct_count/len(train_data)*100, 4), epoch)
        writer.add_scalar("train/loss", round(train_loss/len(train_data)*100, 4), epoch)
        writer.add_scalar("train/loss_adv", round(train_loss_adv/len(train_data)*100, 4), epoch)
        
        if epoch%5==0:
            model.eval()
            valid_data.dataset.transform = transform_val
            db_val = torch.utils.data.DataLoader(valid_data, batch_size=args.task_num, shuffle=True, num_workers=0)
            val_correct_count = 0
            val_adv_correct_count = 0
            val_loss = 0
            val_loss_adv = 0
            for _, (x, y) in enumerate(tqdm(db_val, desc="val")):
                x = x.to(device)
                y = y.to(device)
                correct_count, loss = utils.predict(x, y, model)

                val_correct_count += correct_count
                val_loss += loss.item()*x.shape[0]
                
                adv_correct_count, loss_adv = utils.adv_predict(args, args.test_attack, args.test_eps, x, y, model, mode='val')
                val_adv_correct_count += adv_correct_count
                val_loss_adv += loss_adv.item()
                
            writer.add_scalar("val/acc", round(val_correct_count/len(valid_data)*100, 4), epoch)
            writer.add_scalar("val/loss", round(val_loss/len(valid_data)*100, 4), epoch)
            writer.add_scalar("val/acc_adv", round(val_adv_correct_count/len(valid_data)*100, 4), epoch)
            writer.add_scalar("val/loss_adv", round(val_loss_adv/len(valid_data)*100, 4), epoch)

            last_val = round(val_correct_count/len(valid_data)*100, 4)
            last_val_adv = round(val_adv_correct_count/len(valid_data)*100, 4)

        writer.add_scalar('lr', optim.param_groups[0]['lr'], epoch)
        scheduler.step()
        adv_scheduler.step()

            
            # if last_val > best_val:
            #     best_val = last_val
            # if last_val_adv > best_val_adv:
            #     best_val_adv = last_val_adv

    # test
    model.eval()
    db_test = torch.utils.data.DataLoader(test_data, batch_size=args.task_num, shuffle=True, num_workers=0)
    test_correct_count = 0
    test_adv_correct_count = 0
    for _, (x, y) in enumerate(tqdm(db_test, desc="test")):
        x = x.to(device)
        y = y.to(device)
        correct_count, loss = utils.predict(x, y, model)

        test_correct_count += correct_count
        
        adv_correct_count, loss_adv = utils.adv_predict(args, args.test_attack, args.test_eps, x, y, model, mode='val')
        test_adv_correct_count += adv_correct_count

    test_acc =  round(test_correct_count/len(test_data)*100, 4)
    test_adv_acc = round(test_adv_correct_count/len(test_data)*100, 4)

    result = [log_name, test_acc, test_adv_acc, last_val, last_val_adv, round(train_time,4), round(attack_time,4)]
    with open('result.csv', 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(result)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=32)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    
    # Training options
    argparser.add_argument('--model', type=str, help='type of model to use', default="resnet18")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=200)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=128)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    argparser.add_argument('--lr_ratio', type=float, help='adv lr ratio', default=0.5)
    argparser.add_argument('--train_attack', type=str, help='attack for adversarial training', default="")
    argparser.add_argument('--train_eps', type=float, help='training attack bound', default=6.0)
    argparser.add_argument('--bound', action='store_true', default=False)
    argparser.add_argument('--step', type=int, default=6)
    argparser.add_argument('--lipschitz', type=float, default=0.01)

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--test_eps', type=float, help='attack-eps', default=8.0)
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=6)
    argparser.add_argument('--iter', type=int, default=10)

    argparser.add_argument('--sche', type=str, default="multistep")

    args = argparser.parse_args()

    main(args)