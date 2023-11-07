import  torch
import  numpy as np
import  random
import  argparse

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from tqdm import tqdm
import time
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

import utils

def main(args):
    seed = 706
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:'+str(args.device_num))

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

    model_name = f"step{args.step}_{args.train_eps}_{args.lr}_{args.lr_ratio}_{round(args.lipschitz,4)}"

    writer = SummaryWriter(f"./step_log/{model_name}")
    approx_writer = SummaryWriter("./step_log/approx_loss")
    adv_writer = SummaryWriter("./step_log/adv_loss")

    all_time = time.time()
    train_time = 0
    # loss_time = 0

    for epoch in range(args.epoch):
        train_st = time.time()
        db = torch.utils.data.DataLoader(train_data, batch_size=args.task_num, shuffle=True, num_workers=2)
        train_correct_count = 0
        train_loss = 0
        train_loss_adv = 0

        approx_loss_sum = 0
        loss_adv_sum = 0

        bound_rate = 0
        bound_rate_elements = 0
        
        model.train()

        for _, (x, y) in enumerate(tqdm(db, desc="train")):
            x = x.to(device)
            y = y.to(device)
            logit = model(x)
            pred = torch.nn.functional.softmax(logit, dim=1)
            outputs = torch.argmax(pred, dim=1)
            train_correct_count += (outputs == y).sum().item()
            loss = torch.nn.functional.cross_entropy(logit, y)
            train_loss += loss.item()*x.shape[0]
            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()

            y_onehot = F.one_hot(y, num_classes = pred.shape[1])

            # adversarial training
            if args.step==1:
                # Quadratic upper bound
                # L(h(x')) <= L(h(x))+(h(x')-h(x))L'(h(x)) + K/2*||h(x')-h(x)||^2
                at = utils.setAttack("PGD_Linf", model, args.train_eps, args)
                x_adv = at.perturb(x, y)
                logit = model(x)
                logit_adv = model(x_adv)
                softmax = F.softmax(logit, dim=1)
                norm = torch.norm(logit_adv-logit)
                loss = torch.nn.functional.cross_entropy(logit, y, reduction='none')
                loss_adv = torch.nn.functional.cross_entropy(logit_adv, y, reduction='none')

                # cross entropy 미분 -> y'-y
                approx_loss = loss + torch.sum((logit_adv-logit)*(softmax-y_onehot), dim=1) + args.lipschitz/2.0*torch.pow(norm, 2)
                approx_loss_sum += approx_loss.sum().item()
                loss_adv_sum += loss_adv.sum().item()

                bound_rate_elements += x.shape[0]
                bound_rate += (loss_adv<=approx_loss).sum().item() 

            elif args.step==2:
                # h(x')에 대해 미분한 결과=0이 되는 성질 이용
                # ||h(x')-(h(x)-1/K*L'(h(x)))||^2
                at = utils.setAttack("PGD_Linf", model, args.train_eps, args)
                x_adv = at.perturb(x, y)
                logit = model(x)
                logit_adv = model(x_adv)
                softmax = F.softmax(logit, dim=1)

                # cross entropy 미분 -> y'-y
                approx_loss = torch.pow(torch.norm(logit_adv - (logit - 1.0/args.lipschitz*(softmax-y_onehot))),2)
            
            elif args.step==3: 
                # approximation #1-1
                # linear approximation
                # ||delta*h'(x)+1/K*L'(h(x))||^2
                
                at = utils.setAttack("PGD_Linf", model, args.train_eps, args)
                x_adv = at.perturb(x, y)
                ##################################################
                # delta를 정확히 계산
                delta = torch.reshape((x_adv - x), (x.shape[0],-1)).to(device)
                # delta를 그냥 eps로 계산

                ##################################################
                logit = model(x)
                softmax = F.softmax(logit, dim=1)

                # calculate jacobian
                x_copy = x.detach().clone().requires_grad_(True)

                logits = model(x_copy)
                logits_sum = torch.sum(logits, dim=0)

                jacobian = torch.zeros(x_copy.size(0), args.n_way, x_copy.size(1), x_copy.size(2), x_copy.size(3), dtype=x_copy.dtype)

                for i in range(args.n_way):
                    logits_sum[i].backward(retain_graph=True)
                    x_grad = x_copy.grad.clone().detach()
                    #x_grad = x_grad.unsqueeze(1)  # [batch_size, 1, input_channels, input_height, input_width]
                    jacobian[:, i, :, :, :] = x_grad

                # print(jacobian.shape, x.shape[0], args.n_way)
                jacobian = torch.reshape(jacobian, (x.shape[0], args.n_way, -1)).to(device)

                # cross entropy 미분 -> y'-y
                approx_loss = torch.pow(torch.norm(torch.matmul(delta, jacobian.transpose(1, 2))+1.0/args.lipschitz*(softmax-y_onehot)),2)

            elif args.step==4: 
                # approximation #1-2
                # linear approximation + heuristic
                logit = model(x)
                softmax = F.softmax(logit, dim=1)

                approx_loss = torch.pow((1.0+args.train_eps)/args.lipschitz*torch.norm(softmax-y_onehot), 2)

            elif args.step==5:
                # minus upper bound
                at = utils.setAttack("PGD_Linf", model, args.train_eps, args)
                x_adv = at.perturb(x, y)
                logit_adv = model(x_adv)
                loss_adv = torch.nn.functional.cross_entropy(logit_adv, y, reduction='none')

                logit = model(x)
                softmax = F.softmax(logit, dim=1)

                approx_loss = loss-1.0/(2*args.lipschitz)*torch.pow(torch.norm(softmax-y_onehot),2)
                approx_loss_sum += approx_loss.sum().item()
                loss_adv_sum += loss_adv.sum().item()

                bound_rate_elements += x.shape[0]
                bound_rate += (loss_adv<=approx_loss).sum().item() 

            elif args.step==6:
                # plus upper bound
                at = utils.setAttack("PGD_Linf", model, args.train_eps, args)
                x_adv = at.perturb(x, y)
                logit_adv = model(x_adv)
                loss_adv = torch.nn.functional.cross_entropy(logit_adv, y, reduction='none')

                logit = model(x)
                softmax = F.softmax(logit, dim=1)

                approx_loss = loss+3.0/(2*args.lipschitz)*torch.pow(torch.norm(softmax-y_onehot),2)
                approx_loss_sum += approx_loss.sum().item()
                loss_adv_sum += loss_adv.sum().item()

                bound_rate_elements += x.shape[0]
                bound_rate += (loss_adv<=approx_loss).sum().item()

            elif args.step==7:
                at = utils.setAttack("PGD_Linf", model, args.train_eps, args)
                x_adv = at.perturb(x, y)
                logit = model(x)
                logit_adv = model(x_adv)
                softmax = F.softmax(logit, dim=1)

                approx_loss_2 = torch.pow(torch.norm(logit_adv - (logit - 1.0/args.lipschitz*(softmax-y_onehot))),2)

                approx_loss_6 = loss+3.0/(2*args.lipschitz)*torch.pow(torch.norm(softmax-y_onehot),2)
                approx_loss = approx_loss_2.sum() + approx_loss_6.sum()
                
                approx_loss_sum += approx_loss.sum().item()

            elif args.step==8:
                at = utils.setAttack("PGD_Linf", model, args.train_eps, args)
                x_adv = at.perturb(x, y)

                delta = torch.reshape((x_adv - x), (x.shape[0],-1)).to(device)

                logit = model(x)
                softmax = F.softmax(logit, dim=1)

                x_copy = x.detach().clone().requires_grad_(True)

                logits = model(x_copy)
                logits_sum = torch.sum(logits, dim=0)

                jacobian = torch.zeros(x_copy.size(0), args.n_way, x_copy.size(1), x_copy.size(2), x_copy.size(3), dtype=x_copy.dtype)

                for i in range(args.n_way):
                    logits_sum[i].backward(retain_graph=True)
                    x_grad = x_copy.grad.clone().detach()
                    jacobian[:, i, :, :, :] = x_grad

                jacobian = torch.reshape(jacobian, (x.shape[0], args.n_way, -1)).to(device)

                approx_loss_3 = torch.pow(torch.norm(torch.matmul(delta, jacobian.transpose(1, 2))+1.0/args.lipschitz*(softmax-y_onehot)),2)
                
                approx_loss_6 = loss+3.0/(2*args.lipschitz)*torch.pow(torch.norm(softmax-y_onehot),2)
                approx_loss = approx_loss_3.sum() + approx_loss_6.sum()
                
                approx_loss_sum += approx_loss.sum().item()

            elif args.step==9:
                logit = model(x)
                softmax = F.softmax(logit, dim=1)

                approx_loss_4 = torch.pow((1.0+args.train_eps)/args.lipschitz*torch.norm(softmax-y_onehot), 2)
                approx_loss_6 = loss+3.0/(2*args.lipschitz)*torch.pow(torch.norm(softmax-y_onehot),2)
                approx_loss = approx_loss_4.sum() + approx_loss_6.sum()
                
                approx_loss_sum += approx_loss.sum().item()

            if args.step>0:
                approx_loss = approx_loss.sum()
                train_loss_adv += approx_loss.item() * x.shape[0]
                adv_optim.zero_grad()
                approx_loss.backward()
                adv_optim.step()
            
        train_time += time.time()-train_st

        if args.step in [1, 5, 6]:
            writer.add_scalar("bound_rate", round(bound_rate/bound_rate_elements*100,4), epoch)
            approx_writer.add_scalar(f"{model_name}/", round(approx_loss_sum/len(train_data), 4), epoch)
            adv_writer.add_scalar(f"{model_name}/", round(loss_adv_sum/len(train_data), 4), epoch)
        writer.add_scalar("train/acc", round(train_correct_count/len(train_data)*100, 4), epoch)
        writer.add_scalar("train/loss", round(train_loss/len(train_data)*100, 4), epoch)
        writer.add_scalar("train/approx", round(train_loss_adv/len(train_data)*100, 4), epoch)
    
        model.eval()
        db_val = torch.utils.data.DataLoader(valid_data, batch_size=args.task_num, shuffle=True, num_workers=0)
        val_correct_count=0
        val_adv_correct_count=0
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

    all_time = time.time() - all_time

    writer.add_scalar("time/all", all_time, 0)
    writer.add_scalar("time/train", train_time, 0)
    

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=112)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    
    # Training options
    argparser.add_argument('--model', type=str, help='type of model to use', default="")
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=256)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    argparser.add_argument('--lr_ratio', type=float, help='adv lr ratio', default=0.5)

    # NAM options
    argparser.add_argument('--lipschitz', type=float, help='lipschitz constant', default=4/270) # 계산한 값을 토대로
    argparser.add_argument('--step', type=int, help='step in proof', default=1)
    argparser.add_argument('--twice', action='store_true', help='apply standard loss value twice', default=False)
    argparser.add_argument('--train_eps', type=float, help='to check upper bound', default=2.0)

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--test_eps', type=float, help='attack-eps', default=6)
    argparser.add_argument('--iter', type=int, default=10)

    args = argparser.parse_args()

    main(args)