import  torch, argparse, csv, os
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import utils

def main(args):
    utils.set_seed()

    device = torch.device('cuda:'+str(args.device_num))

    model = utils.setModel(args.model, args.n_way, args.imgsz, args.imgc, pretrained=None)
    model = model.to(device)

    norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    transform = transforms.Compose([transforms.RandomCrop(args.imgsz, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])
    transform_val = transforms.Compose([transforms.Resize((args.imgsz, args.imgsz)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    train_data = CIFAR10(root='../data', train=True, download=False, transform=transform)
    train_size = int(len(train_data) * 0.8) # 80% training data
    valid_size = len(train_data) - train_size # 20% validation data
    train_data, valid_data = random_split(train_data, [train_size, valid_size])
    test_data = CIFAR10(root='../data', train=False, download=False, transform=transform_val)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[int(args.epoch*0.5), int(args.epoch*0.8)], gamma=0.1) 

    best_acc = 0
    best_adv_acc = 0
    
    for epoch in range(args.epoch):
        train_data.dataset.transform = transform
        train_db = DataLoader(train_data, batch_size=args.task_num, shuffle=True, num_workers=2)
        
        train_correct_count = 0
        train_loss = 0

        model.train()

        for _, (x, y) in enumerate(tqdm(train_db, desc="train")):
            x = x.to(device)
            y = y.to(device)
            correct_count, loss = utils.predict(x, y, model)
            train_correct_count += correct_count
            train_loss += loss.item()*x.shape[0]
            
            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"epoch: {epoch}\tacc: {round(100.0*train_correct_count/len(train_data), 4)}\tloss: {round(train_loss/len(train_data), 4)}")
        
        if epoch%5==0:
            model.eval()
            valid_data.dataset.transform = transform_val
            db_val = DataLoader(valid_data, batch_size=args.task_num, shuffle=True, num_workers=0)
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

            val_acc = round(100.0*val_correct_count/len(valid_data), 4)
            val_adv_acc = round(100.0*val_adv_correct_count/len(valid_data), 4)

            print(f"Val epoch: {epoch}\nacc: {round(100.0*val_correct_count/len(valid_data), 4)}\tloss: {round(val_loss/len(valid_data), 4)}")
            print(f"adv_acc: {round(100.0*val_adv_correct_count/len(valid_data), 4)}\tadv_loss: {round(val_loss_adv/len(valid_data), 4)}")

            if best_acc < val_acc:
                best_acc = val_acc
                best_adv_acc = val_adv_acc
                torch.save(model.state_dict(), f'./models/{args.model}.pt')

        scheduler.step()

    # test
    model.eval()
    db_test = DataLoader(test_data, batch_size=args.task_num, shuffle=True, num_workers=0)
    test_correct_count = 0
    test_adv_correct_count = 0
    for _, (x, y) in enumerate(tqdm(db_test, desc="test")):
        x = x.to(device)
        y = y.to(device)
        correct_count, loss = utils.predict(x, y, model)

        test_correct_count += correct_count
        
        adv_correct_count, loss_adv = utils.adv_predict(args, args.test_attack, args.test_eps, x, y, model, mode='val')
        test_adv_correct_count += adv_correct_count

    test_acc = round(100.0*test_correct_count/len(test_data), 4)
    test_adv_acc = round(100.0*test_adv_correct_count/len(test_data), 4)
   
    result = [args.model, test_acc, test_adv_acc, best_acc, best_adv_acc]
    with open('model_performance.csv', 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(result)

    torch.save(model.state_dict(), f'./models/{args.model}_test.pt')
    print(result)


def clip(args):
    utils.set_seed()

    # set gpu
    device = torch.device('cuda:'+str(args.device_num))

    # load model
    model = utils.setModel(args.model, args.n_way, args.imgsz, args.imgc, pretrained=None)
    model.load_state_dict(torch.load(f'./models/{args.model}_test.pt'))
    model = model.to(device)

    # load dataset (use test set)
    norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform = transforms.Compose([transforms.Resize((args.imgsz, args.imgsz)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    data = CIFAR10(root='../data', train=False, download=False, transform=transform)
    db = DataLoader(data, batch_size=args.task_num, shuffle=True, num_workers=0)
    
    at = utils.setAttack(args.test_attack, model, args.test_eps, args)

    model.eval()
    clean, adver, clip = 0, 0, 0
    for _, (x, y) in enumerate(tqdm(db)):
        x = x.to(device)
        y = y.to(device)

        # calculate performance of clean images
        correct_count, _ = utils.predict(x, y, model)
        clean += correct_count
        
        # adversarial attack
        advx = at.perturb(x, y)
        correct_count, _ = utils.predict(advx, y, model)
        adver += correct_count

        # clip adversarial images
        clip_advx = utils.clamp(advx, min=args.clip_min, max=args.clip_max)
        correct_count, _ = utils.predict(clip_advx, y, model)
        clip += correct_count

    clean_acc = round(100.0*clean/len(data), 4)
    adv_acc = round(100.0*adver/len(data), 4)
    clip_acc = round(100.0*clip/len(data), 4)
    print(clean_acc, adv_acc, clip_acc)

    result = [args.model, args.test_attack, args.test_eps, args.clip_min, args.clip_max, clean_acc, adv_acc, clip_acc]
    with open('clip_performance.csv', 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(result)


def make_image(args):
    utils.set_seed()
    device = torch.device('cuda:'+str(args.device_num))

    save_path = f"./images/{args.model}/{args.test_attack}_{args.test_eps}/{args.clip_min}_{args.clip_max}"
    os.makedirs(save_path, exist_ok=True)

    # load model
    model = utils.setModel(args.model, args.n_way, args.imgsz, args.imgc, pretrained=None)
    model.load_state_dict(torch.load(f'./models/{args.model}_test.pt'))
    model = model.to(device)

    # load dataset (use test set)
    norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    
    img_transform = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2.0, 2.0, 2.0)),
        transforms.ToPILImage()
    ])

    data = CIFAR10(root='../data', train=False, download=False, transform=transform)
    db = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)
    
    at = utils.setAttack(args.test_attack, model, args.test_eps, args)

    model.eval()
    save_idx = 0
    for _, (x, y) in enumerate(tqdm(db)):
        x = x.to(device)
        y = y.to(device)

        # calculate performance of clean images
        clean, _ = utils.predict(x, y, model)
        
        # adversarial attack
        advx = at.perturb(x, y)
        adv, _ = utils.predict(advx, y, model)

        # clip adversarial images
        clip_advx = utils.clamp(advx, min=args.clip_min, max=args.clip_max)
        clip, _ = utils.predict(clip_advx, y, model)

        if clean+adv+clip==3:
            # save clean images
            print(img_transform(x.cpu().squeeze()))
            exit()
            img_transform(x.cpu().squeeze()).save(f"{save_path}/{save_idx}_clean.png")

            
            # save adversarial attacked image
            img = img_transform(advx.cpu().squeeze())
            img.save(f"{save_path}/{save_idx}_adv.png")

            # save clip adversarial images
            img = img_transform(clip_advx.cpu().squeeze())
            img.save(f"{save_path}/{save_idx}_clip.png")

            save_idx += 1

        if save_idx>9:
            exit()

def make_histogram(args):
    pass


       
        
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
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.01)

    # adversarial attack options
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--test_eps', type=float, help='attack-eps', default=8.0)
    argparser.add_argument('--iter', type=int, default=10)

    argparser.add_argument('--mode', type=str, default='clip')
    argparser.add_argument('--clip_min', type=float, default=0.0)
    argparser.add_argument('--clip_max', type=float, default=1.0)

    args = argparser.parse_args()

    if args.mode == 'pretrain':
        print('Pretrain Mode')
        main(args)
    elif args.mode == 'save':
        print('Image Save Mode')
        make_image(args)
    elif args.mode == 'histogram':
        print("Histogram Save Mode")
        make_histogram(args)
    else:
        print('Clip Performance Mode')
        clip(args)