import torch.nn.functional as F
import torch

class QAUB:
    def __init__(self, args):
        self.train_eps = args.train_eps
        self.step = args.step
        # self.lipschitz = args.lipschitz
        
        if self.step==4:
            self.func = self.step4
        elif self.step==6:
            self.func = self.step6
        elif self.step==9: 
            self.func = self.step9

    def approx_loss(self, model, x, y):
        return self.func(model, x, y) # batched

    def step4(self, model, x, y):
        # linear approximation + heuristic
        logit = model(x)
        pred = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = pred.shape[1])
        # approx_loss = torch.pow((1.0+self.train_eps)/self.lipschitz*torch.norm(pred-y_onehot), 2)
        approx_loss = torch.pow(self.train_eps*torch.norm(pred-y_onehot), 2)
        return approx_loss

    def step6(self, model, x, y):
        # plus upper bound
        logit = model(x)
        pred = F.softmax(logit, dim=1)
        loss = F.cross_entropy(logit, y)
        y_onehot = F.one_hot(y, num_classes = pred.shape[1])
        # approx_loss = loss+3.0/(2*self.lipschitz)*torch.pow(torch.norm(pred-y_onehot),2)
        approx_loss = loss+self.train_eps*torch.pow(torch.norm(pred-y_onehot),2)
        return approx_loss

    def step9(self, model, x, y):
        logit = model(x)
        pred = F.softmax(logit, dim=1)
        loss = F.cross_entropy(logit, y)
        
        y_onehot = F.one_hot(y, num_classes = pred.shape[1])
        approx_loss_4 = torch.pow(self.train_eps*torch.norm(pred-y_onehot), 2)
        approx_loss_6 = loss+self.train_eps*torch.pow(torch.norm(pred-y_onehot),2)
        approx_loss = approx_loss_4 + approx_loss_6
        return approx_loss