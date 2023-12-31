import torch
import numpy as np
from    torch.nn import functional as F
import time

class aRUB:
    def __init__(self, net, rho, q, n_way, imgc, imgsz):
        self.rho = rho
        self.q = q
        self.n_way = n_way
        self.imgc = imgc
        self.imgsz = imgsz
        self.net = net

    def norm_func(self, x):
            return torch.norm(x, p=self.q)
        
    def perturb(self, data, label):
        '''
        Computes approximation of logits
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''
        data.requires_grad = True
        logits = self.net(data)
        logits_sum = torch.sum(logits, dim=0)        
        flag = True
        for logit in logits_sum:
            logit.backward(retain_graph=True)
            data_grad = torch.unsqueeze(data.grad, 1) #[75,3,28,28]->[75,1,3,28,28]
            if(flag):
                jacobian = data_grad.clone().detach()
                flag = False
            else:
                jacobian = torch.cat([jacobian, data_grad], dim = 1)
                
        for i in range(self.n_way-1):
            jacobian[:,self.n_way-1-i] = jacobian[:,self.n_way-1-i]-jacobian[:,self.n_way-2-i]
        # data.requires_grad = True
        # logits = self.net(data)
        # logits_sum = torch.sum(logits, dim=0)
        
        # jacobian = torch.zeros(data.size(0), data.size(1), data.size(2), data.size(3), self.n_way, dtype=data.dtype)
        
        # for i in range(self.n_way):
        #     logits_sum[i].backward(retain_graph=True)
        #     data_grad = torch.unsqueeze(data.grad, 1)  # [75, 3, 28, 28] -> [75, 1, 3, 28, 28]
        #     jacobian[:, :, :, :, i] = data_grad
            
        label_onehot = F.one_hot(label, num_classes=self.n_way)
        logits_label = torch.sum(torch.mul(logits, label_onehot), dim=1)
        
        label_onehot = torch.unsqueeze(label_onehot, 2).expand(label.shape[0],self.n_way,self.imgc*self.imgsz*self.imgsz).view(label.shape[0],self.n_way,self.imgc, self.imgsz,self.imgsz)
        jac_label = torch.sum(torch.mul(jacobian, label_onehot), dim=1)

        logits = logits - torch.unsqueeze(logits_label,1)
        jacobian = jacobian - torch.unsqueeze(jac_label, 1)
        jacobian = jacobian.view(-1, self.imgc, self.imgsz, self.imgsz)

        jac_norm_bat = torch.vmap(self.norm_func)(jacobian)

        logits_adv = logits + self.rho * (jac_norm_bat.view(-1, self.n_way))
        

        return logits_adv
    
    def jacobian(self, data):
        data.requires_grad = True
        logits = self.net(data)
        logits_sum = torch.sum(logits, dim=0)
        
        jacobian = torch.zeros(data.size(0), data.size(1), data.size(2), data.size(3), self.n_way, dtype=data.dtype)
        
        for i in range(self.n_way):
            logits_sum[i].backward(retain_graph=True)
            data_grad = torch.unsqueeze(data.grad, 1)  # [75, 3, 28, 28] -> [75, 1, 3, 28, 28]
            jacobian[:, :, :, :, i] = data_grad
            
        return jacobian

    