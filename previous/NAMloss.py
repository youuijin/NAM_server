import torch
import torch.nn.functional as F
import utils

class NAMLoss:
    def __init__(self, net, k):
        self.net = net
        self.k = k

    def get_loss(self, x, y, sum=False, reduction=True):
        logit = self.net(x.unsqueeze(dim=1))
        loss = F.cross_entropy(logit, y, reduction='none')
        softmax = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
        softmax = softmax - y_onehot
        norm = torch.sum(softmax, softmax, dim=1)

        approx_loss = self.k * norm
        if sum:
            approx_loss += loss
        if reduction:
            approx_loss = torch.mean(approx_loss)

        return approx_loss
        
    # def get_loss(self, x, y, reduction=True):
    #     logit = self.net(x)
    #     softmax = F.softmax(logit, dim=1)
    #     y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
    #     softmax = softmax - y_onehot
    #     norm = torch.diagonal(torch.matmul(softmax, softmax.T))

    #     if reduction:
    #         return torch.mean(self.k * norm)
    #     else:
    #         return self.k * norm


if __name__ == '__main__':
    a = torch.randn(256,10)
    import time
    start = time.time()
    for i in range(100000):
        b = torch.diagonal(a@a.T)
    end = time.time()
    print(end-start)

    start = time.time()
    for i in range(100000):
        b = torch.sum(a*a, dim=1)
    end = time.time()
    print(end-start)