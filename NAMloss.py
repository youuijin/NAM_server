import torch
import torch.nn.functional as F
import utils

class NAMLoss:
    def __init__(self, net, k):
        self.net = net
        self.k = k

    def get_loss(self, x, y, sum=False, reduction=True):
        logit = self.net(x)
        loss = F.cross_entropy(logit, y, reduction='none')
        softmax = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
        softmax = softmax - y_onehot
        norm = torch.diagonal(torch.matmul(softmax, softmax.T))

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
    model = utils.setModel("resnet18", 5, 112, 3)
    loss = NAMLoss(model, 1)
    x = torch.randn(2, 3, 112, 112, requires_grad=True)
    y = torch.zeros(2, 5)
    y[0][1] = 1
    y[1][2] = 1
    loss.get_loss(x, y)