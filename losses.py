import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class WGAN_loss():
    def compute_G_loss(self, net, fake):
        self.fake_D = net(fake)
        return -self.fake_D.mean()

    def compute_gradient_penalty(self, real, fake, net_D):
        self.Lambda = 10
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real.size()).cuda()
        compose = alpha * real + (1 - alpha) * fake
        compose = Variable(compose.cuda(), requires_grad=True)
        D_compose = net_D.forward(compose)
        gradients = autograd.grad(outputs=D_compose, inputs=compose, grad_outputs=torch.ones(D_compose.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)
        gradient_penalty = ((gradients[0].norm(2, dim=1) - 1) ** 2).mean() * self.Lambda
        return gradient_penalty

    def compute_D_loss(self, net_D, fake, real):
        self.gradient_penalty = self.compute_gradient_penalty(real.data, fake.data, net_D)
        self.D_fake = net_D.forward(fake.detach())
        self.D_real = net_D.forward(real)

        self.loss_D = self.D_fake.mean() - self.D_real.mean()
        self.loss_GP = self.loss_D + self.gradient_penalty
        return self.loss_GP


