import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
from losses import WGAN_loss
from p2s.simulator import Simulator
import random

class Generator(nn.Module):
    def __init__(self, turb_params, batch_size, restorer, device):
        super(Generator,self).__init__()
        self.batch_size = batch_size
        self.simulator = Simulator(turb_params, data_path='./p2s/utils').to(device, dtype=torch.float32)
        for param in self.simulator.parameters():
            param.requires_grad = False
        self.restorer = restorer

    def forward(self, x):
        y = self.restorer(x.permute(1,0,2,3).unsqueeze(0))
        output = self.simulator(torch.cat([y]*self.batch_size, dim=0))
        return output, y


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, norm=nn.InstanceNorm2d):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, True)]
        mult = 1
        for idx in range(4):
            mult_prev = mult
            mult = min(2 ** idx, 8)
            model += [nn.Conv2d(ndf * mult_prev, ndf * mult, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),
                        norm(ndf * mult), nn.LeakyReLU(0.2, True)]
        model += [nn.Conv2d(ndf * mult, 8, kernel_size=3, padding=1, bias=True)]
        self.global_model = nn.Sequential(*model)

        local_model = [nn.Conv2d(input_nc, ndf, kernel_size=5, stride=3, padding=1, bias=True), nn.InstanceNorm2d(ndf),
                        nn.LeakyReLU(0.2, True)]
        local_model += [nn.Conv2d(ndf, 2 * ndf, kernel_size=3, stride=2, dilation=1, padding=1, bias=True),
                        nn.InstanceNorm2d(2 * ndf), nn.LeakyReLU(0.2, True)]
        local_model += [nn.Conv2d(2 * ndf, 1, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
                        nn.InstanceNorm2d(4 * ndf), nn.LeakyReLU(0.2, True)]
        self.local_model = nn.Sequential(*local_model)

    def forward(self, input):
        global_input = input
        loc_h = np.random.randint(0, high=int(input.size(2)) - 21)
        loc_w = np.random.randint(0, high=int(input.size(3)) - 21)
        local_input = input[:, :, loc_h:loc_h + 21, loc_w:loc_w + 21]
        global_output = self.global_model(global_input)
        local_output = self.local_model(local_input)
        out = torch.cat((global_output, local_output), 1)
        return out


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class TurbGAN(nn.Module):
    def __init__(self, save_path, model_g, model_d, lr_g=2e-6, lr_d=1e-5, continue_train=False):
        super(TurbGAN, self).__init__()
        self.model_G = model_g
        self.model_D = model_d
        self.sigma_noise = 1e-4
        self.optimizer_G = optim.Adam(self.model_G.parameters(), lr=lr_g, betas=[0.9, 0.999])
        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=lr_d, betas=[0.9, 0.999])
        self.crit_WGAN = WGAN_loss()
        self.loss_G = torch.tensor([0])
        self.loss_D = torch.tensor([0])
        self.model_names = ['G', 'D']
        self.loss_names = ['loss_G', 'percep_loss', 'pixel_loss', 'wgan_loss_G', 'loss_D']
        self.save_path = save_path
        self.save_img_path = os.path.join(save_path, 'results')
        self.ckpt_path = os.path.join(save_path, 'checkpoints')
        os.makedirs(self.save_path) if not os.path.exists(self.save_path) else exec('pass')

        self.scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, mode='min',factor=0.5, patience=20)
        self.scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_D, mode='min',factor=0.5, patience=20)
        if not continue_train:
            self.model_D.apply(weight_init)

    def _interp_noise(self, img, factor):
        b,c,h,w = img.shape
        noise = torch.randn((b, c, h//factor, w//factor), device=img.device)
        noise = F.interpolate(noise, scale_factor=factor, mode='bilinear', align_corners=True)
        return noise

    def add_noise(self, img):
        r = random.random()
        # factor = random.choice([1,2,4,8,16,32,64])
        factor = 1
        if factor == 1:
            noise = ((r*self.sigma_noise)**0.5)*torch.randn(img.shape, device=img.device)
        else:
            noise = ((r*self.sigma_noise)**0.5)*self._interp_noise(img, factor)
        out = img + noise
        return out.clamp(0,1)

    def set_input(self, input, real):
        self.input_blk = input.cuda()
        self.real = real.cuda()

    def forward(self):
        self.fake, self.recon = self.model_G(self.input_blk)

    def backward_G(self):
        self.loss_G = self.crit_WGAN.compute_G_loss(self.model_D, self.fake)
        self.loss_G.backward(retain_graph=True)
        
    def backward_D(self):
        self.loss_D = self.crit_WGAN.compute_D_loss(self.model_D, self.add_noise(self.fake), self.add_noise(self.real))
        self.loss_D.backward(retain_graph=True)
        
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize(self, step):
        # print(step, '1')
        # GPUtil.showUtilization()
        self.forward()
        # print(step, '2')
        # GPUtil.showUtilization()
        # update D
        if not (step > 280000 and step % 6 ==0):
            self.set_requires_grad(self.model_D, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        # update G
        if step > 280000 and step % 6 == 0:
            self.set_requires_grad(self.model_D, False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()    
        # print(step, '3')
        # GPUtil.showUtilization()

    def update_learning_rate(self, metric):
        self.scheduler_G.step(metric)
        self.scheduler_D.step(metric)

    def save_networks(self, step):
        save_name_G = 'model_G_{}.pth'.format(step)
        torch.save({'step': step, 
                    'state_dict': self.model_G.state_dict(),
                    'optimizer' : self.optimizer_G.state_dict()}, os.path.join(self.ckpt_path, save_name_G)) 
        save_name_D = 'model_D_{}.pth'.format(step)
        torch.save({'step': step, 
                    'state_dict': self.model_D.state_dict(),
                    'optimizer' : self.optimizer_D.state_dict()}, os.path.join(self.ckpt_path, save_name_D)) 

    def get_current_info(self, step):
        message = 'step: {} '.format(step)
        message += 'loss_g: {} loss_d: {} '.format(self.loss_G.item(), self.loss_D.item())
        message += 'PSNR: {} '.format(self.loss_G.item())
        message += 'lr_g: {:7f}; lr_d: {:7f}\n'.format(self.optimizer_G.param_groups[0]['lr'], self.optimizer_D.param_groups[0]['lr'])
        print(message)

    def save_results(self, step):
        out = self.recon.data.squeeze().float().cpu().clamp_(0, 1).numpy() * 255
        out = np.transpose(out, (1, 2, 0)).round().astype(np.uint8)
        out_save = Image.fromarray(out)
        out_save.save(os.path.join(self.save_img_path, f'img_{step}.jpg'), "JPEG")

