import os, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm
import kornia as ko
import sys
sys.path.append('../')

import models.pix2pix_model as network


class Pix2PixTrainer():
    def __init__(self, params, device):
        super(Pix2PixTrainer, self).__init__()
        self.params = params
        self.device = device
        self.batch_size = params["batch_size"]
        self.ngf = 64
        self.ndf = 64
        self.input_size = 256
        self.crop_size = 256
        self.resize_scale = 286
        self.fliplr = True
        self.train_epoch = 200
        self.lrD = 0.0002
        self.lrG = 0.0002
        self.L1_lambda = 100
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.inverse_order = True
        self.trainloader = params["trainloader"]
        self.testloader = params["testloader"]
        self.logger = params["logger"]
        self.noise_std = params["noise_std"]
        self.total_epochs = params["epochs"]


        self.edge_operator = ko.filters.Sobel(normalized=True)

        # network
        self.G = network.generator(self.ngf)
        self.D = network.discriminator(self.ndf)
        self.G.weight_init(mean=0.0, std=0.02)
        self.D.weight_init(mean=0.0, std=0.02)
        self.G, self.D = self.G.to(device), self.D.to(device)

        # loss
        self.BCE_loss = nn.BCELoss().to(device)
        self.L1_loss = nn.L1Loss().to(device)

        # Adam optimizer
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta1))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta1))

        self.train_hist = {}
        self.train_hist['D_losses'] = []
        self.train_hist['G_losses'] = []

    def sobel_edge(self, inp):
        # use kornia to apply sobel filter to all channels
        edge_maps = torch.zeros_like(inp)
        for ch in range(inp.shape[1]):
            edge_maps[:, ch : ch + 1, :, :] = self.edge_operator(
                inp[:, ch : ch + 1, :, :]
            )
        return edge_maps

    def gaussian_noise(self, inp, std, epoch):
        # for cifar std :'gaussian_noise': ((0.04,), (0.06,), (.08,), (.09,), (.10,)),
        # for imagenet std : 'gaussian_noise': ((.08,), (.12,), (0.18,), (0.26,), (0.38,)),
        torch.manual_seed(epoch * torch.randint(1000, (1,)))
        return torch.clamp(inp + torch.randn_like(inp) * std, 0, 1)

    def train_one_epoch(self, epoch):
        self.G.train()
        self.D.train()
        D_losses = []
        G_losses = []
        num_iter = 0
        for batch_id, (x, _) in enumerate(tqdm(self.trainloader)):
            # train discriminator D
            print("batch_id: ", batch_id, "x: ", x.shape)
            self.D.zero_grad()

            y = self.sobel_edge(x) # target edge map = original image edges 
            x = self.sobel_edge(self.gaussian_noise(x, self.noise_std, epoch)) # input edge map = noisy image edges                

            x, y = Variable(x.to(self.device)), Variable(y.to(self.device))

            D_result = self.D(x, y).squeeze()
            D_real_loss = self.BCE_loss(D_result, Variable(torch.ones(D_result.size()).to(self.device)))

            G_result = self.G(x)
            D_result = self.D(x, G_result).squeeze()
            D_fake_loss = self.BCE_loss(D_result, Variable(torch.zeros(D_result.size()).to(self.device)))

            D_train_loss = (D_real_loss + D_fake_loss) * 0.5
            D_train_loss.backward()
            self.D_self.paramsimizer.step()

            self.train_hist['D_losses'].append(D_train_loss.data[0])

            D_losses.append(D_train_loss.data[0])

            # train generator G
            self.G.zero_grad()

            G_result = self.G(x)
            D_result = self.D(x, G_result).squeeze()

            G_train_loss = self.BCE_loss(D_result, Variable(torch.ones(D_result.size()).to(self.device))) + self.params.L1_lambda * self.L1_loss(G_result, y)
            G_train_loss.backward()
            self.G_self.paramsimizer.step()

            self.train_hist['G_losses'].append(G_train_loss.data[0])

            G_losses.append(G_train_loss.data[0])

            num_iter += 1

            print('Epoch [%d/%d], Step[%d/%d], D_loss: %.4f, G_loss: %.4f'
                    % (epoch, self.train_epoch, batch_id, len(self.trainloader), D_train_loss.data[0], G_train_loss.data[0]))
        
        return D_losses, G_losses
        
if __name__ == "__main__":
    # try loading the model and training one epoch
    
    t = Pix2PixTrainer(params, device)

