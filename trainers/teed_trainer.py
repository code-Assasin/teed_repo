"""
Code for Training TEED model
"""
from __future__ import print_function

import os
import time
import kornia as ko
import numpy as np
from tqdm import tqdm

import sys
import wandb
from PIL import Image

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append("../")
from utils.loss_util import *
from models.teed_model import TEED

class TEEDTrainer:
    def __init__(self, params, device="cuda"):
        super(TEEDTrainer, self).__init__()
        self.params = params
        self.device = device
        self.logger = params["logger"]
        self.noise_std = params["noise_std"]
        self.total_epochs = params["epochs"]
        self.trainloader = params["trainloader"]
        self.testloader = params["testloader"]
        self.net = TEED().to(device)
        self.edge_operator = ko.filters.Sobel(normalized=True)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )

        schedule_epochs = self.total_epochs // 10

        if params["scheduler"] == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.total_epochs
            )
        elif params["scheduler"] == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[2.5 * schedule_epochs, 5 * schedule_epochs],
                gamma=0.1,
            )
        elif params["scheduler"] == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=params["lr"],
                steps_per_epoch=len(self.trainloader),
                epochs=self.total_epochs,
            )
        elif params["scheduler"] == "constant":
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, lr=params["lr"], factor=1
            ) # dummy scheduler .. no change in lr
        else:
            raise NotImplementedError(
                f"Scheduler {params['scheduler']} not implemented"
            )

        self.bdcn_reg = [1.0, 1.0, 1.0, 1.0]  # for bdcn loss2-B4
        self.cats_reg = [1.0, 1.0]  # for cats loss [0.01, 4.0]

        self.cats_loss = TEEDLoss(loss_type="cats", device=device)
        self.bdcn_loss = TEEDLoss(loss_type="cosine", device=device)

        self.checkpoint_path = params["checkpoint_path"]

        # if self.load_checkpoint:
        #     print(f"Restoring weights from: {self.checkpoint_path}")
        #     self.net.load_state_dict(
        #         torch.load(self.checkpoint_path, map_location=self.device)
        #     )

    def visualizer(self, epoch, images, orig_edges, noisy_edges, preds_list):
        # sample just first image from batch
        images = images[0].cpu().detach().numpy().transpose(1, 2, 0)
        orig_edges = orig_edges[0].cpu().detach().numpy().transpose(1, 2, 0)
        noisy_edges = noisy_edges[0].cpu().detach().numpy().transpose(1, 2, 0)
        preds_list = [
            torch.sigmoid(preds[0]).cpu().detach().numpy().transpose(1, 2, 0)
            for preds in preds_list
        ]

        # convert numpy arrays to PIL Images
        images_pil = Image.fromarray((images * 255).astype(np.uint8))
        orig_edges_pil = Image.fromarray((orig_edges * 255).astype(np.uint8))
        noisy_edges_pil = Image.fromarray((noisy_edges * 255).astype(np.uint8))
        preds_list_pil = [
            Image.fromarray((pred * 255).astype(np.uint8)) for pred in preds_list
        ]

        # create wandb Images
        images_wb = wandb.Image(images_pil)
        orig_edges_wb = wandb.Image(orig_edges_pil)
        noisy_edges_wb = wandb.Image(noisy_edges_pil)
        preds_list_wb = [wandb.Image(pred_pil) for pred_pil in preds_list_pil]

        # log images
        wandb.log(
            {
                "images": images_wb,
                "orig_edges": orig_edges_wb,
                "noisy_edges": noisy_edges_wb,
                "preds": preds_list_wb,
                "Epoch": epoch,
            }
        )
        return

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
        self.net.train()
        loss_avg = 0
        for batch_id, (img, _) in enumerate(tqdm(self.trainloader)):
            orig_images = img.to(self.device)
            noisy_images = self.gaussian_noise(orig_images, self.noise_std, epoch).to(
                self.device
            )
            # noisy_images = orig_images

            #  apply sobel
            noisy_edges = self.sobel_edge(noisy_images)  # Input
            orig_edges = self.sobel_edge(orig_images)  # Output

            preds_list = self.net(noisy_edges)

            bdcn_curr = 0
            cats_curr = 0
            for count in range(len(preds_list)):
                bdcn_curr += self.bdcn_loss(
                    preds_list[count], orig_edges, self.bdcn_reg[count]
                )
                cats_curr += self.cats_loss(
                    preds_list[count], orig_edges, self.cats_reg
                )

            tLoss = bdcn_curr + 1 * cats_curr

            self.optimizer.zero_grad()
            tLoss.backward()
            self.optimizer.step()
            if self.params["scheduler"] != "multistep":
                self.scheduler.step()
            loss_avg += tLoss.item()

            self.logger.log(
                {
                    "Total train loss": tLoss.item(),
                    "bdcn loss": bdcn_curr.item(),
                    "cats loss": cats_curr.item(),
                },
            )

        loss_avg = loss_avg / len(self.trainloader)
        self.logger.log({"Epoch": epoch, "Train Average Loss": loss_avg})
        return loss_avg

    def test(self, epoch, log_interval_vis=5):
        # Put model in evaluation mode
        self.net.eval()
        loss_avg = 0
        with torch.no_grad():
            for batch_id, (img, _) in enumerate(tqdm(self.testloader)):
                orig_images = img.to(self.device)
                noisy_images = self.gaussian_noise(orig_images, self.noise_std, epoch).to(
                    self.device
                )
                # noisy_images = orig_images

                #  apply sobel
                noisy_edges = self.sobel_edge(noisy_images)
                orig_edges = self.sobel_edge(orig_images)

                preds_list = self.net(noisy_edges)

                bdcn_curr = 0
                cats_curr = 0
                for count in range(len(preds_list)):
                    bdcn_curr += self.bdcn_loss(
                        preds_list[count], orig_edges, self.bdcn_reg[count]
                    )
                    cats_curr += self.cats_loss(
                        preds_list[count], orig_edges, self.cats_reg
                    )

                tLoss = bdcn_curr + 1 * cats_curr
                loss_avg += tLoss.item()

            # log images from the last batch
            self.visualizer(epoch, orig_images, orig_edges, noisy_edges, preds_list)

            loss_avg = loss_avg / len(self.testloader)
            self.logger.log({"Epoch": epoch, "Test Average Loss": loss_avg})

        return loss_avg

    def solve(self):
        # add progress bar
        for epoch in tqdm(range(0, self.total_epochs)):
            
            self.logger.log({"Epoch": epoch, "Learning Rate": self.optimizer.param_groups[0]["lr"]})
            train_avg_loss = self.train_one_epoch(epoch)

            if self.params["scheduler"] == "multistep":
                self.scheduler.step()

            # Save model after end of every epoch
            if epoch % 10 == 0 or epoch == self.total_epochs - 1:

                test_avg_loss = self.test(epoch)
                torch.save(
                    self.net.state_dict(),
                    os.path.join(
                        self.params["checkpoint_path"], f"model_epoch_{epoch}.pth"
                    ),
                )

        return train_avg_loss, test_avg_loss
