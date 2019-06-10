import argparse
import numpy as np
import os
# import tensorflow as tf
import torch
from codebase import utils as ut
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from codebase.models.vae import VAE

def refine(train_loader_set, mean_set, variance_set, z_dim, device, tqdm, writer,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', y_status='none', reinitialize=False):
    # Optimization

    i = 0
    with tqdm(total=iter_max) as pbar:
        while True:
            for index, train_loader in enumerate(train_loader_set):
                print("Iteration:", i)
                print("index: ", index)

                z_prior_m = torch.nn.Parameter(mean_set[index].cpu(), requires_grad=False).to(device)
                z_prior_v = torch.nn.Parameter(variance_set[index].cpu(), requires_grad=False).to(device)
                vae = VAE(z_dim=z_dim, name=model_name,
                          z_prior_m=z_prior_m, z_prior_v=z_prior_v).to(device)
                optimizer = optim.Adam(vae.parameters(), lr=1e-3)
                if i == 0:
                    print("Load model")
                    ut.load_model_by_name(vae, global_step=20000)
                else:
                    print("Load model")
                    ut.load_model_by_name(vae, global_step=iter_save)
                for batch_idx, (xu, yu) in enumerate(train_loader):
                     # i is num of gradient steps taken by end of loop iteration
                    optimizer.zero_grad()

                    xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
                    yu = yu.new(np.eye(10)[yu]).to(device).float()
                    loss, summaries = vae.loss_encoder(xu)

                    loss.backward()
                    optimizer.step()

                    # Feel free to modify the progress bar

                    pbar.set_postfix(
                        loss='{:.2e}'.format(loss))

                    pbar.update(1)

                    i += 1
                    # Log summaries
                    if i % 50 == 0: ut.log_summaries(writer, summaries, i)

                    if i == iter_max:
                        ut.save_model_by_name(vae, 0)
                        return

                # Save model
                ut.save_model_by_name(vae, iter_save)



