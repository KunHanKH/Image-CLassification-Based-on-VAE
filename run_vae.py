import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.train import train
from codebase.refine import refine
from pprint import pprint
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision
from codebase.utils import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,     help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=1000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
args = parser.parse_args()
layout = [
    ('model={:s}',  'vae'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = datasets.MNIST(
    root='../MNIST-data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

test_set = datasets.MNIST(
    root='../MNIST-data'
    ,train=False
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader, labeled_subset, _ = ut.get_mnist_data(device, train_set, test_set, use_test_subset=True)

data_individual_length = 5000
# Balance datasets, each digit has 5000 examples.
data_set_individual, data_loader_individual = generate_individual_set_loader(device, train_set, data_individual_length)

z_prior_m = torch.nn.Parameter(torch.zeros(args.z), requires_grad=False).to(device)
z_prior_v = torch.nn.Parameter(torch.ones(args.z), requires_grad=False).to(device)
vae = VAE(z_dim=args.z, name=model_name,
          z_prior_m=z_prior_m, z_prior_v=z_prior_v).to(device)

# train_args:
# 1 -> step 1: get the model
# 2 -> step 2: get mean and variance
# 3 -> step 3: refine the model

# train_args = 1
train_args = None
if train_args == 1:
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train(model=vae,
          train_loader=train_loader,
          # train_loader=data_loader_individual[0],
          labeled_subset=labeled_subset,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=10000,
          iter_save=args.iter_save)
    ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=args.train == 2)

train_args = 2
# train_args = None
mean_set = []
variance_set = []
if train_args == 2:
    ut.load_model_by_name(vae, global_step=20000)
    para_set = [get_mean_variance(vae, data_set_individual[i]) for i in range(10)]
    for i, set in enumerate(para_set):
        temp_mean, temp_variance = ut.resample(10, set[0], set[1])
        mean_set.append(temp_mean)
        variance_set.append(temp_variance)

train_args = 3
# train_args = None
if train_args == 3:
    writer = ut.prepare_writer(model_name, overwrite_existing=False)
    refine(train_loader_set=data_loader_individual,
           # train_loader=data_loader_individual[0],
           mean_set=mean_set,
           variance_set=variance_set,
           z_dim=args.z,
           device=device,
           tqdm=tqdm.tqdm,
           writer=writer,
           iter_max=5000,
           iter_save=1000,
           model_name=model_name)

# train_args = 4
# train_args = None
if train_args == 4:
    z_prior_m = torch.nn.Parameter(torch.zeros(args.z), requires_grad=False).to(device)
    z_prior_v = torch.nn.Parameter(torch.ones(args.z), requires_grad=False).to(device)
    vae = VAE(z_dim=args.z, name=model_name,
              z_prior_m=z_prior_m, z_prior_v=z_prior_v).to(device)
    ut.load_model_by_name(vae, global_step=args.iter_max)
    print("Step 1 model:")
    ut.evaluate_lower_bound(vae, labeled_subset)

    print("Step 2 model:")
    ut.load_model_by_name(vae, global_step=0)
    ut.evaluate_lower_bound(vae, labeled_subset)





# else:
#     ut.load_model_by_name(vae, global_step=args.iter_max)
#     ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=True)
#     # ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=args.train == 2)
#     x = vae.sample_x(200)
#     x = x.view(20, 10, 28, 28).cpu().detach().numpy()
#     fig, axes = plt.subplots(20, 10)
#     for i in range(10):
#         for j in range(10):
#             axes[i, j].imshow(x[i][j])
#             axes[i, j].set_xticks([])
#             axes[i, j].set_yticks([])
#     plt.show()
