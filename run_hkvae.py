import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.hkvae import HKVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--rec_step',       type=int, default=10,        help="Weight on the generative terms")
parser.add_argument('--gw',             type=int, default=1,        help="Weight on the generative terms")
parser.add_argument('--cw',             type=int, default=50,      help="Weight on the class term")
parser.add_argument('--recw',            type=int, default=10,        help="Weight on the generative terms")
parser.add_argument('--kl_xy_xw',        type=int, default=25,       help="Weight on the class term")
parser.add_argument('--kl_xy_yw',        type=int, default=50,       help="Weight on the generative terms")
parser.add_argument('--iter_max',       type=int, default=10000,    help="Number of training iterations")
parser.add_argument('--iter_save',      type=int, default=2000,    help="Save model every n iterations")
parser.add_argument('--run',            type=int, default=0,        help="Run ID. In case you want to run replicates")
parser.add_argument('--train',          type=int, default=1,        help="Flag for training")
args = parser.parse_args()
layout = [
    ('model={:s}',  'hkvae'),
    ('rec_step={:03d}', args.rec_step),
    ('gw={:03d}', args.gw),
    ('cw={:03d}', args.cw),
    ('kl_xy_xw={:03d}', args.kl_xy_xw),
    ('kl_xy_yw={:03d}', args.kl_xy_yw),
    ('recw={:03d}', args.recw),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = datasets.FashionMNIST(
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

CNN = False

train_loader, labeled_subset, test_set = ut.get_mnist_data(device, train_set, test_set,
                                                           use_test_subset=True,
                                                           CNN=CNN)

hkvae = HKVAE(
    rec_weight=args.recw,
    kl_xy_x_weight=args.kl_xy_xw,
    kl_xy_y_weight=args.kl_xy_yw,
    gen_weight=args.gw,
    class_weight=args.cw,
    name=model_name,
    CNN=CNN).to(device)

Train = True
if Train:
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train(model=hkvae,
          train_loader=train_loader,
          labeled_subset=labeled_subset,
          device=device,
          y_status='hk',
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save,
          rec_step=args.rec_step,
          CNN=CNN)
else:
    ut.load_model_by_name(hkvae, args.iter_max)

# pprint(vars(args))
# print('Model name:', model_name)
# print(hkvae.CNN)
# xl, yl = test_set
# yl = torch.tensor(np.eye(10)[yl]).float().to(device)
# test_set = (xl, yl)
# ut.evaluate_lower_bound_HK(hkvae, test_set)
# ut.evaluate_classifier_HK(hkvae, test_set)
