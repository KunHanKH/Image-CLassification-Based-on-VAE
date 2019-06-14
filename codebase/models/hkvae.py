import argparse
import numpy as np
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class HKVAE(nn.Module):
    def __init__(self, nn='v1', name='ssvae', rec_weight=1, kl_xy_x_weight=10,
                 kl_xy_y_weight=10, gen_weight=1, class_weight=100, CNN = False):
        super().__init__()
        self.name = name
        self.CNN = CNN
        self.x_dim = 784
        self.z_dim = 64
        self.y_dim = 10

        self.rec_weight = rec_weight
        self.kl_xy_x_weight = kl_xy_x_weight
        self.kl_xy_y_weight = kl_xy_y_weight

        self.gen_weight = gen_weight
        self.class_weight = class_weight

        nn = getattr(nns, nn)

        if CNN:
            self.enc_xy = nn.Encoder_XY(z_dim=self.z_dim, y_dim=self.y_dim)
            self.enc_x = nn.Encoder_X(z_dim=self.z_dim)
            self.enc_y = nn.Encoder_Y(z_dim=self.z_dim, y_dim=self.y_dim)
        else:
            self.enc_xy = nn.Encoder(z_dim=self.z_dim, y_dim=self.y_dim, x_dim=self.x_dim)
            self.enc_x = nn.Encoder(z_dim=self.z_dim, y_dim=0, x_dim=self.x_dim)
            self.enc_y = nn.Encoder(z_dim=self.z_dim, y_dim=self.y_dim, x_dim=0)

        self.dec = nn.Decoder(z_dim=self.z_dim, y_dim=0, x_dim=self.x_dim)

        self.cls = nn.Classifier(y_dim=self.y_dim, input_dim=self.z_dim*2)


    def negative_elbo_bound(self, x, y):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL_Z, KL_Y and Rec decomposition
        #
        # To assist you in the vectorization of the summation over y, we have
        # the computation of q(y | x) and some tensor tiling code for you.
        #
        # Note that nelbo = kl_z + kl_y + rec
        #
        # Outputs should all be scalar
        ################################################################################

        if self.CNN:
            m_xy, v_xy = self.enc_xy.encode_xy(x, y)
            m_x, v_x = self.enc_x.encode_x(x)
            m_y, v_y = self.enc_y.encode_y(y)
        else:
            m_xy, v_xy = self.enc_xy.encode(x, y)
            m_x, v_x = self.enc_x.encode(x)
            m_y, v_y = self.enc_y.encode(y)

        # kl divergence for latent variable z
        kl_xy_x = ut.kl_normal(m_xy, v_xy, m_x, v_x)
        kl_xy_y = ut.kl_normal(m_xy, v_xy, m_y, v_y)

        # recreation error
        z = ut.sample_gaussian(m_xy, v_xy)
        x_logits = self.dec.decode(z)

        if self.CNN:
            x = torch.reshape(x, (x.shape[0], -1))
        rec = -ut.log_bernoulli_with_logits(x, x_logits)

        kl_xy_x = kl_xy_x.mean()
        kl_xy_y = kl_xy_y.mean()
        rec = rec.mean()
        nelbo = kl_xy_x*self.kl_xy_x_weight + kl_xy_y*self.kl_xy_y_weight + rec*self.rec_weight

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_xy_x, kl_xy_y, rec, m_xy, v_xy

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls.classify(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, y):
        nelbo, kl_xy_x, kl_xy_y, rec, m, v = self.negative_elbo_bound(x, y)

        # classification error
        # concatenate m_xy, v_xy
        mv_cat_x = torch.cat((m, v), dim=1)
        ce = self.classification_cross_entropy(mv_cat_x, y)

        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_xy_x', kl_xy_x),
            ('gen/kl_xy_y', kl_xy_y),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, y):
        if self.CNN:
            m, v = self.enc_y.encode_y(y)
        else:
            m, v = self.enc_y.encode(y)
        return ut.sample_gaussian(m, v)

    def sample_x_given_y(self, y):
        if self.CNN:
            m, v = self.enc_y.encode_y(y)
        else:
            m, v = self.enc_y.encode(y)
        z = ut.sample_gaussian(m, v)
        # return torch.bernoulli(self.compute_sigmoid_given(z))
        return self.compute_sigmoid_given(z)

    def cls_given_x(self, x):
        if self.CNN:
            m, v = self.enc_x.encode_x(x)
        else:
            m, v = self.enc_x.encode(x)
        mv_cat = torch.cat((m, v), dim=1)
        y_logits = self.cls.classify(mv_cat)
        return y_logits.argmax(1)
