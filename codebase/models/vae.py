# Copyright (c) 2018 Rui Shu
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np


class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
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
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################

        (m,v) = self.enc.encode(x)  # compute the encoder outut

        kl = torch.mean(ut.kl_normal(m, v, self.z_prior_m, self.z_prior_v), -1)

        z = ut.sample_gaussian(m, v) #sample a point from the multivariate Gaussian
        logits = self.dec.decode(z) #pass the sampled "Z" through the decoder

        rec = -torch.mean(ut.log_bernoulli_with_logits(x, logits), -1) #Calculate log Prob of the output

        nelbo = torch.mean(kl + rec)
        kl = torch.mean(kl)
        rec = torch.mean(rec)
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################


        X_dupl = ut.duplicate(x, iw) # Input "x" is duplicated "iw" times

        (m, v) = self.enc.encode(X_dupl)  # compute the encoder outut

        z = ut.sample_gaussian(m, v)  # sample a point from the multivariate Gaussian
        logits = self.dec.decode(z)  # pass the sampled "Z" through the decoder

        # Calculate log Prob of the output x_hat given latent z
        ln_P_x_z = ut.log_bernoulli_with_logits(X_dupl, logits)

        # Calculate log(P(z))
        #ln_P_z = -torch.sum(z*z, -1)/2.0
        ln_P_z = ut.log_normal(z, self.z_prior_m, self.z_prior_v)

        # Calculate log(Q(z | x)), Conditional Prob of Latent given x
        #ln_q_z_x = -torch.sum((z-m)*(z-m)/(2.0*v) + torch.log(v), -1)
        ln_q_z_x = ut.log_normal(z, m, v)

        exponent = ln_P_x_z + ln_P_z - ln_q_z_x
        exponent = exponent.reshape(iw, -1)

        L_m_x = ut.log_mean_exp(exponent, 0)

        niwae = -torch.mean(L_m_x)
        kl = torch.tensor(0)
        rec = torch.tensor(0)
        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))

    def plot_sampled_x(self, panel_x, panel_y):


        batch = panel_x*panel_y
        x_dim = 28 # dimension along X or Y of MNIST digits
        sampled_x = self.sample_x(batch)
        im_xy = sampled_x.reshape(batch, x_dim,x_dim)
        print(sampled_x.shape, im_xy.shape)
        im_xy = im_xy.detach().numpy()
        #Init the panel of 200 figures to zero
        panel_figure = np.zeros((x_dim * panel_x, x_dim * panel_y))

        #Split the batch of 200 images in 10 rows and 20 cols
        for i in range(panel_x):
            for j in range(panel_y):
                panel_figure[i*x_dim:(i+1)*x_dim, j*x_dim:(j+1)*x_dim] = im_xy[i*panel_y+j,:,:]

        plt.figure(figsize=(10, 20))
        plt.imshow(panel_figure, cmap='Greys_r')
        #plt.plot(panel_figure)
        plt.show()
        #plt.savefig('./VAE_P1.png')

