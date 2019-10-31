# Copyright (c) 2018 Rui Shu
import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

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
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # Compute the mixture of Gaussian prior

        (m, v) = self.enc.encode(x)  # compute the encoder output
        #print(" ***** \n")
        #print("x xhape ", x.shape)
        #print("m and v shapes = ", m.shape, v.shape)
        prior = ut.gaussian_parameters(self.z_pre, dim=1)

        #print("prior shapes = ", prior[0].shape, prior[1].shape)
        z = ut.sample_gaussian(m, v)  # sample a point from the multivariate Gaussian
        #print("shape of z = ",z.shape)
        logits = self.dec.decode(z)  # pass the sampled "Z" through the decoder

        #print("logits shape = ", logits.shape)
        rec = -torch.mean(ut.log_bernoulli_with_logits(x, logits), -1)  # Calculate log Prob of the output

        log_prob = ut.log_normal(z, m, v)
        log_prob  -= ut.log_normal_mixture(z, prior[0], prior[1])

        kl = torch.mean(log_prob)

        rec = torch.mean(rec)

        nelbo = kl + rec
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
        # Compute the mixture of Gaussian prior
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
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
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

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