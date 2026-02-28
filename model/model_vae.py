import torch
import torch.nn as nn
from torch.nn import functional as F


class CVAE(torch.nn.Module):
    def __init__(self, x_dim=6, cond_dim=3, z_dim=6):
        super(CVAE, self).__init__()

        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.z_dim = z_dim

        # self.encode_sub_module = nn.Sequential(
        #                             nn.Linear(self.x_dim + self.cond_dim, 512),
        #                             nn.LeakyReLU(0.2),
        #                             nn.Linear(512, 256),
        #                             nn.LeakyReLU(0.2),
        #                             nn.Linear(256, 128),
        #                             nn.LeakyReLU(0.2)
        #                         )

        # self.decode_sub_module = nn.Sequential(
        #                             nn.Linear(self.cond_dim + self.z_dim, 128),
        #                             nn.LeakyReLU(0.2),
        #                             nn.Linear(128, 256),
        #                             nn.LeakyReLU(0.2),
        #                             nn.Linear(256, 512),
        #                             nn.LeakyReLU(0.2),
        #                             nn.Linear(512, self.x_dim),
        #                             nn.Sigmoid()
        #                         )

        self.encode_sub_module = nn.Sequential(
                                    nn.Linear(self.x_dim + self.cond_dim, 1024),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(1024, 512),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(512, 256),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(256, 128),
                                    nn.LeakyReLU(0.2)
                                )

        self.decode_sub_module = nn.Sequential(
                                    nn.Linear(self.cond_dim + self.z_dim, 1024),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(1024, 512),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(512, 256),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(256, 128),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(128, self.x_dim),
                                    nn.Sigmoid()
                                )

        self.fc_z_1 = nn.Linear(128, self.z_dim)
        self.fc_z_2 = nn.Linear(128, self.z_dim)
        
    def encode(self, x, cond):
        """
        Args:
            x: joints
            cond: end position
        """         
        input = torch.cat((x, cond), -1)
        h1 = self.encode_sub_module(input)
        return self.fc_z_1(h1), self.fc_z_2(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, cond, z):
        """
        Args:
            cond: end position
            z: latent variable
        """        
        input = torch.cat((cond, z), -1)
        return self.decode_sub_module(input)

    def forward(self, x, cond):
        """
        Args:
            x: joints
            cond: end position
        """
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        return self.decode(cond, z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, mu, logvar):
    # loss_bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    loss_mse = nn.MSELoss()(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return loss_mse + loss_kld


def main():
    model = CVAE()
    recon_x, mu, logvar = model(torch.randn(1, 6), torch.randn(1, 3))

    print('recon_x.shape: ', recon_x.shape)
    print('mu.shape:      ', mu.shape)
    print('logvar.shape:  ', logvar.shape)


if __name__ == "__main__":
    main()
