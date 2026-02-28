import torch
import torch.nn as nn


class Generator(torch.nn.Module):
    def __init__(self, target_dim=6, z_dim=6, output_dim=6, net_arch=None, layer_norm=False, use_dropout=False):
        super(Generator, self).__init__()
        
        if net_arch is None:
            # net_arch = [128, 256, 256]
            # net_arch = [512, 256, 128]
            net_arch = [1024, 512, 256, 128]

        net_modules = [nn.Linear(target_dim + z_dim, net_arch[0]), nn.LeakyReLU(0.2, inplace=True)]
        
        for idx in range(len(net_arch) - 1):
            net_modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            if use_dropout:
                net_modules.append(nn.Dropout(p=0.2))
            net_modules.append(nn.LeakyReLU(0.2, inplace=True))
            if layer_norm:
                net_modules.append(nn.LayerNorm(net_arch[idx + 1]))
        
        net_modules.append(nn.Linear(net_arch[-1], output_dim))
        net_modules.append(nn.Sigmoid())

        self.net = nn.Sequential(*net_modules)

    def forward(self, end_position, noise):
        input = torch.cat((end_position, noise), -1)
        return self.net(input)


class Discriminator(torch.nn.Module):
    def __init__(self, joint_dim = 6, target_dim=3, output_dim=1, net_arch=None, layer_norm=False, use_dropout=False):
        super(Discriminator, self).__init__()
        
        if net_arch is None:
            # net_arch = [128, 256, 256]
            # net_arch = [512, 256, 128]
            net_arch = [1024, 512, 256, 128]

        net_modules = [nn.Linear(joint_dim + target_dim, net_arch[0]), nn.LeakyReLU(0.2, inplace=True)]
        
        for idx in range(len(net_arch) - 1):
            net_modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            if use_dropout:
                net_modules.append(nn.Dropout(p=0.2))
            net_modules.append(nn.LeakyReLU(0.2, inplace=True))
            if layer_norm:
                net_modules.append(nn.LayerNorm(net_arch[idx + 1]))
        
        net_modules.append(nn.Linear(net_arch[-1], output_dim))
        net_modules.append(nn.Sigmoid())

        self.net = nn.Sequential(*net_modules)

    def forward(self, joint_angle, end_position):
        input = torch.cat((joint_angle, end_position), -1)
        output = self.net(input)
        return output.view(-1, 1).squeeze(1)


def main():
    model_G = Generator()
    print(model_G)
    output_G = model_G(torch.randn(10, 3), torch.randn(10, 6))
    print(output_G.shape)  # torch.Size([10, 6])

    print('------')

    model_D = Discriminator()
    print(model_D)
    output_D = model_D(torch.randn(10, 6), torch.randn(10, 3))
    print(output_D.shape)  # torch.Size([10])

    print('------')

    model_G = Generator()
    print(model_G)
    output_G = model_G(torch.randn(3), torch.randn(6))
    print(output_G.shape)  # torch.Size([6])

    print('------')

    model_D = Discriminator()
    print(model_D)
    output_D = model_D(torch.randn(6), torch.randn(3))
    print(output_D.shape)  # torch.Size([1])


if __name__ == "__main__":
    main()
