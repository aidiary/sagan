from comet_ml import ExistingExperiment
import matplotlib.pyplot as plt
import torch
from model import Generator, Discriminator
from data import make_datapath_list, GAN_Img_Dataset, ImageTransform

experiment = ExistingExperiment(previous_experiment='e746c2c19f194d588fdfdbb7dc573602')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(z_dim=20, image_size=64)
    G.load_state_dict(torch.load('checkpoints/G.pt'))
    D.load_state_dict(torch.load('checkpoints/D.pt'))
    G.to(device)
    D.to(device)

    batch_size = 8
    z_dim = 20
    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    # generate fake images
    fake_images, am1, am2 = G(fixed_z.to(device))

    # real images
    train_img_list = make_datapath_list()
    mean = (0.5, )
    std = (0.5, )
    train_dataset = GAN_Img_Dataset(file_list=train_img_list, transform=ImageTransform(mean, std))
    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
    imges = next(iter(train_dataloader))

    plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(imges[i][0].cpu().detach().numpy(), 'gray')

        plt.subplot(2, 5, 5 + i + 1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')
    plt.savefig('results.png')
    experiment.log_figure(figure_name='results.png', figure=None, overwrite=True)

    plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')

        plt.subplot(2, 5, 5 + i + 1)
        am = am1[i].view(16, 16, 16, 16)
        am = am[7][7]
        plt.imshow(am.cpu().detach().numpy(), 'Reds')
    experiment.log_figure(figure_name='attention_map.png', figure=None, overwrite=True)


if __name__ == "__main__":
    main()
