from comet_ml import Experiment

import os
import torch
import torch.nn as nn
from model import Generator, Discriminator
from data import make_datapath_list, ImageTransform, GAN_Img_Dataset

experiment = Experiment(project_name="sagan", auto_metric_logging=False)


def main():
    os.makedirs('checkpoints', exist_ok=True)

    # create models
    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(z_dim=20, image_size=64)
    G.apply(weights_init)
    D.apply(weights_init)
    print('*** initialize weights')

    # load data
    train_img_list = make_datapath_list()
    print('*** num of data:', len(train_img_list))

    mean = (0.5, )
    std = (0.5, )
    train_dataset = GAN_Img_Dataset(file_list=train_img_list, transform=ImageTransform(mean, std))

    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

    num_epochs = 300
    G_update, D_update = train_model(G, D, train_dataloader, num_epochs)

    torch.save(G.state_dict(), 'checkpoints/G.pt')
    torch.save(D.state_dict(), 'checkpoints/D.pt')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_model(G, D, dataloader, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('***', device)

    # optimizer
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    # loss function
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # hyperparameter
    z_dim = 20
    mini_batch_size = 64

    G.to(device)
    D.to(device)

    G.train()
    D.train()

    torch.backends.cudnn.benchmark = True

    batch_size = dataloader.batch_size
    iteration = 1

    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        for imges in dataloader:
            # batchのサイズが1だとBatchNormalizationがエラーになるので無視
            if imges.size()[0] == 1:
                continue
            imges = imges.to(device)

            # Discriminatorの学習
            mini_batch_size = imges.size()[0]

            # 真の画像を判定
            # d_out_realが大きな値になるとd_loss_realが小さくなる
            d_out_real, _, _ = D(imges)

            # 偽の画像を生成して判定
            # d_out_fakeが小さい値になるとd_loss_fakeが小さくなる
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images, _, _ = G(input_z)
            d_out_fake, _, _ = D(fake_images)

            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            d_loss = d_loss_real + d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # Generatorの学習
            # 偽の画像を生成
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images, _, _ = G(input_z)
            d_out_fake, _, _ = D(fake_images)

            # d_out_fakeが大きい値になるとg_lossが小さくなる
            g_loss = -d_out_fake.mean()

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # optimizerに対してGのパラメータしか与えていないためPyTorchでは
            # Dのパラメータの固定処理が不要（更新対象ではないため更新されない）
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        print('Epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G_Loss:{:.4f}'.format(
            epoch, epoch_d_loss / batch_size, epoch_g_loss / batch_size))

        experiment.log_metric('d_loss', epoch_d_loss / batch_size, step=epoch)
        experiment.log_metric('g_loss', epoch_g_loss / batch_size, step=epoch)

    return G, D


if __name__ == "__main__":
    main()
