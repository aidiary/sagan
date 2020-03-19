import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        # pointwise convolution
        # チャネル数を1/8にして圧縮するConv2d
        # valueは重み付け対象なので元の特徴マップのまま
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # 行方向に規格化する
        self.softmax = nn.Softmax(dim=-2)

        # 元の入力xとself-attention mapであるoを足し算するときの係数
        # output = x + gamma * o
        # 最初は0で学習対象
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        X = x

        # self-attentionなので対象は全てX
        proj_query = self.query_conv(X).view(X.shape[0], -1, X.shape[2] * X.shape[3])
        proj_query = proj_query.permute(0, 2, 1)
        proj_key = self.key_conv(X).view(X.shape[0], -1, X.shape[2] * X.shape[3])

        # Sは類似度行列を意味する
        S = torch.bmm(proj_query, proj_key)

        attention_map_T = self.softmax(S)
        attention_map = attention_map_T.permute(0, 2, 1)

        # attention重みにvalueをかけてattention-mapを得る
        proj_value = self.value_conv(X).view(X.shape[0], -1, X.shape[2] * X.shape[3])
        o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))

        o = o.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
        out = x + self.gamma * o

        return out, attention_map


class Generator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            # ConvTranspose2dの後にSpectral Normalizationを挿入して重みを正規化
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(z_dim, image_size * 8, kernel_size=4, stride=1)),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(image_size * 8,
                                   image_size * 4,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1)), nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(image_size * 4,
                                   image_size * 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1)), nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True))

        self.self_attention1 = SelfAttention(in_dim=image_size * 2)

        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(image_size * 2, image_size, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size), nn.ReLU(inplace=True))

        self.self_attention2 = SelfAttention(in_dim=64)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1, kernel_size=4, stride=2, padding=1), nn.Tanh())

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map1 = self.self_attention1(out)
        out = self.layer4(out)
        out, attention_map2 = self.self_attention2(out)
        out = self.last(out)

        return out, attention_map1, attention_map2


class Discriminator(nn.Module):
    def __init__(self, z_dim=20, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, image_size, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(image_size, image_size * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(image_size * 2, image_size * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.self_attention1 = SelfAttention(in_dim=image_size * 4)

        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(image_size * 4, image_size * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True))

        self.self_attention2 = SelfAttention(in_dim=image_size * 8)

        self.last = nn.Conv2d(image_size * 8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map1 = self.self_attention1(out)
        out = self.layer4(out)
        out, attention_map2 = self.self_attention2(out)
        out = self.last(out)
        return out, attention_map1, attention_map2
