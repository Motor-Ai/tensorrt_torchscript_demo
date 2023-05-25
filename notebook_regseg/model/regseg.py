import torch
import torch.nn as nn
from torch.nn import functional as F


def generate_stage2(ds, block_fun):
    blocks = []
    for d in ds:
        blocks.append(block_fun(d))
    return blocks


class RegSeg(nn.Module):
    def __init__(self, num_classes):
        super(RegSeg, self).__init__()
        self.stem = ConvBnAct(in_channels=3,
                              out_channels=32,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.body = RegSegBody([[1], [1, 2]] + 4 * [[1, 4]] + 7 * [[1, 14]])
        self.decoder = Decoder(num_classes, self.body.channels())

    def forward(self, x):
        # input_shape = x.shape[-2:]
        x = self.stem(x)
        x = self.body(x)
        x = self.decoder(x)
        # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        x = torch.argmax(x, dim=1).type(torch.float32)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, channels):
        super(Decoder, self).__init__()
        channels4, channels8, channels16 = channels["4"], channels["8"], channels["16"]
        self.head16 = ConvBnAct(channels16, 128, 1)
        self.head8 = ConvBnAct(channels8, 128, 1)
        self.head4 = ConvBnAct(channels4, 8, 1)
        self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv4 = ConvBnAct(64+8, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x4, x8, x16 = x["4"], x["8"], x["16"]

        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)

        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        x4 = self.classifier(x4)
        return x4


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.batch_norm(self.conv(x))
        return self.act(x)


class RegSegBody(nn.Module):
    def __init__(self, ds):
        super(RegSegBody, self).__init__()
        group_width = 16
        self.stage4 = DBlock(in_channels=32,
                             out_channels=48,
                             dilations=[1],
                             group_width=group_width,
                             stride=2)
        self.stage8 = nn.Sequential(
            DBlock(in_channels=48, out_channels=128, dilations=[1], group_width=group_width, stride=2),
            DBlock(in_channels=128, out_channels=128, dilations=[1], group_width=group_width, stride=1),
            DBlock(in_channels=128, out_channels=128, dilations=[1], group_width=group_width, stride=1)
        )
        self.stage16 = nn.Sequential(
            DBlock(in_channels=128, out_channels=256, dilations=[1], group_width=group_width, stride=2),
            *generate_stage2(ds[:-1], lambda d: DBlock(256, 256, d, group_width, 1)),
            DBlock(256, 320, ds[-1], group_width, 1)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)

        return {"4": x4, "8": x8, "16": x16}

    def channels(self):
        return {"4": 48, "8": 128, "16": 320}


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, group_width, stride, attention="gct"):
        super(DBlock, self).__init__()
        avg_downsample = True
        groups = out_channels // group_width
        if attention == "gct":
            self.gct = GCT(in_channels)
        else:
            self.gct = None
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        if len(dilations) == 1:
            dilation = dilations[0]
            self.conv2=nn.Conv2d(out_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 groups=groups,
                                 padding=dilation,
                                 dilation=dilation,
                                 bias=False)
        else:
            self.conv2 = DilatedConv(out_channels, dilations, group_width=group_width, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU(inplace=True)

        if attention == "se":
            self.se = SqueezeExcitationModule(out_channels, in_channels // 4)
        else:
            self.se = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        if self.gct is not None:
            x = self.gct(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        if self.se is not None:
            x = self.se(x)
        x = self.bn3(self.conv3(x))
        x = self.act3(x + shortcut)
        return x


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg = nn.AvgPool2d(2, 2, ceil_mode=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.avg = None
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.avg is not None:
            x = self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class SqueezeExcitationModule(nn.Module):
    def __init__(self, w_in, w_se):
        super(SqueezeExcitationModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(w_in, w_se, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(w_se, w_in, 1, bias=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.act1(self.conv1(y))
        y = self.act2(self.conv2(y))
        return x * y


class DilatedConv(nn.Module):
    def __init__(self, w, dilations, group_width, stride, bias):
        super(DilatedConv, self).__init__()
        self.num_splits = len(dilations)
        assert (w % self.num_splits == 0)
        temp = w // self.num_splits
        assert (temp % group_width == 0)
        groups = temp // group_width
        convs = []
        for d in dilations:
            convs.append(nn.Conv2d(temp, temp, 3, padding=d, dilation=d, stride=stride, bias=bias, groups=groups))
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        x = torch.tensor_split(x, self.num_splits, dim=1)
        outputs = []
        for i in range(self.num_splits):
            outputs.append(self.convs[i](x[i]))
        return torch.cat(outputs, dim=1)


class GCT(nn.Module):
    """ Taken from paper https://arxiv.org/pdf/1909.11519.pdf (Gated Channel Transformation for Visual Recognition)
       Repo: https://github.com/z-x-yang/GCT """

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate
