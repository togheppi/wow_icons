from base_networks import *
import utils
import numpy as np

class DCGANGenerator(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim):
        super(DCGANGenerator, self).__init__()

        self.layers = torch.nn.Sequential(
            DeconvBlock(input_dim, base_filter * 8, 4, 1, 0),
            DeconvBlock(base_filter * 8, base_filter * 4),
            DeconvBlock(base_filter * 4, base_filter * 2),
            DeconvBlock(base_filter * 2, base_filter),
            DeconvBlock(base_filter, output_dim, activation='tanh', norm=None)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class DCGANDiscriminator(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim):
        super(DCGANDiscriminator, self).__init__()

        self.layers = torch.nn.Sequential(
            ConvBlock(input_dim, base_filter, activation='lrelu', norm=None),
            ConvBlock(base_filter, base_filter * 2, activation='lrelu'),
            ConvBlock(base_filter * 2, base_filter * 4, activation='lrelu'),
            ConvBlock(base_filter * 4, base_filter * 8, activation='lrelu'),
            ConvBlock(base_filter * 8, output_dim, 4, 1, 0, activation='sigmoid', norm=None)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class animeGANGenerator(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim, upsample='deconv'):
        super(animeGANGenerator, self).__init__()
        self.input_dim = input_dim
        self.base_filter = base_filter

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.base_filter * 8 * 4 * 4),
            torch.nn.BatchNorm1d(self.base_filter * 8 * 4 * 4),
            torch.nn.LeakyReLU(0.2, True)
        )

        self.upsample = torch.nn.Sequential(
            Upsample2xBlock(base_filter * 8, base_filter * 4, upsample=upsample, activation='lrelu'),   # BxCx8x8
            Upsample2xBlock(base_filter * 4, base_filter * 2, upsample=upsample, activation='lrelu'),   # BxCx16x16
            Upsample2xBlock(base_filter * 2, base_filter, upsample=upsample, activation='lrelu'),       # BxCx32x32
            ConvBlock(base_filter, base_filter, 3, 1, 1, activation='lrelu'),
            Upsample2xBlock(base_filter, output_dim, upsample=upsample, activation='tanh', norm=None)  # BxCx64x64
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        out = self.fc(x)
        out = out.view(-1, self.base_filter * 8, 4, 4)   # BxCx4x4
        out = self.upsample(out)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class animeGANDiscriminator(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim):
        super(animeGANDiscriminator, self).__init__()

        self.layers = torch.nn.Sequential(
            ConvBlock(input_dim, base_filter, activation='lrelu', norm=None),
            ConvBlock(base_filter, base_filter * 2, activation='lrelu'),
            ConvBlock(base_filter * 2, base_filter * 4, activation='lrelu'),
            ConvBlock(base_filter * 4, base_filter * 8, activation='lrelu'),
            ConvBlock(base_filter * 8, output_dim, 4, 1, 0, activation='sigmoid', norm=None)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class DRAGANGenerator(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim, image_size):
        super(DRAGANGenerator, self).__init__()
        self.base_filter = base_filter
        self.image_size = image_size
        self.input_dim = input_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.base_filter * 8),
            torch.nn.BatchNorm1d(self.base_filter * 8),
            torch.nn.ReLU(),
            torch.nn.Linear(self.base_filter * 8, self.base_filter * self.image_size // 4 * self.image_size // 4),
            torch.nn.BatchNorm1d(self.base_filter * self.image_size // 4 * self.image_size // 4),
            torch.nn.ReLU()
        )

        self.deconv = torch.nn.Sequential(
            DeconvBlock(self.base_filter, self.base_filter // 2, activation='relu'),
            DeconvBlock(self.base_filter // 2, output_dim, activation='tanh', norm=None)
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        out = self.fc(x)
        out = out.view(-1, self.base_filter, self.image_size // 4, self.image_size // 4)
        out = self.deconv(out)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class DRAGANDiscriminator(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim, image_size):
        super(DRAGANDiscriminator, self).__init__()
        self.base_filter = base_filter
        self.image_size = image_size

        self.conv = torch.nn.Sequential(
            ConvBlock(input_dim, self.base_filter, activation='lrelu', norm=None),
            ConvBlock(self.base_filter, self.base_filter * 2, activation='lrelu'),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.base_filter * 2 * self.image_size // 4 * self.image_size // 4, self.base_filter * 16),
            torch.nn.BatchNorm1d(self.base_filter * 16),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(self.base_filter * 16, output_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, self.base_filter * 2 * self.image_size // 4 * self.image_size // 4)
        out = self.fc(out)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class ChainerDRAGANGenerator(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim, up_layers=4, norm='batch'):
        super(ChainerDRAGANGenerator, self).__init__()
        self.base_filter = base_filter
        self.input_dim = input_dim

        # Input fully-connected layer
        self.input_fc = DenseBlock(input_dim, base_filter * 4 * 4, norm=norm)

        # Upsampling layer
        upsample_blocks = []
        num_filter = base_filter
        for i in range(up_layers - 1):
            upsample_blocks.append(Upsample2xBlock(num_filter, num_filter // 2, norm=norm))
            num_filter = num_filter // 2
        self.upsample = torch.nn.Sequential(*upsample_blocks)

        self.upsample.add_module(Upsample2xBlock(base_filter, output_dim, activation='tanh', norm=None))

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        out = self.input_fc(x)
        out = out.view(-1, self.base_filter, 4, 4)
        out = self.upsample(out)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class ChainerDRAGANDiscriminator(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim, down_layers=4, norm='batch'):
        super(ChainerDRAGANDiscriminator, self).__init__()
        self.base_filter = base_filter

        # Convolution layers
        self.conv = ConvBlock(input_dim, base_filter, activation='lrelu', norm=None)

        conv_blocks = []
        num_filter = base_filter
        for i in range(down_layers - 1):
            conv_blocks.append(ConvBlock(num_filter, num_filter * 2, activation='lrelu', norm=norm))
            num_filter = num_filter * 2
        self.downsample = torch.nn.Sequential(*conv_blocks)

        # Output fully-connected layer
        self.output_fc = DenseBlock(base_filter * 8 * 4 * 4, output_dim, activation='sigmoid', norm=None)

    def forward(self, x):
        out = self.conv(x)
        out = self.downsample(out)
        out = out.view(-1, self.base_filter * 8 * 4 * 4)
        out = self.output_fc(out)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class MoeGANGenerator(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim, num_resnet, upsample='deconv', norm=None):
        super(MoeGANGenerator, self).__init__()
        self.input_dim = input_dim
        self.base_filter = base_filter

        # Input fully-connected layer
        self.input_fc = DenseBlock(input_dim, base_filter * 16 * 16, norm=norm)

        # Resnet blocks
        resnet_blocks = []
        for i in range(num_resnet):
            resnet_blocks.append(ResnetBlock(base_filter, norm=norm))
        self.resnet_blocks = torch.nn.Sequential(*resnet_blocks)

        # Middle convolution layer
        self.mid_conv = ConvBlock(base_filter, base_filter, 3, 1, 1)

        # Upsampling layer
        self.upsample = torch.nn.Sequential(
            Upsample2xBlock(base_filter, base_filter, upsample=upsample, norm=norm),
            Upsample2xBlock(base_filter, base_filter, upsample=upsample, norm=norm),
            Upsample2xBlock(base_filter, base_filter, upsample=upsample, norm=norm)
        )

        # Output convolution layer
        self.output_conv = ConvBlock(base_filter, output_dim, 9, 1, 4, activation='tanh', norm=norm)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        out = self.input_fc(x)
        out = out.view(-1, self.base_filter, 16, 16)
        residual = out
        out = self.resnet_blocks(out)
        out = self.mid_conv(out)
        out = torch.add(out, residual)
        out = self.upsample(out)
        out = self.output_conv(out)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class MoeGANDiscriminator(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim, norm=None):
        super(MoeGANDiscriminator, self).__init__()
        self.base_filter = base_filter

        # Convolution layers
        self.conv_blocks1 = torch.nn.Sequential(
            ConvBlock(input_dim, base_filter, 4, 2, 1, activation='lrelu', norm=None),
            ResnetBlock(base_filter, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True),
            ResnetBlock(base_filter, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True)
        )

        self.conv_blocks2 = torch.nn.Sequential(
            ConvBlock(base_filter, base_filter * 2, 4, 2, 1, activation='lrelu', norm=norm),
            ResnetBlock(base_filter * 2, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True),
            ResnetBlock(base_filter * 2, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True)
        )

        self.conv_blocks3 = torch.nn.Sequential(
            ConvBlock(base_filter * 2, base_filter * 4, 4, 2, 1, activation='lrelu', norm=norm),
            ResnetBlock(base_filter * 4, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True),
            ResnetBlock(base_filter * 4, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True)
        )

        self.conv_blocks4 = torch.nn.Sequential(
            ConvBlock(base_filter * 4, base_filter * 8, 3, 2, 1, activation='lrelu', norm=norm),
            ResnetBlock(base_filter * 8, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True),
            ResnetBlock(base_filter * 8, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True)
        )

        self.conv_blocks5 = torch.nn.Sequential(
            ConvBlock(base_filter * 8, base_filter * 16, 3, 2, 1, activation='lrelu', norm=norm),
            ResnetBlock(base_filter * 16, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True),
            ResnetBlock(base_filter * 16, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True)
        )

        # Output convolution layer
        self.output_conv = ConvBlock(base_filter * 16, base_filter * 32, 3, 2, 1, activation='lrelu', norm=None)
        # Output fully-connected layer
        self.output_fc = DenseBlock(base_filter * 32 * 2 * 2, output_dim, activation='sigmoid', norm=None)

    def forward(self, x):
        out = self.conv_blocks1(x)
        out = self.conv_blocks2(out)
        out = self.conv_blocks3(out)
        out = self.conv_blocks4(out)
        out = self.conv_blocks5(out)
        out = self.output_conv(out)
        out = out.view(out.size()[0], -1)
        out = self.output_fc(out)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class MoeGANGenerator2(torch.nn.Module):
    def __init__(self, input_dim, tag_dim, base_filter, output_dim, num_resnet, upsample='ps', norm='batch'):
        super(MoeGANGenerator2, self).__init__()
        self.input_dim = input_dim
        self.tag_dim = tag_dim
        self.base_filter = base_filter

        # Input fully-connected layer
        self.input_fc = DenseBlock(input_dim + tag_dim, base_filter * 16 * 16, norm=norm)

        # Resnet blocks
        resnet_blocks = []
        for i in range(num_resnet):
            resnet_blocks.append(ResnetBlock(base_filter, norm=norm))
        self.resnet_blocks = torch.nn.Sequential(*resnet_blocks)

        # Middle convolution layer
        self.mid_conv = ConvBlock(base_filter, base_filter, 3, 1, 1, norm=norm)

        # Upsampling layer
        self.upsample = torch.nn.Sequential(
            Upsample2xBlock(base_filter, base_filter * 4, upsample=upsample, norm=norm),
            Upsample2xBlock(base_filter * 4, base_filter * 4, upsample=upsample, norm=norm),
            Upsample2xBlock(base_filter * 4, base_filter * 4, upsample=upsample, norm=norm)
        )

        # Output convolution layer
        self.output_conv = ConvBlock(base_filter * 4, output_dim, 9, 1, 4, activation='tanh', norm=None)

    def forward(self, input, label):
        input = input.view(-1, self.input_dim)
        x = torch.cat([input, label], 1)
        out = self.input_fc(x)
        out = out.view(-1, self.base_filter, 16, 16)
        residual = out
        out = self.resnet_blocks(out)
        out = self.mid_conv(out)
        out = torch.add(out, residual)
        out = self.upsample(out)
        out = self.output_conv(out)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class MoeGANDiscriminator2(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim, tag_dims, norm=None):
        super(MoeGANDiscriminator2, self).__init__()
        self.base_filter = base_filter

        # Convolution layers
        self.conv_blocks1 = torch.nn.Sequential(
            ConvBlock(input_dim, base_filter, 4, 2, 1, activation='lrelu', norm=None),
            ResnetBlock(base_filter, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True),
            ResnetBlock(base_filter, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True)
        )

        self.conv_blocks2 = torch.nn.Sequential(
            ConvBlock(base_filter, base_filter * 2, 4, 2, 1, activation='lrelu', norm=norm),
            ResnetBlock(base_filter * 2, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True),
            ResnetBlock(base_filter * 2, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True)
        )

        self.conv_blocks3 = torch.nn.Sequential(
            ConvBlock(base_filter * 2, base_filter * 4, 4, 2, 1, activation='lrelu', norm=norm),
            ResnetBlock(base_filter * 4, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True),
            ResnetBlock(base_filter * 4, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True)
        )

        self.conv_blocks4 = torch.nn.Sequential(
            ConvBlock(base_filter * 4, base_filter * 8, 3, 2, 1, activation='lrelu', norm=norm),
            ResnetBlock(base_filter * 8, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True),
            ResnetBlock(base_filter * 8, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True)
        )

        self.conv_blocks5 = torch.nn.Sequential(
            ConvBlock(base_filter * 8, base_filter * 16, 3, 2, 1, activation='lrelu', norm=norm),
            ResnetBlock(base_filter * 16, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True),
            ResnetBlock(base_filter * 16, activation='lrelu', norm=norm),
            torch.nn.LeakyReLU(0.2, True)
        )

        # Output convolution layer
        self.output_conv = ConvBlock(base_filter * 16, base_filter * 32, 3, 2, 1, activation='lrelu', norm=None)
        # Fully-connected layer for output
        self.output_fc = DenseBlock(base_filter * 32 * 2 * 2, output_dim, activation='sigmoid', norm=None)
        # Fully-connected layer for labels
        self.label_fc1 = DenseBlock(base_filter * 32 * 2 * 2, tag_dims[0], activation=None, norm=None)
        self.label_fc2 = DenseBlock(base_filter * 32 * 2 * 2, tag_dims[1], activation=None, norm=None)
        self.label_fc3 = DenseBlock(base_filter * 32 * 2 * 2, tag_dims[2], activation=None, norm=None)

    def forward(self, input):
        x = self.conv_blocks1(input)
        x = self.conv_blocks2(x)
        x = self.conv_blocks3(x)
        x = self.conv_blocks4(x)
        x = self.conv_blocks5(x)
        x = self.output_conv(x)
        x = x.view(x.size()[0], -1)
        out = self.output_fc(x)
        label1 = self.label_fc1(x)
        label2 = self.label_fc2(x)
        label3 = self.label_fc3(x)
        # label = torch.cat([label1, label2, label3], 1)
        aa = out.data.cpu()
        if np.isnan(aa.numpy()[0]):
            print('nan detected!!')
        return out, (label1, label2, label3)

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class animeGANGenerator2(torch.nn.Module):
    def __init__(self, input_dim, tag_dim, base_filter, output_dim, upsample='deconv', norm='batch'):
        super(animeGANGenerator2, self).__init__()
        self.input_dim = input_dim
        self.tag_dim = tag_dim
        self.base_filter = base_filter

        self.fc = DenseBlock(input_dim + tag_dim, self.base_filter * 8 * 4 * 4, activation='lrelu', norm=norm)

        self.upsample = torch.nn.Sequential(
            # Upsample2xBlock(base_filter * 16, base_filter * 8, upsample=upsample, activation='lrelu', norm=norm),   # BxCx8x8
            Upsample2xBlock(base_filter * 8, base_filter * 4, upsample=upsample, activation='lrelu', norm=norm),   # BxCx16x16
            Upsample2xBlock(base_filter * 4, base_filter * 2, upsample=upsample, activation='lrelu', norm=norm),       # BxCx32x32
            Upsample2xBlock(base_filter * 2, base_filter, upsample=upsample, activation='lrelu', norm=norm),            # BxCx64x64
            ConvBlock(base_filter, base_filter, 3, 1, 1, activation='lrelu', norm=norm),
            ConvBlock(base_filter, base_filter, 3, 1, 1, activation='lrelu', norm=norm),
            ConvBlock(base_filter, base_filter, 3, 1, 1, activation='lrelu', norm=norm),
            ConvBlock(base_filter, base_filter, 3, 1, 1, activation='lrelu', norm=norm),
            ConvBlock(base_filter, base_filter, 3, 1, 1, activation='lrelu', norm=norm),
            Upsample2xBlock(base_filter, output_dim, upsample=upsample, activation='tanh', norm=None)   # BxCx128x128
        )

    def forward(self, input, label):
        input = input.view(-1, self.input_dim)
        x = torch.cat([input, label], 1)
        out = self.fc(x)
        out = out.view(-1, self.base_filter * 8, 4, 4)   # BxCx4x4
        out = self.upsample(out)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class animeGANDiscriminator2(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim, tag_dims, norm='batch'):
        super(animeGANDiscriminator2, self).__init__()

        self.layers = torch.nn.Sequential(
            ConvBlock(input_dim, base_filter, activation='lrelu', norm=None),
            ConvBlock(base_filter, base_filter, 3, 1, 1, activation='lrelu', norm=norm),
            ConvBlock(base_filter, base_filter, 3, 1, 1, activation='lrelu', norm=norm),
            ConvBlock(base_filter, base_filter, 3, 1, 1, activation='lrelu', norm=norm),
            ConvBlock(base_filter, base_filter * 2, activation='lrelu', norm=norm),
            ConvBlock(base_filter * 2, base_filter * 4, activation='lrelu', norm=norm),
            ConvBlock(base_filter * 4, base_filter * 8, activation='lrelu', norm=norm),
            # ConvBlock(base_filter * 8, base_filter * 16, activation='lrelu', norm=norm)
        )
        # Output conv
        self.output_conv = ConvBlock(base_filter * 8, output_dim, 4, 1, 0, activation='sigmoid', norm=None)

        # Fully-connected layer for labels
        self.label_fc1 = DenseBlock(base_filter * 8 * 4 * 4, tag_dims[0], activation=None, norm=None)
        # self.label_fc2 = DenseBlock(base_filter * 8 * 4 * 4, tag_dims[1], activation=None, norm=None)
        # self.label_fc3 = DenseBlock(base_filter * 8 * 4 * 4, tag_dims[2], activation=None, norm=None)

    def forward(self, input):
        x = self.layers(input)
        out = self.output_conv(x)
        x = x.view(x.size()[0], -1)
        label1 = self.label_fc1(x)
        # label2 = self.label_fc2(x)
        # label3 = self.label_fc3(x)
        return out, label1

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class ChainerDRAGANGenerator2(torch.nn.Module):
    def __init__(self, input_dim, tag_dim, base_filter, output_dim, up_layers=4, norm='batch', upsample='deconv'):
        super(ChainerDRAGANGenerator2, self).__init__()
        self.base_filter = base_filter
        self.input_dim = input_dim
        self.tag_dim = tag_dim

        # Input fully-connected layer
        self.input_fc = DenseBlock(input_dim + tag_dim, base_filter * 4 * 4, norm=norm)

        # Upsampling layer
        upsample_blocks = []
        num_filter = base_filter
        for i in range(up_layers - 1):
            upsample_blocks.append(Upsample2xBlock(num_filter, num_filter // 2, upsample=upsample, norm=norm))
            num_filter = num_filter // 2
        upsample_blocks.append(Upsample2xBlock(num_filter, output_dim, upsample=upsample, activation='tanh', norm=None))
        self.upsample = torch.nn.Sequential(*upsample_blocks)

    def forward(self, input, label):
        input = input.view(-1, self.input_dim)
        x = torch.cat([input, label], 1)
        out = self.input_fc(x)
        out = out.view(-1, self.base_filter, 4, 4)
        out = self.upsample(out)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)


class ChainerDRAGANDiscriminator2(torch.nn.Module):
    def __init__(self, input_dim, base_filter, output_dim, tag_dims, down_layers=4, norm='batch'):
        super(ChainerDRAGANDiscriminator2, self).__init__()
        self.base_filter = base_filter

        # Convolution layers
        self.input_conv = ConvBlock(input_dim, base_filter, activation='lrelu', norm=None)

        conv_blocks = []
        num_filter = base_filter
        for i in range(down_layers - 1):
            conv_blocks.append(ConvBlock(num_filter, num_filter * 2, activation='lrelu', norm=norm))
            num_filter = num_filter * 2
        self.downsample = torch.nn.Sequential(*conv_blocks)

        # Fully-connected layer for output
        self.output_fc = DenseBlock(num_filter * 4 * 4, output_dim, activation='sigmoid', norm=None)

        # Fully-connected layer for labels
        self.label_fc1 = DenseBlock(num_filter * 4 * 4, tag_dims[0], activation=None, norm=None)
        # self.label_fc2 = DenseBlock(num_filter * 4 * 4, tag_dims[1], activation=None, norm=None)
        # self.label_fc3 = DenseBlock(num_filter * 4 * 4, tag_dims[2], activation=None, norm=None)

    def forward(self, input):
        x = self.input_conv(input)
        x = self.downsample(x)
        x = x.view(x.size()[0], -1)
        out = self.output_fc(x)
        label1 = self.label_fc1(x)
        # label2 = self.label_fc2(x)
        # label3 = self.label_fc3(x)
        return out, label1

    def weight_init(self):
        for m in self.modules():
            utils.weights_init(m)