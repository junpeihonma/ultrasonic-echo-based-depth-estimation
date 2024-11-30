import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])

def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class SimpleAudioNet(nn.Module):
    ## strucure adapted from VisualEchoes [ECCV 2020]
    r"""A Simple 3-Conv CNN followed by a fully connected layer
    """
    def __init__(self, conv1x1_dim, audio_shape, audio_feature_length, output_nc=1): 
        super(SimpleAudioNet, self).__init__()
        self._n_input_audio = audio_shape[0]
        
        # kernel size
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        # strides
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
        
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        """ Encoder """
        self.conv1 = create_conv(self._n_input_audio, 32, kernel=self._cnn_layers_kernel_size[0], paddings=0, stride=self._cnn_layers_stride[0])
        self.conv2 = create_conv(32, 64, kernel=self._cnn_layers_kernel_size[1], paddings=0, stride=self._cnn_layers_stride[1])
        self.conv3 = create_conv(64, conv1x1_dim, kernel=self._cnn_layers_kernel_size[2], paddings=0, stride=self._cnn_layers_stride[2])

        layers = [self.conv1, self.conv2, self.conv3]
        self.feature_extraction = nn.Sequential(*layers)
        self.conv1x1 = create_conv(conv1x1_dim * cnn_dims[0] * cnn_dims[1], audio_feature_length, 1, 0)

        """ Decoder """
        self.rgbdepth_upconvlayer1 = unet_upconv(512, 512) 
        self.rgbdepth_upconvlayer2 = unet_upconv(512, 256)
        self.rgbdepth_upconvlayer3 = unet_upconv(256, 128)
        self.rgbdepth_upconvlayer4 = unet_upconv(128, 64)
        self.rgbdepth_upconvlayer5 = unet_upconv(64, 32)
        self.rgbdepth_upconvlayer6 = unet_upconv(32, 16)
        self.rgbdepth_upconvlayer7 = unet_upconv(16, 1, True)
        self.rgbdepth_upconvlayer7_2 = unet_upconv(16, 4, True)

    def _conv_output_dim (self, dimension, padding, dilation, kernel_size, stride):
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x, depth_or_spec="depth"):

        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.conv1x1(x)

        if depth_or_spec == "depth":
            rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(x)     
            rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(rgbdepth_upconv1feature)
            rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(rgbdepth_upconv2feature)
            rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(rgbdepth_upconv3feature)
            rgbdepth_upconv5feature = self.rgbdepth_upconvlayer5(rgbdepth_upconv4feature)
            rgbdepth_upconv6feature = self.rgbdepth_upconvlayer6(rgbdepth_upconv5feature)
            pre_depth = self.rgbdepth_upconvlayer7(rgbdepth_upconv6feature) 
            return pre_depth

        elif depth_or_spec == "spec":
            rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(x)
            rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(rgbdepth_upconv1feature)
            rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(rgbdepth_upconv2feature)
            rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(rgbdepth_upconv3feature)
            rgbdepth_upconv5feature = self.rgbdepth_upconvlayer5(rgbdepth_upconv4feature)
            rgbdepth_upconv6feature = self.rgbdepth_upconvlayer6(rgbdepth_upconv5feature)
            rgbdepth_upconv7feature = self.rgbdepth_upconvlayer7_2(rgbdepth_upconv6feature)
            pre_spec = torch.nn.functional.interpolate(rgbdepth_upconv7feature, size=(257,4381),mode='bilinear') 
            return pre_spec