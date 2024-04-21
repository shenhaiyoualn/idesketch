
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  default_init_weights, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
import  torch

@BACKBONES.register_module()
class MLANet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in "Photo-Realistic Single
    Image Super-Resolution Using a Generative Adversarial Network".

    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_blocks (int): Block number in the trunk network. Default: 16.
        upscale_factor (int): Upsampling factor. Support x2, x3 and x4.
            Default: 4.
    """
    _supported_upscale_factors = [2, 3, 4]

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_blocks=16,
                 upscale_factor=2):

        super(MLANet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.upscale_factor = upscale_factor

        # hyper-params
        self.scale = 2
        n_FAMs = 8
        n_feats = 64
        kernel_size = 3
        act = nn.LeakyReLU(True)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])

        # define head module
        head = []
        head.append(
            wn(nn.Conv2d(3, n_feats, 3, padding=3//2)))

        # define body module
        body = []
        for i in range(n_FAMs):
            body.append(
                FAM(n_feats, wn=wn, act=act))

        # define tail module
        tail = Tail(2, n_feats, kernel_size, wn)

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = tail


    def forward(self, x):
        input = x
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x + torch.nn.functional.upsample(input, scale_factor=2, mode='bicubic')

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 or  name.find('skip') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            # Initialization methods like `kaiming_init` are for VGG-style
            # modules. For modules with residual paths, using smaller std is
            # better for stability and performance. There is a global residual
            # path in MSRResNet and empirically we use 0.1. See more details in
            # "ESRGAN: Enhanced Super-Resolution Generative Adversarial
            # Networks"
            for m in [self.head, self.body, self.tail]:
                default_init_weights(m, 0.1)
        else:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')



def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class Tail(nn.Module):
    def __init__(
        self,  scale, n_feats, kernel_size, wn):
        super(Tail, self).__init__()
        out_feats = scale*scale*3
        self.tail_k3 = wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2, dilation=1))
        self.tail_k5 = wn(nn.Conv2d(n_feats, out_feats, 5, padding=5//2, dilation=1))
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.scale_k3 = Scale(0.5)
        self.scale_k5 = Scale(0.5)


    def forward(self, x):
        x0 = self.pixelshuffle(self.scale_k3(self.tail_k3(x)))
        x1 = self.pixelshuffle(self.scale_k5(self.tail_k5(x)))

        return x0+x1

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class FAM(nn.Module):
    def __init__(
        self, n_feats, wn, act=nn.ReLU(True)):
        super(FAM, self).__init__()

        self.b0 = MAB(n_feats=n_feats,reduction_factor=4)
        self.b1 = MAB(n_feats=n_feats,reduction_factor=4)
        self.b2 = MAB(n_feats=n_feats,reduction_factor=4)
        self.b3 = MAB(n_feats=n_feats,reduction_factor=4)

        self.reduction1 = wn(nn.Conv2d(n_feats*2, n_feats, 1))
        self.reduction2 = wn(nn.Conv2d(n_feats*2, n_feats, 1))
        self.reduction3 = wn(nn.Conv2d(n_feats*2, n_feats, 1))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)+x0
        x2 = self.b2(x1)+x1
        x3 = self.b3(x2)

        res1 = self.reduction1(channel_shuffle(torch.cat([x0, x1],dim=1), 2))
        res2 = self.reduction2(channel_shuffle(torch.cat([res1, x2], dim=1), 2))
        res = self.reduction3(channel_shuffle(torch.cat([res2,x3], dim=1), 2))

        return self.res_scale(res) + self.x_scale(x)

class MAB(nn.Module):
    def __init__(self, n_feats, reduction_factor=4, distillation_rate=0.25):
        super(MAB, self).__init__()
        self.reduce_channels = nn.Conv2d(n_feats, n_feats//reduction_factor,1)
        self.reduce_spatial_size = nn.Conv2d(n_feats//reduction_factor, n_feats//reduction_factor, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(7, stride=3)
        self.increase_channels = conv_block(n_feats//reduction_factor, n_feats, 1)

        self.conv1 = conv_block(n_feats//reduction_factor, n_feats//reduction_factor,3,dilation=1,act_type='lrelu')
        self.conv2 = conv_block(n_feats // reduction_factor, n_feats//reduction_factor, 3,dilation=2,act_type='lrelu')

        self.sigmoid = nn.Sigmoid()

        self.conv00 = conv_block(n_feats, n_feats,3, act_type=None)
        self.conv01 = conv_block(n_feats, n_feats,3,act_type='lrelu')

        self.bottom11 = conv_block(n_feats,n_feats,1,act_type=None)
        self.bottom11_dw = conv_block(n_feats, n_feats,5, groups=n_feats,act_type=None)

    def forward(self, x):
        x  = self.conv00(self.conv01(x))
        rc = self.reduce_channels(x)
        rs = self.reduce_spatial_size(rc)
        pool = self.pool(rs)
        conv = self.conv2(pool)
        conv = conv + self.conv1(pool)
        up =  torch.nn.functional.upsample(conv, size=(rc.shape[2],rc.shape[3]), mode='nearest')
        up =  up + rc
        out = (self.sigmoid(self.increase_channels(up)) * x) *  self.sigmoid(self.bottom11_dw(self.bottom11(x)))
        return out
