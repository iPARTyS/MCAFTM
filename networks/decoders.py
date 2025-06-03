import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from EFTM import EFTMBlock, Attention


class MlpConv(nn.Module):
    def __init__(self, in_features, scale=4, act_layer=nn.GELU):
        super().__init__()

        self.fc1 = nn.Conv2d(in_features, in_features // scale, 1, 1, 0, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(in_features // scale, in_features, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.point_wise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.point_wise_conv(x)

        return x


class GSACM(nn.Module):
    def __init__(self, dim, group=4, act_layer=nn.Hardswish):

        super().__init__()
        self.dim = dim
        self.group = group

        seg_dim = self.dim // self.group

        self.agg0 = SeparableConv2d(seg_dim, seg_dim, 1, 1, 0)
        self.norm0 = nn.BatchNorm2d(seg_dim)
        self.act0 = act_layer()

        self.agg1 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(seg_dim)
        self.act1 = act_layer()

        self.agg2 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 2)
        self.norm2 = nn.BatchNorm2d(seg_dim)
        self.act2 = act_layer()

        self.agg3 = SeparableConv2d(seg_dim, seg_dim, 7, 1, 3)
        self.norm3 = nn.BatchNorm2d(seg_dim)
        self.act3 = act_layer()

    def forward(self, x):
        B, C, H, W = x.shape
        assert C % self.group == 0, f"dim {C} should be divided by group {self.group}."

        seg_dim = self.dim // self.group
        x = x.chunk(self.group, dim=1)

        x0 = self.act0(self.norm0(self.agg0(x[0])))
        x1 = self.act1(self.norm1(self.agg1(x[1])))
        x2 = self.act2(self.norm2(self.agg2(x[2])))
        x3 = self.act3(self.norm3(self.agg3(x[3])))

        x = torch.cat([x0, x1, x2, x3], dim=1)

        return x


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


#   Multi-scale depth-wise convolution (MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel
        self.s_a = SAB()
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])
        self.c_a = CAB(in_channels)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        sa_x = self.s_a(x) * x
        outputs.append(sa_x)
        ca_x = self.c_a(x) * x
        outputs.append(ca_x)
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them

        return outputs


class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB)
    """

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


#   Multi-scale convolution block (MSCB)
def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
              add=True, activation='relu6'):
    """
    create a series of multi-scale convolution blocks.
    """
    convs = []
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(mscb)
    if n > 1:
        for i in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(mscb)
    conv = nn.Sequential(*convs)
    return conv


#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


#   Large-kernel grouped attention gate (LGAG)
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)

    #   Spatial attention block (SAB)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class MSGA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, activation='relu'):
        super(MSGA, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1)
        )

        self.conv1 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        # self.convx_2 = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1, bias=False)
        self.convx = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False)
        self.up_dwc2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            act_layer(activation, inplace=True)
        )

        self.gelu = nn.Sequential(
            act_layer('gelu', inplace=True)
        )
        self.init_weights('normal')
        self.mlp = MlpConv(out_channels, 2)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x2 = channel_shuffle(x2, self.in_channels)

        x2 = self.up_dwc2(x2)

        x = x1 + x2

        avg_out1 = torch.mean(x, dim=1, keepdim=True)
        max_out1, _ = torch.max(x, dim=1, keepdim=True)

        x_1 = torch.cat([avg_out1, max_out1], dim=1)

        x_w = self.sigmoid(self.conv(x_1))

        x1 = x * x_w
        x = x1 + x

        x = self.mlp(x)

        return x



class MCAFTM(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu', image_size=352, num_heads=1):
        # // channels=[512, 320, 128, 64]
        super(MCAFTM, self).__init__()
        eucb_ks = 3  # kernel size for eucb
        self.head = num_heads
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])

        self.GSACM1 = GSACM(channels[0])
        self.GSACM2 = GSACM(channels[1])
        self.GSACM3 = GSACM(channels[2])
        self.GSACM4 = GSACM(channels[3])

        self.cross4 = MSGA(in_channels=channels[0], out_channels=channels[1])
        self.cross3 = MSGA(in_channels=channels[1], out_channels=channels[2])
        self.cross2 = MSGA(in_channels=channels[2], out_channels=channels[3])
        self.cross1 = MSGA(in_channels=channels[2], out_channels=channels[3])

        self.wave4 = Attention(512,16,False)
        self.wave3 = EFTMBlock(320,10,1)
        self.wave2 = EFTMBlock(128,4,1)
        self.wave1 = EFTMBlock(64, 2,1)
        # self.Dw = EFTMBlock(64,num_heads=8,sr_ratio=1)
        self.con4_1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[0])
        )

        self.con4 = nn.Sequential(
            nn.Conv2d(channels[0], channels[2], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[2])
        )

        self.con3_1 = nn.Sequential(
            nn.Conv2d(channels[1], channels[1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[1])
        )
        self.con3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[2])
        )

        self.con2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[2], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[2])
        )

        self.con1 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[3])
        )
        self.con = nn.Sequential(
            nn.Conv2d(channels[3], channels[3], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[3])
        )
        self.sab = SAB()

    def forward(self, x4, x3, x2, x1, H, W):

        d4 = self.GSACM1(x4)
        x4_v = self.wave4(x4, H // 32, W // 32)
        d4 = x4_v * d4
        # MSCAM4
        d4 = self.cab4(d4) * d4
        d4 = self.sab(d4) * d4
        d4_m = self.mscb4(d4)
        d4 = self.con4_1(d4_m)

        # Additive aggregation 3
        d3 = self.cross4(x3, d4)
        d3 = self.GSACM2(d3)
        x3_v  = self.wave3(x3, H // 16, W // 16)
        d3 = x3_v * d3
        # MSCAM3
        d3 = self.cab3(d3) * d3
        d3 = self.sab(d3) * d3
        d3_m = self.mscb3(d3)
        d3 = self.con3_1(d3_m)

        d2 = self.cross3(x2, d3)
        d2 = self.GSACM3(d2)
        x2_v = self.wave2(x2, H // 8, W // 8)
        d2 = x2_v * d2
        # MSCAM2
        d2 = self.cab2(d2) * d2
        d2 = self.sab(d2) * d2
        d2_m = self.mscb2(d2)
        d2 = self.con2(d2_m)

        d1 = self.cross2(x1, d2)
        d1 = self.GSACM4(d1)
        d1_v = self.wave1(x1, H // 4, W // 4)
        d1 = d1_v * d1
        # MSCAM1
        d1 = self.cab1(d1) * d1
        d1 = self.sab(d1) * d1

        d1_m = self.mscb1(d1)

        d1 = self.con(d1_m)

        return  [d4, d3, d2, d1]


if __name__ == '__main__':
    model = MCAFTM(image_size=224, num_heads=9).cuda()
    x4 = torch.randn(4, 512, 7, 7).cuda()
    x3 = torch.randn(4, 320, 14, 14).cuda()
    x2 = torch.randn(4, 128, 28, 28).cuda()
    x1 = torch.randn(4, 64, 56, 56).cuda()
    # predict1, predict2, predict3, predict4 = model(x)
    # print(predict1.shape)  #  deep_supervision true   predict[0] [2, 1, 256, 256] , predict[1] [2, 1, 128, 128] 这两项用于监督
    # print(predict2.shape)
    # print(predict3.shape)
    # print(predict4.shape)
    # predict1, predict2, predict3, predict4 = model(x4, 7,7)
    predict1, predict2, predict3, predict4 = model(x4, x3, x2, x1, 224, 224)
    print(predict1.shape)
    print(predict2.shape)
    print(predict3.shape)
    print(predict4.shape)
