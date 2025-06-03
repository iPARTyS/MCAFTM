import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import pywt


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bridge=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.bridge = False
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, q=None):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        B, N, C = x.shape

        if self.bridge:
            q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x.permute(0, 2, 1).reshape(B, C,H,W)
        return x

class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)

            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)

            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class SE_1D(nn.Module):
    def __init__(self, in_channels, se_channels):
        super().__init__()

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, se_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(se_channels, in_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        y = self.fc(x)
        return x * y


class QueryGenerator(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.se = SE_1D(dim, dim_out)

    def forward(self, x):
        B, C, _ = x.shape
        x = self.se(x)

        q_lvl1 = x[..., :392].reshape(B, -1, C * 8)
        q_lvl2 = x[..., 392:1372].reshape(B, -1, C * 5)
        q_lvl3 = x[..., 1372:2940].reshape(B, -1, C * 2)


        return [q_lvl1, q_lvl2, q_lvl3]



class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W)+self.fc1(x)))
        out = self.fc2(ax)
        return out

class SE_1D(nn.Module):
    def __init__(self, in_channels, se_channels):
        super().__init__()

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, se_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(se_channels, in_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        y = self.fc(x)
        return x * y



class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.reshape(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class FET(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, bridge=False, L_num=3):
        super().__init__()
        assert L_num > 0, "Laplacian number (L_num) must be in [1, 2, 3]"
        self.L_num = L_num  # Laplacian number (L0, L1, ...)
        self.bridge = bridge
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.dwt = DWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.ra_dwt = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )

        self.hf_agg = nn.Conv3d(dim // 4, dim // 4, kernel_size=(3, 1, 1), bias=False, groups=dim // 4)
        self.filter2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.filter3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        # Gaussian Kernel
        ### parameters
        kernet_shapes = [3, 5]
        s_value = np.power(2, 1 / 3)
        sigma = 1.6

        ### Kernel weights for Laplacian pyramid

        self.sig = nn.Sigmoid()
        self.linear_upsample = nn.Linear(dim // 4, dim)
        self.proj = nn.Linear(dim + dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, q=None):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_d = self.reduce(x)
        x_dwt = self.dwt(x_d)
        x_dwt_filter = self.filter(x_dwt)

        kv = self.kv_embed(x_dwt_filter).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Spatial Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)


        # Efficient Attention
        global_context = F.softmax(k.reshape(B, N//4, C).transpose(1, 2), dim=2) @ v.reshape(B, N//4, C)
        out_efficient_att = F.softmax(q.reshape(B, N, C), dim=1) @ global_context

        # R Attention
        x_dwt_hf = self.filter3(x_dwt)*(1-self.sig(self.filter2(x_dwt)))
        x_dwt_hf = x_dwt_hf.reshape(B,self.num_heads,-1,C // self.num_heads)

        v = v + x_dwt_hf

        # Spatial Attention @ Enhanced Value
        out_spatial_boundary = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Final Projection
        out = self.proj(torch.cat([out_spatial_boundary, out_efficient_att], dim=-1))

        return out


class FETBlock(nn.Module):

    def __init__(self, in_dim, num_heads, sr_ratio, L_num=3):
        super().__init__()
        self.in_dim = in_dim
        self.norm1 = nn.LayerNorm(in_dim)
        self.concat_linear = nn.Linear(in_dim, in_dim)
        self.attn = FET(in_dim, num_heads, sr_ratio, L_num=L_num)
        self.mlp = MixFFN_skip(in_dim, int(in_dim * 4))
        self.norm2 = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor, H, W, q=None) -> torch.Tensor:
        B, C, H,W= x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        x = self.concat_linear(x)

        x = x + self.attn(self.norm1(x), H, W, q)
        x = x + self.mlp(self.norm2(x), H, W)
        x = x.permute(0,2,1).reshape(B, C, H, W)
        return x



if __name__ == '__main__':

