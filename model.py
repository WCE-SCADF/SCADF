from torch import nn
import torch
import torch.nn.functional as F
from itertools import repeat
import collections.abc
from einops import rearrange

# Convolutional Layers
def conv_layer(in_channels, out_channels, kernel_size, stride=1):
    if not isinstance(kernel_size, collections.abc.Iterable):
        kernel_size = tuple(repeat(kernel_size, 2))
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
    
# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
# Activation types
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        act_func = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        act_func = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        act_func = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'silu':
        pass
    elif act_type == 'gelu':
        pass
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return act_func


# S2FB
class S2FB_2(nn.Module):
    def __init__(self, n_feat, reduction, bias):
        super(S2FB_2, self).__init__()
        self.DSC = depthwise_separable_conv(n_feat * 2, n_feat)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)



    def forward(self, x1, x2):
        FEA_1 = self.DSC(torch.cat((x1, x2), 1))
        res = self.CA_fea(FEA_1) + x1
        # res = FEA_1 + x1
        return res
    
# Enhanced Channel Attention Block with Depthwise Separable Convolution
class CAB_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB_dsc, self).__init__()
        modules_body = [depthwise_separable_conv(n_feat, n_feat),
                        act,
                        depthwise_separable_conv(n_feat, n_feat)]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        return res
    
# Self attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

class SCADFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, deploy=False):
        super(SCADFBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation('lrelu')

        self.conv = nn.Conv2d(in_channels*3, in_channels, kernel_size=1, stride= 1)
       
        self.rbr_3x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                        stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
        self.rbr_3x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                        stride=1, padding=(1, 0), dilation=1, groups=1, padding_mode='zeros')
        self.rbr_1x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                        stride=1, padding=(0, 1), dilation=1, groups=1, padding_mode='zeros')
        self.CAB = CAB_dsc(n_feat = in_channels, kernel_size=3, reduction=4, bias = False, act = activation('lrelu'))
        self.TB = SelfAttention(dim=in_channels, num_heads=2)
        #self.channel_attention_1 = CALayer(in_channels, reduction=4)
        #self.channel_attention_2 = CALayer(in_channels, reduction=4)

    def forward(self, inputs):
        return self.activation(
                inputs 
                + self.rbr_3x1_branch(inputs)
                 + self.rbr_1x3_branch(inputs) 
                    + self.CAB(inputs)
                     + self.rbr_3x3_branch(self.TB(inputs)))

class Upsample_Block(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
        super(Upsample_Block, self).__init__()
        self.conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, input):
        c = self.conv(input)
        upsample = self.pixel_shuffle(c)
        return upsample
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, s_factor):
        super(UpSample, self).__init__()
        self.up_B = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.up_R = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x_R):
        return self.up_R(x_R)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, s_factor):
        super(DownSample, self).__init__()
        self.down_B = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        self.down_R = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                    nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x_R):
        return self.down_R(x_R)

# thanks for RLFN: https://github.com/bytedance/RLFN
class ESA(nn.Module):
    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        #         self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        #         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        #         v_range = self.relu(self.conv_max(v_max))
        #         c3 = self.relu(self.conv3(v_range))
        #         c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m
class MSSCADF(nn.Module):
    def __init__(self, in_channels,act_type):
        super(MSSCADF, self).__init__()
        self.down_1 = DownSample(in_channels=in_channels, out_channels=in_channels*2, s_factor=0.5)
        self.repblock1 = SCADFBlock(in_channels=in_channels*2, out_channels=in_channels*2, act_type=act_type)
       

        self.down_2 = DownSample(in_channels=in_channels, out_channels=in_channels*4, s_factor=0.25)
        self.repblock2 = SCADFBlock(in_channels=in_channels*4, out_channels=in_channels*4, act_type=act_type)
        

        self.repblock3 = SCADFBlock(in_channels=in_channels, out_channels=in_channels, act_type=act_type)
        
        self.up_1 = UpSample(in_channels=in_channels*2, out_channels=in_channels,s_factor=2)
        self.up_2 = UpSample(in_channels=in_channels*4, out_channels=in_channels, s_factor=4)
        self.esa = ESA(16, in_channels, nn.Conv2d)
        self.conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.act = activation('lrelu')

    def forward(self, inputs):
        x_2_down = self.down_1(inputs)
        x_2 = self.repblock1(x_2_down)
        x_2_up = self.up_1(x_2)

        x_4_down = self.down_2(inputs)
        x_4 = self.repblock2(x_4_down)
        x_4_up = self.up_2(x_4)

        x = self.repblock3(inputs)
        out = torch.cat([x_2_up, x, x_4_up], 1) # concatenation along channel axis
        out = self.act(self.conv(out))
        out = out + inputs
        out = self.esa(out)
        return out
    
class EC_MSSCADF(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_nums=48, upscale_factor=1):
        super(EC_MSSCADF, self).__init__()
        self.fea_conv = conv_layer(in_channels=in_channels, out_channels=feature_nums, kernel_size=3)

        self.reprfb1 = MSSCADF(in_channels=feature_nums, act_type='lrelu')
        self.down_1 = DownSample(feature_nums, feature_nums*2, s_factor=0.5)
        self.reprfb2 = MSSCADF(in_channels=feature_nums*2, act_type='lrelu')
        self.down_2 = DownSample(feature_nums*2, feature_nums*4, s_factor=0.5)
        self.reprfb3 = MSSCADF(in_channels=feature_nums*4, act_type='lrelu')
        self.up_2 = UpSample(feature_nums*4, feature_nums*2, s_factor=2)
        self.reprfb4 = MSSCADF(in_channels=feature_nums*2, act_type='lrelu')
        self.up_1 = UpSample(feature_nums*2, feature_nums, s_factor=2)
        self.reprfb5 = MSSCADF(in_channels=feature_nums, act_type='lrelu')
        self.lr_conv = conv_layer(in_channels=feature_nums, out_channels=feature_nums, kernel_size=3)

        self.upsampler = Upsample_Block(in_channels=feature_nums, out_channels=out_channels,
                                        upscale_factor=upscale_factor)

    def forward(self, inputs):
        outputs_feature = self.fea_conv(inputs)
        
        outputs_1 = self.reprfb1(outputs_feature)
        outputs_2 = self.down_1(outputs_1)

        outputs_3 = self.reprfb2(outputs_2)
        outputs_4 = self.down_2(outputs_3)

        outputs_5 = self.reprfb3(outputs_4)
        outputs_6 = outputs_5 + outputs_4 # residual 1

        outputs_7 = self.up_2(outputs_6)
        outputs_8 = self.reprfb4(outputs_7)
        outputs_9 = outputs_8 + outputs_2
        outputs_10 = self.up_1(outputs_9)
        outputs_11 = self.reprfb5(outputs_10)

        outputs = self.lr_conv(outputs_11) + outputs_feature

        
        outputs = self.upsampler(outputs)
        return outputs