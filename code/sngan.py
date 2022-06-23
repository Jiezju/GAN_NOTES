import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules import conv
from torch.nn.modules import Linear
from torch.nn.modules.utils import _pair


# define _l2normalization
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)


def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    w_shape = W.size
    w = W.data.reshape(-1, w_shape[-1])

    if u is None:
        u = torch.FloatTensor(1, W.size(-1)).normal_(0, 1).cuda()
    u_hat = u
    v_hat = None
    # Ip = 1 is usually enough
    for _ in range(Ip):
        v_hat = _l2normalize(torch.matmul(u_hat, torch.transpose(W.data, 0, 1)), eps=1e-12)
        u_hat = _l2normalize(torch.matmul(v_hat, W.data), eps=1e-12)
    sigma = torch.matmul(torch.matmul(v_hat, w), torch.transpose(u_hat))
    return sigma, u_hat


class SNConv2d(conv._ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.
    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:
    .. math::
        \begin{array}{ll}
        out(N_i, C_{out_j})  = bias(C_{out_j})
                       + \sum_{{k}=0}^{C_{in}-1} weight(C_{out_j}, k)  \star input(N_i, k)
        \end{array}
    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    | :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.
    | :attr:`padding` controls the amount of implicit zero-paddings on both
    |  sides for :attr:`padding` number of points for each dimension.
    | :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    | :attr:`groups` controls the connections between inputs and outputs.
      `in_channels` and `out_channels` must both be divisible by `groups`.
    |       At groups=1, all inputs are convolved to all outputs.
    |       At groups=2, the operation becomes equivalent to having two conv
                 layers side by side, each seeing half the input channels,
                 and producing half the output channels, and both subsequently
                 concatenated.
            At groups=`in_channels`, each input channel is convolved with its
                 own set of filters (of size `out_channels // in_channels`).
    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:
        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension
    .. note::
         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.
    .. note::
         The configuration when `groups == in_channels` and `out_channels = K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.
         In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(in\_channels=C_{in}, out\_channels=C_{in} * K, ..., groups=C_{in})`
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
        W(Tensor): Spectrally normalized weight
        u (Tensor): the right largest singular value of W.
    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SNLinear(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
       Args:
           in_features: size of each input sample
           out_features: size of each output sample
           bias: If set to False, the layer will not learn an additive bias.
               Default: ``True``
       Shape:
           - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
             additional dimensions
           - Output: :math:`(N, *, out\_features)` where all but the last dimension
             are the same shape as the input.
       Attributes:
           weight: the learnable weights of the module of shape
               `(out_features x in_features)`
           bias:   the learnable bias of the module of shape `(out_features)`
           W(Tensor): Spectrally normalized weight
           u (Tensor): the right largest singular value of W.
       """

    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.linear(input, self.W_, self.bias)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, use_BN=False, downsample=False):
        super(ResBlock, self).__init__()
        # self.conv1 = SNConv2d(n_dim, n_out, kernel_size=3, stride=2)
        hidden_channels = in_channels
        self.downsample = downsample

        self.resblock = self.make_res_block(in_channels, out_channels, hidden_channels, use_BN, downsample)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)

    def make_res_block(self, in_channels, out_channels, hidden_channels, use_BN, downsample):
        model = []
        if use_BN:
            model += [nn.BatchNorm2d(in_channels)]

        model += [nn.ReLU()]
        model += [SNConv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(hidden_channels, out_channels, kernel_size=3, padding=1)]
        if downsample:
            model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)

    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        if self.downsample:
            model += [nn.AvgPool2d(2)]
            return nn.Sequential(*model)
        else:
            return nn.Sequential(*model)

    def forward(self, input):
        return self.resblock(input) + self.residual_connect(input)
