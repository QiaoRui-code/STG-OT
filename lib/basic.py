
import torch.nn as nn
import torch
__all__ = ['SqueezeLayer']


class SqueezeLayer(nn.Module):
    def __init__(self, downscale_factor):
        super(SqueezeLayer, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._upsample(x, logpx)
        else:
            return self._downsample(x, logpx)

    def _downsample(self, x, logpx=None):
        squeeze_x = squeeze(x, self.downscale_factor)
        if logpx is None:
            return squeeze_x
        else:
            return squeeze_x, logpx

    def _upsample(self, y, logpy=None):
        unsqueeze_y = unsqueeze(y, self.downscale_factor)
        if logpy is None:
            return unsqueeze_y
        else:
            return unsqueeze_y, logpy


def unsqueeze(input, upscale_factor=2):
    '''
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor**2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width)

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)

def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
#     print(dx.shape, dx.requires_grad, y.shape, y.requires_grad)
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1

def sample_gaussian_like(y):
    return torch.randn_like(y)

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):

    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)
