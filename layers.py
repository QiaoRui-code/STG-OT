import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from lib.basic import squeeze, unsqueeze
from lib.basic import Swish, Lambda, divergence_approx, divergence_bf,sample_rademacher_like,sample_gaussian_like
from lib.regularization import RegularizedODEfunc
from torch_geometric.nn import GCNConv
__all__ = ["ODEnet","ODEfunc"]


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, graph, use_bias=True):
        super(GraphConvolution, self).__init__()

        self._layer = nn.Linear(input_dim, output_dim)
        self.conv1 = GCNConv(input_dim, output_dim, graph, bias=use_bias)
        # self.conv2 = GCNConv(10, output_dim, graph, bias=use_bias)
        self.graph = graph
        self.bias = nn.Linear(1, output_dim, bias=False)
        self.gate = nn.Linear(1, output_dim)

    def forward(self, t, input_feature):

        self.adjacency = torch.tensor(self.graph).clone().detach().requires_grad_(True).to(input_feature)
        self.adjacency = self.adjacency-torch.diag(self.adjacency)+torch.eye(self.adjacency.shape[0]).to(input_feature)
        support = torch.mm(input_feature, self.adjacency)
        output = self._layer(support)
        return output * torch.sigmoid(self.gate(t.view(1, 1))) \
            + self.bias(t.view(1, 1))

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}

class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(
        self, hidden_dims, input_shape, graph, strides, conv, layer_type="GraphConvolution", nonlinearity="softplus", num_squeeze=0
    ):
        super(ODEnet, self).__init__()
        self.num_squeeze = num_squeeze


        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape
        strides = [None] * (len(hidden_dims) + 1)
        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            layer_kwargs = {}
            base_layer = {
                "GraphConvolution":GraphConvolution,
            }[layer_type]
            layer = base_layer(hidden_shape[0], dim_out, graph, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y):
        dx = y
        # squeeze
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # unsqueeze
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)
        return dx



class ODEfunc(nn.Module):

    def __init__(self, diffeq, divergence_fn="approximate", residual=False, rademacher=False):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        # t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
            else:
                self._e = sample_gaussian_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32
                                                                     ).to(divergence)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, z, logpz=None, integration_times=None, reverse=False):

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))

        if self.training:
            state_t = odeint(
                self.odefunc,
                (z, _logpz) + reg_states,
                integration_times.to(z),
                atol=[self.atol, self.atol] + [1e20] * len(reg_states) if self.solver == 'dopri5' else self.atol,
                rtol=[self.rtol, self.rtol] + [1e20] * len(reg_states) if self.solver == 'dopri5' else self.rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        self.regularization_states = state_t[2:]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()

def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, integration_times=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, integration_times=integration_times, reverse=reverse)
            return x, logpx
