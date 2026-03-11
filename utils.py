import torch
from torch import Tensor
import warnings
import random
import numpy as np
import os
import scanpy as sc
from sklearn.metrics import roc_curve, roc_auc_score
import os
import logging
import torch
from layers import CNF
import math
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    #os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger



def _get_data_points(adata, embedding_name):
    if embedding_name is None:
        # 直接使用原始数据（adata.X）
        data_points = adata.X
    else:
        # 尝试从 obsm 中获取指定嵌入（如 "pca"）
        obsm_key = f"X_{embedding_name}"
        if obsm_key not in adata.obsm:
            raise KeyError(
                f"Embedding '{embedding_name}' not found in `adata.obsm`. "
                f"Available embeddings: {list(adata.obsm.keys())}"
            )
        data_points = adata.obsm[obsm_key]

    # 提取速度信息（如果有）
    velocity_points = None
    if "velocity" in adata.layers:
        velocity_points = adata.layers["velocity"]
    elif f"velocity_{embedding_name}" in adata.obsm:
        velocity_points = adata.obsm[f"velocity_{embedding_name}"]

    return data_points, velocity_points
def load_data(args,path):
    adata = sc.read_h5ad(path)
    labels = np.array(adata.obs["time"])
    data, velocity = _get_data_points(adata,args.embedding_name)
    if issparse(data):
        data = data.toarray()
    if velocity is not None and issparse(velocity):
        velocity = velocity.toarray()
   
    #scaler = StandardScaler()
    #scaler.fit(data)
    #data = scaler.transform(data)
    #     if self.velocity is not None:
    #         self.velocity = self.velocity / scaler.scale_
    # self.use_velocity = self.velocity is not None

    ncells = data.shape[0]
    assert labels.shape[0] == ncells
    #max_dim = args.max_dim
    #if max_dim is not None and data.shape[1] > max_dim:
        #print(f"Warning: Clipping dimensionality from {data.shape[1]} to {max_dim}")
        #data = data[:, :max_dim]
        #if args.use_velocity:
            #velocity = velocity[:, :max_dim]
    return adata, data, labels, velocity

def sample_index(n, data, label, label_subset, w=None):
    arr = np.arange(data.shape[0])[label == label_subset]
    p_sub = np.random.choice(arr, size=n)
    if w is None:
        w_ = torch.ones(len(p_sub))
    else:
        w_ = w[p_sub].clone()
    w_ = w_ / w_.sum()
    return p_sub, w_


def base_density():
    def standard_normal_logprob(z):
        logZ = -0.5 * math.log(2 * math.pi)
        return torch.sum(logZ - z.pow(2) / 2, 1, keepdim=True)
    return standard_normal_logprob

def base_sample():
    return torch.randn


def known_base_density():
    return False


def calc_and_log_metrics(causal_pred, true_cm, threshold=0.5, plot_roc=False):
    causal_graph =np.array (causal_pred > threshold).astype(int)
    tp = np.mean(causal_graph * true_cm)
    tn = np.mean((1-causal_graph) * (1-true_cm))
    fp = np.mean(causal_graph * (1-true_cm))
    fn = np.mean((1-causal_graph) * true_cm)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(true_cm.reshape(-1)>0.5,causal_pred.detach().numpy().reshape(-1))
    if plot_roc:
        fpr, tpr, thres = roc_curve(true_cm.reshape(-1) > 0.5,causal_pred.detach().numpy().reshape(-1), pos_label=1)
        fig = plt.figure(figsize=[4, 4])
        plt.plot(fpr, tpr)
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig('roc_curve.png', dpi=300)
    return auc
def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)

def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def sigmoid_gumbel_sample(graph, tau=1):
    prob = torch.sigmoid(graph[:, :, None])
    logits = torch.concat([prob, (1-prob)], axis=-1)
    samples = gumbel_softmax(logits, tau=tau)[:, :, 0]
    return samples


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_nfe(model):
    class AccNumEvals(object):
        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, CNF):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals

def count_total_time(model):
    class Accumulator(object):
        def __init__(self):
            self.total_time = 0

        def __call__(self, module):
            if isinstance(module, CNF):
                self.total_time = (
                    self.total_time + module.sqrt_end_time * module.sqrt_end_time
                )

    accumulator = Accumulator()
    model.apply(accumulator)
    return accumulator.total_time

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return torch.sum(logZ - torch.tensor(z).pow(2) / 2, 1, keepdim=True)

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
