import numpy as np
import torch
import scipy
import warnings
import time
import torch.distributed as dist
import scipy
from pymfe.mfe import MFE

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


def agg_all_metrics(outputs):
    if len(outputs) == 0:
        return outputs
    res = {}
    keys = [k for k in outputs[0].keys() if not isinstance(outputs[0][k], dict)]
    for k in keys:
        all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs])
        if k != 'epoch':
            res[k] = np.mean(all_logs)
        else:
            res[k] = all_logs[-1]
    return res


def gather_metrics(metrics):
    for k, v in metrics.items():
        if v.dim() == 0:
            v = v[None]
        v_all = [torch.zeros_like(v) for _ in range(dist.get_world_size())]
        dist.all_gather(v_all, v)
        v_all = torch.cat(v_all)
        metrics[k] = v_all

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def align_loss_similarity(x, y, tau):
    #similarity = torch.matmul(x, y.T) / tau
    #return - torch.diagonal(similarity, 0).mean() 
    return - (torch.sum(x * y, dim=-1) / tau).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def gaussian_kernel(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean()

def align_gaussian(x, y, t):
    return (x - y).norm(p=2, dim=1).pow(2).mul(-t).exp().mean()

def gaussian_kernel_tau(x, tau=0.2):
    t = 1 / (2 * tau)
    return (torch.pdist(x, p=2).pow(2).mul(-t) + 2 * t).exp().mean()

def align_gaussian_tau(x, y, tau=0.2):
    t = 1 / (2 * tau)
    return ((x - y).norm(p=2, dim=1).pow(2).mul(-t) + 2 * t).exp().mean()

def riesz_kernel(x, s):
    if s < 0:
        return - torch.pdist(x, p=2).pow(-s)
    else:
        return torch.pdist(x, p=2).pow(-s)

def align_riesz_loss(x, y, s):
    if s < 0:
        return - (x - y).norm(p=2, dim=1).pow(-s).mean()
    else:
        return (x - y).norm(p=2, dim=1).pow(-s).mean()

def riesz_kernel_smooth(x, s):
    if s < 0:
        return - torch.pdist(x, p=2).pow(-s)
    else:
        return torch.pdist(x, p=2).pow(-s)

def align_riesz_loss_smooth(x, y, s):
    if s < 0:
        return - (x - y).norm(p=2, dim=1).pow(-s).mean()
    else:
        return (x - y).norm(p=2, dim=1).pow(-s).mean()

def linear_kernel_loss(x, t, p=2):
    return torch.pdist(x, p=2).pow(p).mul(-t).mean()

def linear_align(x, y, t, p=2):
    return (x - y).norm(p=2, dim=1).pow(p).mul(- 2 * t).mean()

def log_kernel_loss(x, t, p=2):
    return - (1/2) * torch.log(t * torch.pdist(x, p=2).pow(p) + 1).mean()

def log_align(x, y, t, p=2):
    return - torch.log(t * (x - y).norm(p=2, dim=1).pow(p) + 1).mean() # it is 2 * (1 / 2)

def imq_kernel(x, c=2):
    return (c / (c**2 + torch.pdist(x, p=2).pow(2)).sqrt()).mean()

def align_imq(x, y, c):
    return (c / (c**2 + (x - y).norm(p=2, dim=1).pow(2)).sqrt()).mean()

def tolerance_loss(x, y):
    loss = 0
    x = torch.tensor(x)
    c = 0
    for g in np.unique(y):
        idx = np.where(y == g)
        z = scipy.linalg.blas.sgemm(alpha=1.0, a=x[idx], b=x[idx].T)
        z = np.triu(z)
        np.fill_diagonal(z, 0)
        z = np.hstack(z)
        z = z[z != 0]
        c += z.shape[0]
        loss += z.sum()
    
    return loss / c


def effective_rank(z, eps=1e-7):
    s = scipy.linalg.svd(z, full_matrices=False, compute_uv=False)
    while np.linalg.norm(s, 1) < eps:
        s = s * 10
    while np.linalg.norm(s, 1) > 1000:
        s = s / 10
    s = s / (np.linalg.norm(s, 1)) + eps
    entropy = scipy.stats.entropy(s)
    return np.exp(entropy)


def sort_rows_decreasing(distance_matrix, labels):
    # Sort indices for each row in the original array
    sorted_indices = np.argsort(distance_matrix, axis=1)[:, ::-1]
    
    # Use advanced indexing to rearrange elements in the other array
    sorted_labels = np.take_along_axis(labels, sorted_indices, axis=1)

    return sorted_indices, sorted_labels


def lsc(z, y):
    eps = 1e-5
    N = y.shape[0]
    
    z = np.array(z, dtype='float32')
    z = scipy.linalg.blas.sgemm(alpha=1.0, a=z, b=z.T)
    thetas = np.arccos(np.clip(z, -1 + eps, 1 - eps))
    labels = []

    for i in y:
        row = []
        for j in y:
            if i == j:
                row.append(0)
            else:
                row.append(1)
        
        labels.append(row)
    labels = np.array(labels)

    _, neigboring_labels = sort_rows_decreasing(thetas, labels)
    
    cnt = 0
    
    for label in neigboring_labels:
        cnt += np.where(label == 1)[0][0]  + 1
    return cnt / N

def f4(z, y):
    mfe = MFE(groups="complexity", features=["f4"])
    mfe.fit(z, y)
    f4 = mfe.extract() 
    return f4

"""
def wasserstein_uniformity(z, d):
    z = np.array(z)
    mu = np.mean(z, axis=0)
    cov = np.cov(z.T)
    return np.sqrt(np.linalg.norm(mu) ** 2 + 1 + np.trace(cov) - (2 / ( d ** (1/2))) * np.trace(cov ** (1/2)))
"""

class dot_prod_dist(scipy.stats.rv_continuous):
    def _pdf(self,x, d):
        a = scipy.special.gamma((d + 1) / 2) / (scipy.special.gamma(d / 2) * np.sqrt(np.pi))
        return np.power(np.sqrt(1 - x**2), d -2) * a


def wasserstein_uniformity(z, optimal_samples, sub_sample=False):
    
    # calc similarities
    similarity = torch.matmul(z, z.T)
    similarity = torch.triu(similarity, diagonal=1)
    triu = np.triu_indices_from(similarity, k=1)
    s = similarity[triu].cpu()
    if sub_sample:
        s = np.random.choice(s, size=50000, replace=False)
    return scipy.stats.wasserstein_distance(s, optimal_samples)

def angle_dist(z):
    eps = 1e-5

    similarity = torch.matmul(z, z.T)
    similarity = torch.triu(similarity, diagonal=1)
    similarity = similarity[similarity != 0]
    phi = torch.arccos(torch.clip(similarity, -1 + eps, 1 - eps))

    return torch.mean(phi), torch.std(phi) 

def used_dimensions(z):
    eps = 1e-5
    cov = np.cov(z.T)
    rank = np.linalg.matrix_rank(cov, tol=eps)
    return rank
	

def viz_array_grid(array, rows, cols, padding=0, channels_last=False, normalize=False, **kwargs):
    # normalization
    '''
    Args:
        array: (N_images, N_channels, H, W) or (N_images, H, W, N_channels)
        rows, cols: rows and columns of the plot. rows * cols == array.shape[0]
        padding: padding between cells of plot
        channels_last: for Tensorflow = True, for PyTorch = False
        normalize: `False`, `mean_std`, or `min_max`
    Kwargs:
        if normalize == 'mean_std':
            mean: mean of the distribution. Default 0.5
            std: std of the distribution. Default 0.5
        if normalize == 'min_max':
            min: min of the distribution. Default array.min()
            max: max if the distribution. Default array.max()
    '''
    array = tonp(array)
    if not channels_last:
        array = np.transpose(array, (0, 2, 3, 1))

    array = array.astype('float32')

    if normalize:
        if normalize == 'mean_std':
            mean = kwargs.get('mean', 0.5)
            mean = np.array(mean).reshape((1, 1, 1, -1))
            std = kwargs.get('std', 0.5)
            std = np.array(std).reshape((1, 1, 1, -1))
            array = array * std + mean
        elif normalize == 'min_max':
            min_ = kwargs.get('min', array.min())
            min_ = np.array(min_).reshape((1, 1, 1, -1))
            max_ = kwargs.get('max', array.max())
            max_ = np.array(max_).reshape((1, 1, 1, -1))
            array -= min_
            array /= max_ + 1e-9

    batch_size, H, W, channels = array.shape
    assert rows * cols == batch_size

    if channels == 1:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1)))
        array = array[:, :, :, 0]
    elif channels == 3:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1),
                          3))
    else:
        raise TypeError('number of channels is either 1 of 3')

    for i in range(rows):
        for j in range(cols):
            img = array[i * cols + j]
            start_h = i * padding + i * H
            start_w = j * padding + j * W
            canvas[start_h: start_h + H, start_w: start_w + W] = img

    canvas = np.clip(canvas, 0, 1)
    canvas *= 255.0
    canvas = canvas.astype('uint8')
    return canvas


def tonp(x):
    if isinstance(x, (np.ndarray, float, int)):
        return np.array(x)
    return x.detach().cpu().numpy()


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        res = []
        for lr in self.base_lrs:
            res.append(np.maximum(lr * np.minimum(-self.last_epoch * 1. / self.num_epochs + 1., 1.), 0.))
        return res


class LinearWarmupAndCosineAnneal(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up, T_max, last_epoch=-1):
        self.warm_up = int(warm_up * T_max)
        self.T_max = T_max - self.warm_up
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        if self.last_epoch == 0:
            return [lr / (self.warm_up + 1) for lr in self.base_lrs]
        elif self.last_epoch <= self.warm_up:
            c = (self.last_epoch + 1) / self.last_epoch
            return [group['lr'] * c for group in self.optimizer.param_groups]
        else:
            # ref: https://github.com/pytorch/pytorch/blob/2de4f245c6b1e1c294a8b2a9d7f916d43380af4b/torch/optim/lr_scheduler.py#L493
            le = self.last_epoch - self.warm_up
            return [(1 + np.cos(np.pi * le / self.T_max)) /
                    (1 + np.cos(np.pi * (le - 1) / self.T_max)) *
                    group['lr']
                    for group in self.optimizer.param_groups]


class BaseLR(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
