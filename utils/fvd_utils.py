
import os
import numpy as np
from einops import rearrange
from sklearn.metrics.pairwise import polynomial_kernel

import torch
import torch.nn.functional as F
from extralibs.pytorch_i3d import InceptionI3d

MAX_BATCH = 8 #16
FVD_SAMPLE_SIZE = 2048
TARGET_RESOLUTION = (224, 224)

def preprocess(videos, target_resolution):
    # videos in {0, ..., 255} as np.uint8 array
    b, t, h, w, c = videos.shape
    all_frames = torch.FloatTensor(videos).flatten(end_dim=1) # (b * t, h, w, c)
    all_frames = all_frames.permute(0, 3, 1, 2).contiguous() # (b * t, c, h, w)
    resized_videos = F.interpolate(all_frames, size=target_resolution,
                                   mode='bilinear', align_corners=False)
    resized_videos = resized_videos.view(b, t, c, *target_resolution)
    output_videos = resized_videos.transpose(1, 2).contiguous() # (b, c, t, *)
    scaled_videos = 2. * output_videos / 255. - 1 # [-1, 1]
    return scaled_videos

def preprocess2(videos, target_resolution):
    # videos in tensor in -1~1
    all_frames = rearrange(videos, 'b c t h w -> (b t) c h w')
    resized_videos = F.interpolate(all_frames, size=target_resolution,
                                   mode='bilinear', align_corners=False)
    return resized_videos

def preprocess_styleganv_i3d(videos, target_resolution):
    # videos in {0, ..., 255} as np.uint8 array
    b, t, h, w, c = videos.shape
    all_frames = torch.FloatTensor(videos).flatten(end_dim=1) # (b * t, h, w, c)
    all_frames = all_frames.permute(0, 3, 1, 2).contiguous() # (b * t, c, h, w)
    resized_videos = F.interpolate(all_frames, size=target_resolution,
                                   mode='bilinear', align_corners=False)
    resized_videos = resized_videos.view(b, t, c, *target_resolution)
    output_videos = resized_videos.transpose(1, 2).contiguous() # (b, c, t, *)
    # scaled_videos = 2. * output_videos / 255. - 1 # [-1, 1]
    return output_videos

def get_fvd_logits(videos, i3d, device, batch_size=None):
    videos = preprocess(videos, TARGET_RESOLUTION)
    # videos = preprocess_styleganv_i3d(videos, TARGET_RESOLUTION)
    embeddings = get_logits(i3d, videos, device, batch_size=batch_size)
    return embeddings

def load_fvd_model(device, i3d_path):
    i3d = InceptionI3d(400, in_channels=3).to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if i3d_path is None:
        i3d_path = os.path.join(current_dir, 'i3d_pretrained_400.pt')
        assert(os.path.exists(i3d_path))
    i3d.load_state_dict(torch.load(i3d_path, map_location=device))
    i3d.eval()
    return i3d

def load_stylegan_v_i3d(device):
    fpath = '/data/code/stylegan-v/i3d_torchscript.pt'
    return torch.jit.load(fpath).eval().to(device)

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1) # unbiased estimate
    m_center = m - torch.mean(m, dim=1, keepdim=True)
    mt = m_center.t()  # if complex: mt = m.t().conj()
    return fact * m_center.matmul(mt).squeeze()


def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = x1.mean(dim=0), x2.mean(dim=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)

    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
    trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    mean = torch.sum((m - m_w) ** 2)
    fd = trace + mean
    return fd


def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)
    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)
    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum
    return mmd


def get_logits(i3d, videos, device, batch_size=None):
    detector_kwargs = dict(rescale=True, resize=True, return_features=True)# 3
    if batch_size is None:
        batch_size = MAX_BATCH
    # assert videos.shape[0] % batch_size == 0, f'{videos.shape[0]}, {batch_size}'
    with torch.no_grad():
        logits = []
        for i in range(0, videos.shape[0], batch_size):
            batch = videos[i:i + batch_size].to(device)
            logits.append(i3d(batch))
            # logits.append(i3d(batch, **detector_kwargs))#4
        logits = torch.cat(logits, dim=0)
        return logits


def compute_fvd(real, samples, i3d, device=torch.device('cpu')):
    # real, samples are (N, T, H, W, C) numpy arrays in np.uint8
    real, samples = preprocess(real, (224, 224)), preprocess(samples, (224, 224))
    first_embed = get_logits(i3d, real, device)
    second_embed = get_logits(i3d, samples, device)

    return frechet_distance(first_embed, second_embed)

