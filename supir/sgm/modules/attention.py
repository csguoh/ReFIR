import math
from inspect import isfunction
from typing import Any, Optional
import numbers
import torch
import torch.nn.functional as F
# from einops._torch_specific import allow_ops_in_compiled_graph
# allow_ops_in_compiled_graph()
from einops import rearrange, repeat
from packaging import version
from torch import nn
import numpy as np
from sklearn.decomposition import PCA

TOTOAL_ATTN1_LAYER = 36
cur_att_layer = 1 # 用于余弦学习率递减操作


def adain_latentBHLC(feat, cond_feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size() # [B, head, L, C]
    B, head, L , C = size
    feat_var = feat.var(dim=2) + eps
    feat_std = feat_var.sqrt().view(B, head, 1, C)
    feat_mean = feat.mean(dim=2).view(B, head, 1, C)
    
    cond_feat_var = cond_feat.var(dim=2) + eps
    cond_feat_std = cond_feat_var.sqrt().view(B, head, 1, C)
    cond_feat_mean = cond_feat.mean(dim=2).view(B, head, 1, C)
    feat = (feat - feat_mean.expand(size)) / feat_std.expand(size)

    target_std = cond_feat_std#(0.9*feat_std + 0.1*cond_feat_std)
    target_mean= cond_feat_mean#(0.9*feat_mean + 0.1*cond_feat_mean)

    return feat * target_std.expand(size) + target_mean.expand(size)





def cosine_schedule(timestep, total_steps):
    return 0.5 * (1 - torch.cos(torch.tensor(np.pi * timestep / total_steps)))



def exponential_schedule(timestep, total_steps, alpha=5.0):
    # Scale timestep to [0, 1]
    scaled_timestep = timestep / total_steps
    # Exponential function
    weight = 1 - math.exp(-alpha * scaled_timestep)
    return weight


def linear_schedule(timestep, total_steps):
    return timestep / total_steps



def match_histograms(source, target, nbins=256):
    """
    Adjust the histogram of the target features to match that of the source features using NumPy for interpolation.

    Parameters:
        source (torch.Tensor): Source domain features (batchsize x seqlen x channel).
        target (torch.Tensor): Target domain features (batchsize x seqlen x channel).
        nbins (int): Number of bins for histogram calculation.

    Returns:
        torch.Tensor: Target domain features adjusted to have histograms matching the source.
    """
    batchsize, seqlen, channel = source.shape
    source = source.reshape(-1, channel)
    target = target.reshape(-1, channel)

    min_val = min(source.min().item(), target.min().item())
    max_val = max(source.max().item(), target.max().item())

    matched_target = torch.empty_like(target)

    for i in range(channel):
        source_hist = np.histogram(source[:, i].cpu().numpy(), bins=nbins, range=(min_val, max_val))[0]
        target_hist = np.histogram(target[:, i].cpu().numpy(), bins=nbins, range=(min_val, max_val))[0]

        source_cdf = np.cumsum(source_hist).astype(np.float32)
        source_cdf /= source_cdf[-1]
        target_cdf = np.cumsum(target_hist).astype(np.float32)
        target_cdf /= target_cdf[-1]

        interp_values = np.interp(target[:, i].cpu().numpy(),
                                  np.linspace(min_val, max_val, nbins),
                                  np.linspace(min_val, max_val, nbins)[np.argsort(target_cdf)[np.searchsorted(source_cdf, target_cdf, side='right') - 1]])

        matched_target[:, i] = torch.from_numpy(interp_values).to(target.dtype).to(target.device)

    matched_target = matched_target.view(batchsize, seqlen, channel)
    return matched_target

    


def ipfp_adjustment(source, target, step_size=0.1, iterations=5):
    """
    Adjust target features towards source features using Iterative Proportional Fitting Procedure.

    Parameters:
        source (torch.Tensor): Source domain features (batch_size x seq_len x feature_size).
        target (torch.Tensor): Target domain features (batch_size x seq_len x feature_size).
        step_size (float): Step size for adjustment.
        iterations (int): Number of iterations to perform adjustment.

    Returns:
        torch.Tensor: Adjusted target features.
    """
    source_flat = source.view(-1, source.size(-1))
    target_flat = target.view(-1, target.size(-1))

    for _ in range(iterations):
        # Compute joint distributions using outer products
        # Sum over batch and sequence to create a pseudo-joint distribution over feature dimensions
        source_sum = torch.einsum('bi,bj->ij', source_flat, source_flat)
        target_sum = torch.einsum('bi,bj->ij', target_flat, target_flat)

        # Compute adjustment factors
        # Avoid division by zero by adding a small epsilon where target_sum is zero
        adjustment_factor = torch.sqrt(source_sum / (target_sum + 1e-8))

        # Apply adjustment factor to target
        target_flat = step_size * (adjustment_factor @ target_flat.t()).t() + (1 - step_size) * target_flat

    # Reshape back to original dimensions
    adjusted_target = target_flat.view_as(target).contiguous()
    return adjusted_target



import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

def compute_adjacency(data, k=10):
    """
    Compute the k-nearest neighbor adjacency matrix for data using sparse representations.
    """
    # Compute the k-nearest graph as a sparse matrix
    A = kneighbors_graph(data, n_neighbors=k, include_self=True, mode='connectivity')
    return A

def sparse_to_torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor.
    """
    data = data.tocoo().astype(np.float32)
    indices = torch.LongTensor([data.row, data.col])
    values = torch.FloatTensor(data.data)
    shape = torch.Size(data.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def manifold_alignment1(source, target, alpha=0.1, regularization=0.1, k=10):
    """
    Align target features to source features using manifold learning based on graph regularization with sparse matrices.

    Parameters:
        source (torch.Tensor): Source domain features (batch_size x seq_len x feature_size).
        target (torch.Tensor): Target domain features (batch_size x seq_len x feature_size).
        alpha (float): Blending factor for alignment.
        regularization (float): Regularization term for manifold smoothness.
        k (int): Number of nearest neighbors for graph construction.

    Returns:
        torch.Tensor: Manifold aligned features.
    """
    # Reshape and merge all samples to fit the sklearn API
    source_flat = source.view(-1, source.shape[-1]).cpu().numpy()
    target_flat = target.view(-1, target.shape[-1]).cpu().numpy()

    # Compute adjacency matrices
    source_adj = compute_adjacency(source_flat, k=k)
    target_adj = compute_adjacency(target_flat, k=k)

    # Compute the graph Laplacians
    source_lap = sp.csgraph.laplacian(source_adj, normed=False)
    target_lap = sp.csgraph.laplacian(target_adj, normed=False)

    # Blend Laplacians
    blended_lap = alpha * source_lap + (1 - alpha) * target_lap

    # Regularize and convert blended Laplacian to torch tensor
    blended_lap = csr_matrix(blended_lap + regularization * sp.eye(blended_lap.shape[0]))
    blended_lap_torch = sparse_to_torch_sparse(blended_lap).to_dense()

    # Use pseudo-inverse for solving (more stable for potentially singular matrices)
    lap_pseudo_inv = torch.pinverse(blended_lap_torch)

    # Apply the transformation: Y = (L + reg * I)^-1 * X
    target_features_flat = target.view(-1, target.shape[-1]).to(torch.float32)
    adjusted_features = torch.mm(lap_pseudo_inv, target_features_flat.t()).t()

    return adjusted_features.view_as(target)




def manifold_alignment(source, target, alpha=0.1, regularization=0.1):
    """
    Align target features to source features using manifold learning based on graph regularization.

    Parameters:
        source (torch.Tensor): Source domain features (batch_size x seq_len x feature_size).
        target (torch.Tensor): Target domain features (batch_size x seq_len x feature_size).
        alpha (float): Blending factor for alignment.
        regularization (float): Regularization term for manifold smoothness.

    Returns:
        torch.Tensor: Manifold aligned features.
    """
    # Compute adjacency matrices (simple Euclidean for demonstration)
    def adjacency_matrix(data):
        D = torch.cdist(data, data, p=2)
        A = torch.exp(-D**2 / regularization)
        return A

    # Get adjacency for source and target
    source_adj = adjacency_matrix(source.view(-1, source.shape[-1]))
    target_adj = adjacency_matrix(target.view(-1, target.shape[-1]))

    # Graph Laplacians
    source_lap = torch.diag(source_adj.sum(1)) - source_adj
    target_lap = torch.diag(target_adj.sum(1)) - target_adj

    # Blend adjacency matrices
    blended_lap = alpha * source_lap + (1 - alpha) * target_lap

    # Align using the blended Laplacian
    # Simple optimization step assuming `features` as variables (requires solver setup)
    # For demonstration, using gradient descent step mock-up
    adjusted_features = target.view(-1, target.shape[-1]) - 0.01 * torch.matmul(blended_lap, target.view(-1, target.shape[-1]))

    return adjusted_features.view_as(target)



def feature_whitening(features):
    """
    Applies ZCA whitening to the given features.

    Parameters:
        features (torch.Tensor): Input features of shape (batch_size x seq_len x feature_size).
        
    Returns:
        torch.Tensor: Whitened features.
    """
    # Flatten the Batchsize and SeqLen dimensions
    original_shape = features.shape
    features = features.view(-1, features.shape[-1])

    # Compute the mean and subtract it
    mean = torch.mean(features, dim=0)
    features -= mean

    # Compute the covariance matrix
    cov = torch.mm(features.t()/math.sqrt(features.size(0)), features/math.sqrt(features.size(0)))
    U, S, V = torch.svd(cov.float())

    # Compute the ZCA Whitening matrix
    epsilon = 1e-5  # Small constant to prevent division by zero
    ZCA_matrix = torch.mm(U, torch.mm(torch.diag(1.0 / torch.sqrt(S + epsilon)), U.t()))

    # Apply the whitening matrix
    features_whitened = torch.mm(features, ZCA_matrix)

    return features_whitened.view(original_shape)



def robust_scale(source, target):
    """
    Apply robust scaling based on median and interquartile range.

    Parameters:
        source (torch.Tensor): Source domain features (batch_size x seq_len x feature_size).
        target (torch.Tensor): Target domain features (batch_size x seq_len x feature_size).

    Returns:
        torch.Tensor: Scaled target features.
    """
    # Flatten features
    source1 = source.view(-1, source.shape[-1])
    target1 = target.view(-1, target.shape[-1])

    # Compute robust statistics
    sou_q25, sou_median, sou_q75 = torch.quantile(source1.float(), torch.tensor([0.25, 0.5, 0.75]).to(source.device), dim=0)
    sou_iqr = sou_q75 - sou_q25

    tar_q25, tar_median, tar_q75 = torch.quantile(target1.float(), torch.tensor([0.25, 0.5, 0.75]).to(source.device), dim=0)
    tar_iqr = tar_q75 - tar_q25

    # Scale features
    target_scaled = (target1 - tar_median) / (tar_iqr + 1e-5)
    target_scaled = target_scaled * sou_iqr + sou_median
    return target_scaled.view_as(target).to(dtype = target.dtype)



def sinkhorn_transport(source, target, reg_lambda=0.5, num_iters=50):
    bs, sl, ch = source.shape
    source = source.view(-1, ch)
    target = target.view(-1, ch)

    M = torch.cdist(source, target, p=2)**2
    K = torch.exp(-M / reg_lambda)

    a = torch.ones((bs * sl, 1), device=source.device) / (bs * sl)
    b = torch.ones((bs * sl, 1), device=target.device) / (bs * sl)

    u = torch.ones((bs * sl, 1), device=source.device)
    v = torch.ones((bs * sl, 1), device=target.device)

    for _ in range(num_iters):
        u = a / (K @ v)
        v = b / (K.t() @ u)

    G = u * K * v.t()

    target_adjusted = G @ target
    target_adjusted = target_adjusted.view(bs, sl, ch)

    return target_adjusted





def coral(source, target):
    """
    Perform CORAL on the target domain features to align them with the source domain features.
    Assumes input dimensions [Batchsize, SeqLen, Channel].

    Parameters:
        source (torch.Tensor): Source domain features (batch_size x seq_len x feature_size).
        target (torch.Tensor): Target domain features (batch_size x seq_len x feature_size).

    Returns:
        torch.Tensor: Adjusted target domain features.
    """
    # Reshape the inputs to [Batchsize*SeqLen, Channel]
    tmp = target
    source = source.view(-1, source.shape[-1])
    target = target.view(-1, target.shape[-1])

    # Step 1: Standardize features (zero-mean)
    source_mean = torch.mean(source, dim=0, keepdim=True)
    target_mean = torch.mean(target, dim=0, keepdim=True)
    source = source - source_mean
    target = target - target_mean

    # Step 2: Compute covariance matrices
    source_cov = source.t().mm(source) / (source.shape[0] - 1)
    target_cov = target.t().mm(target) / (target.shape[0] - 1)

    # Step 3: Compute the Cholesky decomposition of the source covariance matrix
    source_cov_chol = torch.linalg.cholesky(source_cov + 1e-5 * torch.eye(source_cov.size(0), device=source.device))

    # Step 4: Compute the inverse Cholesky decomposition of the target covariance matrix
    target_cov_chol_inv = torch.linalg.inv(torch.linalg.cholesky(target_cov + 1e-5 * torch.eye(target_cov.size(0), device=target.device)))

    # Step 5: Apply the CORAL transform
    target_aligned = target.mm(target_cov_chol_inv).mm(source_cov_chol)

    # Reshape back to the original shape [Batchsize, SeqLen, Channel]
    target_aligned = target_aligned.view_as(tmp)

    return target_aligned




def PCA_align(feat,feat_cond):
    B, L, C = feat.shape
    # Reshape the features to [B*L, C] for PCA
    source_feat_flat = feat_cond.view(B * L, C).cpu().numpy()
    target_feat_flat = feat.view(B * L, C).cpu().numpy()

    # Compute the mean and covariance matrix of the source domain features
    source_mean = np.mean(source_feat_flat, axis=0)

    # Perform PCA on the source domain features
    pca = PCA(n_components=C)
    pca.fit(source_feat_flat - source_mean)

    # Project the target domain features onto the principal components of the source domain
    transformed_target_feat_flat = pca.transform(target_feat_flat - source_mean)
    transformed_source_feat_flat = pca.transform(source_feat_flat - source_mean)
    # Reshape the transformed features back to [B, L, C] and convert to torch tensor
    transformed_target_feat = torch.from_numpy(transformed_target_feat_flat.reshape(B, L, C)).to(device=feat.device, dtype=feat.dtype)
    transformed_source_feat = torch.from_numpy(transformed_source_feat_flat.reshape(B, L, C)).to(device=feat.device, dtype=feat.dtype)
    return transformed_target_feat, transformed_source_feat



def ZCA_align(feat, feat_cond):
    B, L, C = feat_cond.size()
    feat_reshaped = feat_cond.view(-1, C)
    feat = feat.view(-1,C)
    # Compute the mean of the feature maps
    mean = torch.mean(feat_reshaped, dim=0)
    # Subtract the mean from the feature maps
    feat_centered = feat_reshaped - mean
    feat_centered = feat_centered.float()
    # Compute the covariance matrix
    cov_matrix = torch.matmul((feat_centered.T)/math.sqrt(B*L), (feat_centered)/math.sqrt(B*L))
    # Perform Singular Value Decomposition (SVD)
    
    U, S, V = torch.svd(cov_matrix.float())
    
    # Apply ZCA whitening
    epsilon = 1e-5  # Small constant to avoid division by zero
    zca_matrix = torch.matmul(U, torch.matmul(torch.diag(1.0 / torch.sqrt(S + epsilon)), U.T))
    whitened_feat_reshaped = torch.matmul(feat_centered, zca_matrix.T)
    feat = torch.matmul(feat-mean, zca_matrix.T)
    # Reshape the whitened feature maps back to [B, L, C]
    whitened_feat = whitened_feat_reshaped.view(B, L, C)
    feat = feat_reshaped.view(B, L, C)
    return feat, whitened_feat


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size=3, sigma=0.5, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input,hw_size=None):
        # 先改变到二维大小，之后使用高斯滤波，最后把滤波后到结果重塑回来
        B,L,C = input.shape
        if hw_size == None:
            H=W=torch.sqrt(L).long()
        else:
            H,W=hw_size
        assert H*W == L
        x = input.reshape(B,H,W,C).permute(0,3,1,2).contiguous() # B,C,H,W
        x = F.conv2d(x, weight=self.weight.to(input.dtype), groups=self.groups,padding=1)
        x = x.reshape(B,C,L).permute(0,2,1).contiguous()
        return x



def adain_latent1(feat, cond_feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size() # [head, L, C]
    head, L , C = size
    feat_var = feat.var(dim=1) + eps
    feat_std = feat_var.sqrt().view(head, 1, C)
    feat_mean = feat.mean(dim=1).view(head, 1, C)
    
    cond_feat_var = cond_feat.var(dim=1) + eps
    cond_feat_std = cond_feat_var.sqrt().view(head, 1, C)
    cond_feat_mean = cond_feat.mean(dim=1).view(head, 1, C)
    feat = (feat - feat_mean.expand(size)) / feat_std.expand(size)

    target_std = (0.9*feat_std + 0.1*cond_feat_std)
    target_mean= (0.9*feat_mean + 0.1*cond_feat_mean)

    return feat * target_std.expand(size) + target_mean.expand(size)



if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

from .diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs
    ):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # TODO ==========已知需要进行跨帧注意力，实现跨帧注意力==================
            b, l, _ = q.shape # [B,L,C]
            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .contiguous(),
                (q, k, v),
            )
            inp_q,ref_q = torch.stack([q[0],q[2]]).reshape(-1,q.shape[-2],q.shape[-1]),torch.stack([q[1],q[3]]).reshape(-1,q.shape[-2],q.shape[-1]) # [num_head, L, C]
            inp_k,ref_k = torch.stack([k[0],k[2]]).reshape(-1,k.shape[-2],k.shape[-1]),torch.stack([k[1],k[3]]).reshape(-1,k.shape[-2],k.shape[-1]) # [num_head, L, C]
            inp_v,ref_v = torch.stack([v[0],v[2]]).reshape(-1,v.shape[-2],v.shape[-1]),torch.stack([v[1],v[3]]).reshape(-1,v.shape[-2],v.shape[-1]) # [num_head, L, C]

            if n_times_crossframe_attn_in_self == 1: 
                #Cat
                inp_out = xformers.ops.memory_efficient_attention(
                    inp_q, torch.cat([inp_k,ref_k],dim=1), torch.cat([inp_v,ref_v],dim=1), attn_bias=None, op=self.attention_op
                ).reshape(-1,self.heads,q.shape[-2],q.shape[-1])
            elif n_times_crossframe_attn_in_self ==2: 
                # Replace
                inp_out = xformers.ops.memory_efficient_attention(
                    inp_q, ref_k, ref_v, attn_bias=None, op=self.attention_op
                ).reshape(-1,self.heads, q.shape[-2], q.shape[-1])


            elif n_times_crossframe_attn_in_self ==3: 
                # DynaimcCrafter
                inp_out1 = xformers.ops.memory_efficient_attention(
                    inp_q, inp_k, inp_v, attn_bias=None, op=self.attention_op
                ).reshape(-1,self.heads,l,self.dim_head)


                # TODO 对inp_q进行归一化
                # tmp1=adain_latent1(inp_q.view(2,self.heads,l,self.dim_head)[0], ref_k.view(2,self.heads,ref_k.shape[-2],self.dim_head)[0])
                # tmp2=adain_latent1(inp_q.view(2,self.heads,l,self.dim_head)[1], ref_k.view(2,self.heads,ref_k.shape[-2],self.dim_head)[1])
                # inp_q = torch.stack([tmp1,tmp2]).view(-1,l,self.dim_head)
                
                #GS2 = GaussianSmoothing(self.dim_head).to(ref_k.device)
                #ref_k_ = GS2(ref_k,(global_h,global_w))


                inp_out2 = xformers.ops.memory_efficient_attention(
                    inp_q, ref_k, ref_v, attn_bias=None, op=self.attention_op
                ).reshape(-1,self.heads,l,self.dim_head)

                global cur_att_layer
                #weight = cosine_schedule(cur_att_layer, TOTOAL_ATTN1_LAYER)
                
                scale = 1.
                from SUPIR.modules.SUPIR_v0 import decoder_h,decoder_w, register_msk

                register_msk = torch.nn.functional.interpolate(register_msk,size=(decoder_h,decoder_w), mode='bilinear', align_corners=False)
                
                register_msk = register_msk.reshape(2,1,-1,1) # B, head, L ,C
                
                assert register_msk.shape[2] == inp_out2.shape[2], 'the L must equal in interplote msk'
                
                register_msk = scale * register_msk
                
                #inp_out = 0.6*inp_out1 + 0.4*inp_out2 # 这个地方可能需要1-X，未来确保归一化
                inp_out_new = (1-register_msk)*inp_out1 + register_msk*inp_out2 #
                inp_out = adain_latentBHLC(inp_out_new, inp_out1)
                cur_att_layer += 1 # 只统计进入attn1的层数个数
                if cur_att_layer == TOTOAL_ATTN1_LAYER+1:
                    cur_att_layer = 1


            ref_out = xformers.ops.memory_efficient_attention(
                ref_q, ref_k, ref_v, attn_bias=None, op=self.attention_op
            ).reshape(-1,self.heads,l,self.dim_head)

            out = torch.stack([inp_out[0],ref_out[0],inp_out[1],ref_out[1]]).permute(0, 2, 1, 3).reshape(b,l, self.heads * self.dim_head)

        else:
            b, _, _ = q.shape # [B,L,C]
            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (q, k, v),
            )

            # actually compute the attention, what we cannot get enough of
            out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )

            # TODO: Use this directly in the attention operation, as a bias
            if exists(mask):
                raise NotImplementedError
            out = (
                out.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print(
                "We do not support vanilla attention anymore, as it is too expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        return checkpoint(
            self._forward, (x, context,n_times_crossframe_attn_in_self), self.parameters(), self.checkpoint # TODO 需要加一个参数，或者使用一个字典来做
        )

    def _forward(
        self, x, context=None, n_times_crossframe_attn_in_self=0, additional_tokens=None
    ):# TODO 调用的使用解包了
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens,
                #n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerSingleLayerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # on the A100s not quite as fast as the above version
        # (todo might depend on head_dim, check, falls back to semi-optimized kernels for dim!=[16,32,64,128])
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attn_mode="softmax",
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
    ):
        super().__init__()
        print(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None,n_times_crossframe_attn_in_self=0): # TODO 需要加上控制超参数，直接名字参数就可以
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        global global_h,global_w
        global_h = h
        global_w = w
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i],n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self) # TODO 需要加上控制超惨
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
