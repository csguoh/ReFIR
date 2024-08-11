from functools import partial

import torch

from ...util import default, instantiate_from_config


def adain_latent(feat, cond_feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size() # [4, 64, 64]
    C = size[0]
    feat_var = feat.view(C, -1).var(dim=1) + eps
    feat_std = feat_var.sqrt().view(C, 1, 1)
    feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)
    
    cond_feat_var = cond_feat.view(C, -1).var(dim=1) + eps
    cond_feat_std = cond_feat_var.sqrt().view(C, 1, 1)
    cond_feat_mean = cond_feat.view(C, -1).mean(dim=1).view(C, 1, 1)
    feat = (feat - feat_mean.expand(size)) / feat_std.expand(size)

    target_std = (0.9*feat_std + 0.1*cond_feat_std)
    target_mean= (0.9*feat_mean + 0.1*cond_feat_mean)

    return feat * target_std.expand(size) + target_mean.expand(size)


class VanillaCFG:
    """
    implements parallelized CFG
    """

    def __init__(self, scale, dyn_thresh_config=None):
        scale_schedule = lambda scale, sigma: scale  # independent of step
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {
                    "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"
                },
            )
        )

    def __call__(self, x, sigma):
        x_u, x_c = x.chunk(2)
        scale_value = self.scale_schedule(sigma)
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat", "control", 'control_vector', 'mask_x']:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out



class LinearCFG:
    def __init__(self, scale, scale_min=None, dyn_thresh_config=None):
        if scale_min is None:
            scale_min = scale
        scale_schedule = lambda scale, scale_min, sigma: (scale - scale_min) * sigma / 14.6146 + scale_min
        self.scale_schedule = partial(scale_schedule, scale, scale_min)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {
                    "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"
                },
            )
        )

    def __call__(self, x, sigma,is_adain=False):
        x_u, x_c = x.chunk(2)
        scale_value = self.scale_schedule(sigma)
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        # TODO CFG Adain here
        if is_adain:
            x_pred[0]=adain_latent(x_pred[0],x_pred[1])

        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat", "control", 'control_vector', 'mask_x']:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out



class IdentityGuider:
    def __call__(self, x, sigma):
        return x

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out
