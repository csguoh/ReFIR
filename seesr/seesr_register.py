import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple, List, Callable, Dict
from torchvision.utils import save_image
from einops import rearrange, repeat



try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1 # 如果把整个Unet的所有注意力机制都走完了，说明已经完成了一次去噪过程。这个实现太妙了
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = xformers.ops.memory_efficient_attention(query=q,key=k,value=v,scale= kwargs.get("scale"))     
        
        #out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0





def regiter_attention_editor_diffusersV0(model,editor):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        # TODO 这里可以根据不同的place_in_unet来进行Hijack,Encoder/ Decoder/ Midblk
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask
            #self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs
            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
  
            out = torch.einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
            return to_out(out)

        return forward

    def register_editor(net,place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet) # 那这个net 就是class Attention
                return
            elif hasattr(net, 'children'):
                count = register_editor(subnet, place_in_unet)
        return


    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            ...
            #register_editor(net, "down") # Encoder
        elif "mid" in net_name:
            ...
            #register_editor(net, "mid")
        elif "up" in net_name:
            register_editor(net, "up") # Decoder




def regiter_attention_editor_diffusers(model, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = None#torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = None #sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count # 这个用来统计整个Unet使用了多少Attn，方便后续统计扩散步数



class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        #print("MasaCtrl at denoising steps: ", self.step_idx)
        #print("MasaCtrl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, num_heads, **kwargs):
        """
        Performing vanilla attention for a batch of queries, keys, and values
        input: [Bhead, L, C]
        output: [B, head, L, C]
        """
        out = xformers.ops.memory_efficient_attention(query=q,key=k,value=v,scale= kwargs.get("scale"))        
        # sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        # attn = sim.softmax(-1) #[8,2048,1024]
        # out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, '(b h) n d -> b h n d', h=num_heads)
        return out



    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        #return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        if is_cross or self.cur_step<30 or self.cur_att_layer < 28:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)


        q, k, v = map(lambda t: rearrange(t, '(b h) n d -> b h n d', h=num_heads), (q, k, v))

        # Q: [B,head,L,c]
        inp_q,ref_q = torch.stack([q[0],q[2]]).reshape(-1,q.shape[-2],q.shape[-1]),torch.stack([q[1],q[3]]).reshape(-1,q.shape[-2],q.shape[-1]) # [Bnum_head, L, C]
        inp_k,ref_k = torch.stack([k[0],k[2]]).reshape(-1,k.shape[-2],k.shape[-1]),torch.stack([k[1],k[3]]).reshape(-1,k.shape[-2],k.shape[-1]) # [Bnum_head, L, C]
        inp_v,ref_v = torch.stack([v[0],v[2]]).reshape(-1,v.shape[-2],v.shape[-1]),torch.stack([v[1],v[3]]).reshape(-1,v.shape[-2],v.shape[-1]) # [Bnum_head, L, C]


        inp_out1 = self.attn_batch(inp_q, inp_k, inp_v,num_heads,**kwargs)


        inp_out2 = self.attn_batch(inp_q, ref_k, ref_v, num_heads,**kwargs)

        weight = cosine_schedule(self.cur_att_layer-28, 64-28)
        scale = 1.0

        # 这个地方Encoder也会用啊，不对，前面的layer超参数可以控制哈哈
        #global register_mask # [2,1,H,W] 应该选择性的把未激活的像素选择不使用
        from models.unet_2d_condition import register_mask # 必须要在用的时候导入才会使用最新的值
        register_msk =  register_mask 
        #register_msk = upsample_image(register_msk,inp_out2.shape[2])
        # upscale = math.sqrt(inp_q.shape[1] // (register_msk.shape[-1] * register_msk.shape[-2]))
        from models.unet_2d_blocks import decoder_h,decoder_w
        register_msk = torch.nn.functional.interpolate(register_msk,size=(decoder_h,decoder_w), mode='bilinear', align_corners=False)
        register_msk = register_msk.reshape(2,1,-1,1) # B, head, L ,C
        # if register_msk.shape[2] != inp_out2.shape[2]:
        #     print()
        assert register_msk.shape[2] == inp_out2.shape[2], 'the L must equal in interplote msk'

        register_msk = scale * weight
        inp_out_new = (1-register_msk)*inp_out1 + register_msk*inp_out2 # 这个地方可能需要1-X，未来确保归一化

        inp_out = adain_latentBHLC(inp_out_new, inp_out1)

        ref_out = self.attn_batch(ref_q, ref_k, ref_v,num_heads,**kwargs) # [B, head, L, C]


        out = torch.stack([inp_out[0],ref_out[0],inp_out[1],ref_out[1]]) 
        out = rearrange(out, 'b h n d -> b n (h d)', h=num_heads)


        return out




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



def adain_latent(feat, cond_feat, eps=1e-5):
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




# 假设 input 是原始低分辨率图像，大小为 [B, C, H, W]
# L 是给定的上采样后的总像素个数
def upsample_image(input, L):
    B, C, H, W = input.shape
    W_new = int((L * W / H) ** 0.5)
    H_new = int((L * H / W) ** 0.5)

    # 四舍五入修正
    if H_new * W_new < L:
        if W_new < H_new:
            W_new += 1
        else:
            H_new += 1

    if H_new * W_new > L:
        if W_new > H_new:
            W_new -= 1
        else:
            H_new -= 1

    # 使用双线性插值上采样图像
    output = F.interpolate(input, size=(H_new, W_new), mode='bilinear', align_corners=False)
    return output
