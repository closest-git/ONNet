import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from .OpticalFormer_util import *
# import lite_bert
MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

def unitwise_norm(x,axis=None):
    """Compute norms of each output unit separately, also for linear layers."""
    if len(torch.squeeze(x).shape) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
        return torch.norm(x)
    elif len(x.shape) in [2, 3]:  # Linear layers of shape IO or multihead linear
        # axis = 0
        # axis = 1
        keepdims = True
    elif len(x.shape) == 4:  # Conv kernels of shape HWIO
        if axis is None:
            axis = [0, 1, 2,]
        keepdims = True
    else:
        raise ValueError(f'Got a parameter with shape not in [1, 2, 4]! {x}')
    return torch.sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5

def clip_grad_rc(grad,W,row_major=False,eps = 1.e-3,clip=0.02):   
    #   adaptive_grad_clip
    if len(grad.shape)==2:
        nR,nC = grad.shape
    axis = 1 if row_major else 0
    g_norm = unitwise_norm(grad,axis=axis)
    W_norm = unitwise_norm(W,axis=axis)
    assert(g_norm.shape==W_norm.shape)
    W_norm[W_norm<eps] = eps
    # clipped_grad = grad * (W_norm / g_norm)       
    s = torch.squeeze(clip*W_norm / (g_norm+1.0e-6))     
    s = torch.clamp(s, max=1)
    if len(grad.shape)==1 or s.numel()==nC:       #nC                
        grad = grad*s                
    else:                   #nR           
        grad = torch.einsum('rc,r->rc', grad, s)
    return grad

def clip_grad(model,eps = 1.e-3,clip=0.002,method="agc"):   
    known_modules = {'Linear'} 
    for module in model.modules():
        classname = module.__class__.__name__   
        if classname not in known_modules:
            continue
        if classname == 'Conv2d':
            assert(False)
            grad = None            
        elif classname == 'BertLayerNorm':
            grad = None
        else:
            grad = module.weight.grad.data       
            W = module.weight.data 

        #   adaptive_grad_clip
        assert len(grad.shape)==2
        nR,nC = grad.shape
        axis = 1 if nR>nC else 0
        g_norm = unitwise_norm(grad,axis=axis)
        W_norm = unitwise_norm(W,axis=axis)
        W_norm[W_norm<eps] = eps
        # clipped_grad = grad * (W_norm / g_norm)       
        s = torch.squeeze(clip*W_norm / (g_norm+1.0e-6))     
        s = torch.clamp(s, max=1)
        if s.numel()==nC:       #nC                
            grad = grad*s                
        else:                   #nR           
            grad = torch.einsum('rc,r->rc', grad, s)
        module.weight.grad.data.copy_(grad)

        if module.bias is not None:
            v = module.bias.grad.data
            axis = 0
            b_grad = clip_grad_rc(v,module.bias.data,row_major=axis==1,eps = eps,clip=clip)  
            module.bias.grad.data.copy_(b_grad)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout,clip_grad=""):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.isV0 = False
        for _ in range(depth):
            if self.isV0:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
                ]))
            else:
                # self.layers.append(lite_bert.BTransformer(dim, heads, dim * 4, dropout))
                self.layers.append(BTransformer(dim, heads, dim * 4, dropout,clip_grad=clip_grad))
    def forward(self, x, mask = None):
        if self.isV0:
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        else:
            for BTrans in self.layers:
                x = BTrans(x,mask)
        return x

class OpticalFormer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, ff_hidden, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,clip_grad=""):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2       #64
        patch_dim = channels * patch_size ** 2              #48 pixles in each patch
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size
        self.clip_grad = clip_grad
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches , dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, ff_hidden, dropout,clip_grad=self.clip_grad)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Identity() if self.clip_grad=="agc" else nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def name_(self):
       return "ViT_"

    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # x = rearrange(img, 'b c (h p1) (w p2) -> b (h w c) (p1 p2)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n )]
        # x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    def predict(self,output):
        if self.config.support == "binary":
            nGate = output.shape[1] // 2
            #assert nGate == self.n
            pred = 0
            for i in range(nGate):
                no = 2*(nGate - 1 - i)
                val_2 = torch.stack([output[:, no], output[:, no + 1]], 1)
                pred_i = val_2.max(1, keepdim=True)[1]  # get the index of the max log-probability
                pred = pred * 2 + pred_i
        elif self.config.support == "logit":
            nGate = output.shape[1]
            # assert nGate == self.n
            pred = 0
            for i in range(nGate):
                no = nGate - 1 - i
                val_2 = F.sigmoid(output[:, no])
                pred_i = (val_2+0.5).long()
                pred = pred * 2 + pred_i
        else:
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        return pred

    