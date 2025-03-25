import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from torch import einsum

from einops import rearrange
from basicsr.utils.nano import psf2otf

try:
    from flash_attn import flash_attn_func
except:
    print("Flash attention is required")
    raise NotImplementedError


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, ksize=0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.ksize = ksize
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        if ksize:
            self.avg = torch.nn.AvgPool2d(kernel_size=ksize, stride=1, padding=(ksize-1) //2)


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   

        if self.ksize:
            q = q - self.avg(q)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

    
class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h


##########################################################################
## Overlapping Cross-Attention (OCA)
class OCAB(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias, ksize=0):
        super(OCAB, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5
        self.ksize = ksize

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )
        if ksize:
            self.avg = torch.nn.AvgPool2d(kernel_size=ksize, stride=1, padding=(ksize-1) //2)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        if self.ksize:
            qs = qs - self.avg(qs)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))    
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out
    

class AttentionFusion(nn.Module):
    def __init__(self, dim, bias, channel_fusion):
        super(AttentionFusion, self).__init__()
        
        self.channel_fusion = channel_fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=bias)
        )
        self.dim = dim // 2

    def forward(self, x):
        fusion_map = self.fusion(x)
        if self.channel_fusion:
            weight = F.sigmoid(torch.mean(fusion_map, 1, True))
        else:
            weight = F.sigmoid(torch.mean(fusion_map, (2,3), True))
        fused_feature = x[:, :self.dim] * weight + x[:, self.dim:] * (1-weight) # [:, :self.dim] == SA
        return fused_feature



class Transformer_STAF(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_channel_heads, num_spatial_heads, spatial_dim_head, ffn_expansion_factor, bias, LayerNorm_type, channel_fusion, query_ksize=0):
        super(Transformer_STAF, self).__init__()

        self.spatial_attn = OCAB(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias, ksize=query_ksize)
        self.channel_attn = Attention(dim, num_channel_heads, bias, ksize=query_ksize)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.fusion = AttentionFusion(dim*2, bias, channel_fusion)

    def forward(self, x):
        sa = x + self.spatial_attn(self.norm1(x))
        sa = sa + self.spatial_ffn(self.norm2(sa))
        ca = x + self.channel_attn(self.norm3(x))
        ca = ca + self.channel_ffn(self.norm4(ca))
        fused = self.fusion(torch.cat([sa, ca], 1))

        return fused


class MAFG_CA(nn.Module):
    def __init__(self, embed_dim, num_heads, M, window_size=0, eps=1e-6):
        super().__init__()
        self.M = M
        self.Q_idx = M // 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.M = M
        self.wsize = window_size

        self.proj_high = nn.Conv2d(3, embed_dim, kernel_size=1)
        self.proj_rgb = nn.Conv2d(embed_dim, 3, kernel_size=1)

        self.norm = nn.LayerNorm(embed_dim, eps=eps)
        self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=False)
        self.proj_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.max_seq = 2**16-1

        # window based sliding similar to OCAB
        self.overlap_wsize = int(self.wsize * 0.5) + self.wsize
        self.unfold = nn.Unfold(kernel_size=(self.overlap_wsize, self.overlap_wsize), stride=window_size, padding=(self.overlap_wsize-self.wsize)//2)
        self.scale = self.embed_dim ** -0.5
        self.pos_emb_q = nn.Parameter(torch.zeros(self.wsize**2, embed_dim))
        self.pos_emb_k = nn.Parameter(torch.zeros(self.overlap_wsize**2, embed_dim))
        nn.init.trunc_normal_(self.pos_emb_q, std=0.02)
        nn.init.trunc_normal_(self.pos_emb_k, std=0.02)
            
    def forward(self, x):
        x = self.proj_high(x)
        BM,E,H,W = x.shape

        x_seq = x.view(BM,E,-1).permute(0,2,1)
        x_seq = self.norm(x_seq)
        B = BM // self.M
        QKV = self.qkv(x_seq)
        QKV = QKV.view(BM, H, W, 3, -1).permute(3,0,4,1,2).contiguous()
        Q,K,V = QKV[0], QKV[1], QKV[2]
        Q_bm = Q.view(B, self.M, E, H,W)
        _Q = Q_bm[:, self.Q_idx:self.Q_idx+1]
        Q = torch.stack([__Q.repeat(self.M,1,1,1) for __Q in _Q]).view(BM,E,H,W)

        Q = rearrange(Q, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.wsize, p2 = self.wsize)
        K,V = map(lambda t: self.unfold(t), (K,V))
        if K.shape[-1] > 10000: # Inference
            b,_,pp = K.shape
            K = K.view(b,self.embed_dim,-1,pp).permute(0,3,2,1).reshape(b*pp,-1,self.embed_dim)
            V = V.view(b,self.embed_dim,-1,pp).permute(0,3,2,1).reshape(b*pp,-1,self.embed_dim)
        else:
            K,V = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.embed_dim), (K,V))

        # Absolute positional embedding
        Q = Q + self.pos_emb_q
        K = K + self.pos_emb_k

        s, eq, _ = Q.shape
        _, ek, _ = K.shape
        Q = Q.view(s, eq, self.num_heads,self.head_dim).half()
        K = K.view(s, ek, self.num_heads,self.head_dim).half()
        V = V.view(s, ek, self.num_heads,self.head_dim).half()
        if s > self.max_seq: # maximum allowed sequence of flash attention
            outs = []
            sp = self.max_seq
            _max = s // sp + 1
            for i in range(_max):
                outs.append(flash_attn_func(Q[i*sp: (i+1)*sp], K[i*sp: (i+1)*sp], V[i*sp: (i+1)*sp], causal=False))
            out = torch.cat(outs).to(torch.float32)
        else:
            out = flash_attn_func(Q, K, V, causal=False).to(torch.float32)
        out = rearrange(out, '(b nh nw) (ph pw) h d -> b (nh ph nw pw) (h d)', nh=H//self.wsize, nw=W//self.wsize, ph=self.wsize, pw=self.wsize)
        out = self.proj_out(out)

        mixed_feature =  out.view(BM,H,W,E).permute(0,3,1,2).contiguous() + x
        return self.proj_rgb(mixed_feature).reshape(B,-1,H,W)
             
                
##########################################################################
## Aberration Correction Transformers for Metalens
class ACFormer(nn.Module):
    def __init__(self,
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        channel_heads = [1,2,4,8],
        spatial_heads = [2,2,3,4],
        overlap_ratio=[0.5, 0.5, 0.5, 0.5],
        window_size = 8,
        spatial_dim_head = 16,
        bias = False,
        ffn_expansion_factor = 2.66,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        M=13,
        ca_heads=2,
        ca_dim=32,
        window_size_ca=0,
        query_ksize=None
    ):
        
        super(ACFormer, self).__init__()
        self.center_idx = M // 2
        self.ca = MAFG_CA(embed_dim=ca_dim, num_heads=ca_heads, M=M, window_size=window_size_ca)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[Transformer_STAF(dim=dim, window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, channel_fusion=False, query_ksize=0) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[Transformer_STAF(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[1],  num_channel_heads=channel_heads[1], num_spatial_heads=spatial_heads[1], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, channel_fusion=False, query_ksize=0) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[Transformer_STAF(dim=int(dim*2**2), window_size = window_size, overlap_ratio=overlap_ratio[2],  num_channel_heads=channel_heads[2], num_spatial_heads=spatial_heads[2], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, channel_fusion=False, query_ksize=0) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[Transformer_STAF(dim=int(dim*2**3), window_size = window_size, overlap_ratio=overlap_ratio[3],  num_channel_heads=channel_heads[3], num_spatial_heads=spatial_heads[3], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, channel_fusion=False, query_ksize=query_ksize[0] if i % 2 == 1 else 0) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[Transformer_STAF(dim=int(dim*2**2), window_size = window_size, overlap_ratio=overlap_ratio[2],  num_channel_heads=channel_heads[2], num_spatial_heads=spatial_heads[2], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, channel_fusion=True, query_ksize=query_ksize[1] if i % 2 == 1 else 0) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[Transformer_STAF(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[1],  num_channel_heads=channel_heads[1], num_spatial_heads=spatial_heads[1], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, channel_fusion=True, query_ksize=query_ksize[2] if i % 2 == 1 else 0) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[Transformer_STAF(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, channel_fusion=True, query_ksize=query_ksize[3] if i % 2 == 1 else 0) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[Transformer_STAF(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, channel_fusion=True, query_ksize=query_ksize[4] if i % 2 == 1 else 0) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        if inp_img.ndim == 5:
            B,M,C,H,W = inp_img.shape
            center_img = inp_img[:, self.center_idx]
            inp_img = inp_img.view(B*M,C,H,W).contiguous()
        else:
            center_img = inp_img

        if self.ca is None:
            inp_enc_level1 = inp_img.view(B,M*C,H,W)
        else:
            inp_enc_level1 = self.ca(inp_img)

        inp_enc_level1 = self.patch_embed(inp_enc_level1)
        
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)


        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + center_img

        return out_dec_level1