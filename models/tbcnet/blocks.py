import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')   
    
class DAttention(nn.Module):

    def __init__(self, view,size=[4, 28, 28],
                 n_heads=8, n_channels=256,
                 stride=2, ksize=5,
                 pe='CRPB'):

        super().__init__()
       
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.n_head_channels = n_channels // n_heads
        self.pe =pe
        
        self.n_group_heads = 2
        self.n_group = n_heads//self.n_group_heads
        self.n_group_channels = int(self.n_channels//self.n_group)
        
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, 
                      ksize, stride, ksize // 2, 
                      groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 
                      kernel_size=1, stride=1, padding=0, bias=False)
        )
        
        self.conv_offset3D = nn.Sequential(
            nn.Conv3d(self.n_group_channels, self.n_group_channels,
                      kernel_size=(2, ksize, ksize), stride=(1, stride, stride),
                      padding=(0, ksize // 2, ksize // 2), 
                      groups=self.n_group_channels),  
            nn.GroupNorm(1, self.n_group_channels),  
            nn.GELU(),
            nn.Conv3d(self.n_group_channels, 3, 
                      kernel_size=(1, 1, 1), stride=(1, 1, 1), 
                      padding=(0, 0, 0), bias=False)
        )
        
        self.offset_range_factor = -1

        self.proj_q = nn.Conv2d(self.n_channels, self.n_channels,
            kernel_size=1, stride=1, padding=0)
        
        self.proj_k = nn.Conv2d(self.n_channels, self.n_channels,
            kernel_size=1, stride=1, padding=0)
        
        self.proj_v = nn.Conv2d(self.n_channels, self.n_channels,
            kernel_size=1, stride=1, padding=0)

        self.proj_out = nn.Sequential(nn.Conv2d(self.n_channels, self.n_channels,
            kernel_size=1, stride=1, padding=0),nn.Dropout(0.1))
        self.att_sm= nn.Sequential(
            nn.Dropout(0.1), 
            nn.Softmax(dim=1)
        )
        self.positional_encoding = nn.Parameter(torch.zeros(3, self.n_channels, size[0],
                                                            size[1], size[2]))
        if pe=='CRPB':
            self.relative_w = nn.Parameter(torch.randn(view),requires_grad=True)
            self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, size[0] * 2 - 1, size[1] * 2 - 1, size[2] * 2 - 1
                                ,requires_grad=True)
                )
            trunc_normal_(self.rpe_table, std=0.01)
            
            self.relative_w_c = nn.Parameter(torch.randn(view),requires_grad=True)
            self.rpe_table_c = nn.Parameter(
                    torch.zeros(self.n_heads, size[0] * 2 - 1, size[1] * 2 - 1, size[2] * 2 - 1
                                ,requires_grad=True)
                )
            trunc_normal_(self.rpe_table, std=0.01)
        else: 
            self.rpe_table = None
            self.positional_encoding = nn.Parameter(
                torch.zeros(3, self.n_channels, size[0], size[1], size[2]), requires_grad=True) 
     
    def forward(self, input_q,input_v):

        V, B, C, T, H, W = input_v.size()
        dtype, device = input_v.dtype, input_v.device

        samples_v=[]
        attn_bias_l=[]
        grid_q = self._get_query_grid(B * self.n_group, input_q.shape[-2], W, V, dtype, device)
        num0 = input_q.shape[-2] * W
        offset_range = torch.as_tensor([1.0 / (T - 1.0), 1.0 / (H - 1.0), 1.0 / (W - 1.0)], device=device)
        offset_range = offset_range.reshape(1, 3, 1, 1, 1)
        for i in range(V):
            v0 = input_v[i] + self.positional_encoding[i][None, :, :,:,:].to(device)
            v_off = rearrange(v0, 'b (g c) t h w ->(b g) c t h w', 
                                    g=self.n_group, c=self.n_group_channels)
            offset0 = self.conv_offset3D(v_off).contiguous()
            offset = offset0.tanh().mul(offset_range)
            offset = rearrange(offset, 'b p t h w ->b t h w p') # p=3

            reference = self._get_ref_grid(offset0)
            pos = (offset + reference).clamp(-1., +1.) 

            sampled_v = F.grid_sample(
                input=v0.reshape(B * self.n_group, self.n_group_channels, T, H, W),
                grid=pos[..., (0,2,1)], 
                align_corners=False)
            samples_v.append(sampled_v)
            
            if self.pe=='CRPB':
        
                displacement = (grid_q.reshape(B * self.n_group, -1, 3).unsqueeze(2) 
                    - pos.reshape(B*self.n_group, -1, 3).unsqueeze(1))
                
                weight =self.relative_w[i]
                displacement0=displacement[..., i*num0:(i+1)*num0,:, :]
                grid0=displacement0.unsqueeze(-2)[..., (0,2,1)]
                rpe_bias = self.rpe_table[None, ...].expand(B, -1, -1, -1,-1)
                
                bias = F.grid_sample(
                        input=rearrange(weight*rpe_bias, 'b (g c) t h w ->(b g) c t h w', 
                                    c=self.n_group_heads, g=self.n_group),
                        grid=grid0,
                        align_corners=False)
                attn_bias0 = bias.reshape(B * self.n_heads, H * W*1,-1)

                weight_C =self.relative_w_c[i]
                if i == 0:
                    displacement_c = displacement[..., (i+1)*num0:, :, :]
                elif i == V-1:
                    displacement_c = displacement[..., :i*num0, :, :]
                else:
                    displacement_c= torch.cat([
                        displacement[..., :i*num0, :, :], 
                        displacement[..., (i+1)*num0:, :, :]
                    ], dim=1)
                grid_C=displacement_c.unsqueeze(-2)[..., (0,2,1)]
                rpe_bias_C = self.rpe_table_c[None, ...].expand(B, -1, -1, -1,-1)
                bias_C = F.grid_sample(
                        input=rearrange(weight_C*rpe_bias_C, 'b (g c) t h w ->(b g) c t h w', 
                                    c=self.n_group_heads, g=self.n_group),
                        grid=grid_C,
                        align_corners=False)
                attn_bias_C = bias_C.reshape(B * self.n_heads, H * W*(V-1), -1)
                if i == 0:
                    attn_bias = torch.cat([attn_bias_C, attn_bias0], dim=1)
                elif i == V-1:
                    attn_bias = torch.cat([attn_bias0, attn_bias_C], dim=1)
                else:
                    attn_bias = torch.cat([attn_bias_C[ :, :num0, :], 
                                           attn_bias0, attn_bias_C[ :, num0:, :]], dim=1)

                attn_bias_l.append(attn_bias)
                
        samples_v = torch.stack(samples_v, dim=0).permute(1, 2, 3, 4, 5,0)
        samples_v = samples_v.reshape(B, C, 1, -1)
        n_sample =samples_v.shape[-1]

        q = self.proj_q(input_q)
        key = self.proj_k(samples_v)
        value = self.proj_v(samples_v)
        
        q = q.reshape(B * self.n_heads, self.n_head_channels, -1)
        key= key.reshape(B * self.n_heads, self.n_head_channels, n_sample)
        value= value.reshape(B * self.n_heads, self.n_head_channels, n_sample)
        
        attn = torch.einsum('b c m, b c n -> b m n', q, key) # B * h, HW, Ns
        attn = attn * self.n_head_channels ** -0.5
        attn = self.att_sm(attn)
        
        if self.pe=='CRPB':
            attn_bias = torch.cat(attn_bias_l, dim=-1)
            attn = attn + attn_bias
            
        out = torch.einsum('b m n, b c n -> b c m', attn, value)
        y = self.proj_out(out.reshape(B, C, H, W*V))
        return y 

    @torch.no_grad()
    def _get_ref_grid(self, x):
        B = x.size(0)
        T0, H0, W0 = x.shape[-3:]

        dtype = x.dtype if x.is_floating_point() else torch.float32
        device = x.device
        ref_t, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T0 - 0.5, T0, dtype=dtype, device=device),
            torch.linspace(0.5, H0 - 0.5, H0, dtype=dtype, device=device),
            torch.linspace(0.5, W0 - 0.5, W0, dtype=dtype, device=device),
            indexing='ij')
        ref = torch.stack((ref_t, ref_y, ref_x), -1)
        ref = ref.div_(torch.tensor([T0 - 1.0, H0 - 1.0, W0 - 1.0],
                                    dtype=dtype, device=device)).mul_(2.0).sub_(1.0)

        return ref[None, ...].expand(B, -1, -1, -1, -1)

    @torch.no_grad()
    def _get_query_grid(self, batch_size, height, width, num_views, dtype, device):
        query_t = torch.zeros(1, dtype=dtype, device=device)
        query_y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
        query_x = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
        ref_t, ref_y, ref_x = torch.meshgrid(query_t, query_y, query_x, indexing='ij')
        ref = torch.stack((ref_t, ref_y, ref_x), -1).squeeze(0)
        ref = ref.repeat(1, num_views, 1)
        return ref[None, ...].expand(batch_size, -1, -1, -1)
    

class LocalAttention(nn.Module):

    def __init__(self, dim, heads, window_size, drop):
        
        super().__init__()

        window_size = to_2tuple(window_size)

        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(drop)
        self.attn_drop = nn.Dropout(drop)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):

        B, C, H, W = x.size()
        r1, r2 = H // self.window_size[0], W // self.window_size[1]

        x_total = rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', 
                                   h1=self.window_size[0], w1=self.window_size[1]) # B x Nr x Ws x C

        x_total = rearrange(x_total, 'b m n c -> (b m) n c')

        qkv = self.proj_qkv(x_total) # B' x N x 3C
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q * self.scale
        q, k, v = [rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)

        if mask is not None:
            # attn : (b * nW) h w w
            # mask : nW ww ww
            nW, ww, _ = mask.size()
            attn = rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww, w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            attn = rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
        attn=attn.softmax(dim=3)
        attn = self.attn_drop(attn)

        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        x = rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj_out(x)) # B' x N x C
        x = rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0], w1=self.window_size[1]) # B x C x H x W

        return x


class ShiftWindowAttention(LocalAttention):

    def __init__(self, dim, heads, window_size, drop, shift_size, fmap_size):

        super().__init__(dim, heads, window_size, drop)

        self.fmap_size = tuple(fmap_size)
        self.shift_size = shift_size

        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

        img_mask = torch.zeros(*self.fmap_size)  # H W
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
        mask_windows = rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)', 
                                        h1=self.window_size[0],w1=self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW ww ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)
      
    def forward(self, x):

        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        sw_x = super().forward(shifted_x, self.attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x   

class MLPBlock(nn.Module):
       def __init__(self, dim_embed, expansion, drop):
           super().__init__()
           self.norm = LayerNormProxy(dim_embed)
           self.mlp = TransformerMLP(dim_embed, expansion, drop)
           self.scale = LayerScale(dim_embed, init_values=1e-4)
           
       def forward(self, x):
           x = self.norm(x)
           x = self.mlp(x)
           x = self.scale(x)
           return x
       
class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))
    
    def forward(self, x):

        _, _, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


    
class LayerScale(nn.Module):

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)
