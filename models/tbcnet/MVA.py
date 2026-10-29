import torch 
from torch import nn,einsum
import torch.nn.functional as F
from torch.nn.functional import normalize
from torchvision.models.video import r3d_18
from torchvision.models.feature_extraction import create_feature_extractor 
from einops import rearrange

from .blocks import DAttention,LayerNormProxy, LayerScale, LocalAttention, MLPBlock, ShiftWindowAttention


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input):
        if self.training and input.dim() == 2 and input.size(0) == 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


class TBCNet(nn.Module):
    
    def __init__(self,view_num, low_feature_dim,high_feature_dim,num_actions,device,name):
        super(TBCNet, self).__init__()
        
        self.view = view_num
        self.low_feature_dim=low_feature_dim
        self.high_feature_dim=high_feature_dim
        self.name = name

        self.encoder=Encoder(device)
        
        self.fusion_net = GAM(self.low_feature_dim)
        
        self.crossTrans = DA_Transformer(self.view,self.low_feature_dim)
        
        self.Specific_view = branch(self.low_feature_dim,self.low_feature_dim)

        self.pool2D = nn.AdaptiveAvgPool2d(1)
        
        self.Linear_v = nn.Linear(self.low_feature_dim, self.high_feature_dim)
        self.Linear_c = nn.Linear(self.low_feature_dim, self.high_feature_dim)
        self.Classifier= nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(self.high_feature_dim, num_actions),
            nn.Softmax(dim=1) 
        )

        self.Classifier_v= nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.high_feature_dim, num_actions),
            nn.Softmax(dim=1) 
        )
        self.bn_h = SafeBatchNorm1d(self.high_feature_dim)
    def forward(self, xs):
        low_fs = []
        view_fs0 = []
        view_out0 = []
        for v in range(self.view):
            x = xs[:, v]

            low_f0 = self.encoder(x) 
            low_fs.append(low_f0) 

            view_f = low_f0.mean(dim=2)
            view_f = self.Specific_view(view_f)

            view_f = self.pool2D(view_f).squeeze(-1).squeeze(-1)
            view_f =normalize(self.Linear_v(view_f), dim=1) 
            view_fs0.append(view_f)

            view_out=self.Classifier_v(view_f)
            view_out0.append(view_out)
        
        view_fs = torch.stack(view_fs0,dim=1)
        view_out= torch.stack(view_out0,dim=1)
        
        com_all = torch.cat(low_fs, -1)

        com_q,com_qt = self.fusion_net(com_all)
        
        com_fs = torch.stack(low_fs,dim=0)
        com_f = self.crossTrans(com_q,com_fs)
        com_f = self.pool2D(com_f).squeeze(-1).squeeze(-1)
        com_n = normalize(self.Linear_c(com_f), dim=1) 
        com_n = self.bn_h(com_n)
        out = self.Classifier(com_n)
        
        return out,view_out,com_n, view_fs
    
class TBCNet_test(nn.Module):
    def __init__(self,view_num, low_feature_dim,high_feature_dim,num_actions,device,name):
        super(TBCNet_test, self).__init__()
        
        self.view = view_num
        self.low_feature_dim=low_feature_dim
        self.high_feature_dim=high_feature_dim
        self.name = name
        self.encoder=Encoder(device)
        
        self.fusion_net = GAM(self.low_feature_dim)
        
        self.crossTrans = DA_Transformer(self.view,self.low_feature_dim)
        
        self.pool2D = nn.AdaptiveAvgPool2d(1)
        
        self.Linear_v = nn.Linear(self.low_feature_dim, self.high_feature_dim)
        self.Linear_c = nn.Linear(self.low_feature_dim, self.high_feature_dim)

        self.Classifier= nn.Sequential(
            nn.Dropout(0.1),  
            nn.Linear(self.high_feature_dim, num_actions),
            nn.Softmax(dim=1)  
        )

        self.Classifier_v= nn.Sequential(
            nn.Dropout(0.1),  
            nn.Linear(self.high_feature_dim, num_actions),
            nn.Softmax(dim=1) 
        )
        self.bn_h = SafeBatchNorm1d(self.high_feature_dim)
    def forward(self, xs):
        low_fs = []
        for v in range(self.view):

            x = xs[:, v]
            
            low_f0 = self.encoder(x) 
            low_fs.append(low_f0) 
        
        com_all = torch.cat(low_fs, -1)

        com_q,com_qt = self.fusion_net(com_all)
        
        com_fs = torch.stack(low_fs,dim=0)
        com_f = self.crossTrans(com_q,com_fs)
        com_f = self.pool2D(com_f).squeeze(-1).squeeze(-1)
        com_n = normalize(self.Linear_c(com_f), dim=1)
        com_n = self.bn_h(com_n)
        out = self.Classifier(com_n)
        
        return out      
 
 

class TBCNet_main(nn.Module):

    def __init__(self, view_num, low_feature_dim,high_feature_dim,num_actions,device,name): # fusion
        super(TBCNet_main, self).__init__()
        self.view = view_num
        self.low_feature_dim=low_feature_dim
        self.high_feature_dim=high_feature_dim
        

        self.name = name
        self.encoder=Encoder(device)

        self.fusion_net = GAM(self.low_feature_dim)
        self.crossTrans = DA_Transformer(self.view,self.low_feature_dim)

        self.pool2D = nn.AdaptiveAvgPool2d(1)
        self.Linear_c = nn.Linear(self.low_feature_dim, self.high_feature_dim)
        self.bn_h = SafeBatchNorm1d(self.high_feature_dim)
        self.Classifier= nn.Sequential(
            nn.Dropout(0.1),  
            nn.Linear(self.high_feature_dim, num_actions),
            nn.Softmax(dim=1) 
        )
 
    def forward(self, xs):
        low_fs = []
        for v in range(self.view):
            x = xs[:, v]
            
            low_f0 = self.encoder(x) 
            low_fs.append(low_f0) 
        
        com_all = torch.cat(low_fs, -1)

        com_q,com_qt = self.fusion_net(com_all)
        
        com_fs = torch.stack(low_fs,dim=0)
        com_f = self.crossTrans(com_q,com_fs)
        com_f = self.pool2D(com_f).squeeze(-1).squeeze(-1)
        com_n = normalize(self.Linear_c(com_f), dim=1)
        com_n = self.bn_h(com_n)
        out = self.Classifier(com_n)
        
        return  out

    
    
class TBCNet_trans(TBCNet_main):
    def __init__(self, weights_path,view_num, low_feature_dim,high_feature_dim,num_actions,device,name): 

        super(TBCNet_trans, self).__init__(view_num, low_feature_dim,high_feature_dim,num_actions,device,name)
        
        model0 = TBCNet_main(view_num, low_feature_dim,high_feature_dim,num_actions,device,name).to(device) 
        # r3d_18
        weights = torch.load(weights_path,map_location='cpu')
        model0.load_state_dict(weights)
        
        self.name = model0.name+'v'
        
        self.encoder=model0.encoder

        self.crossTrans = model0.crossTrans

        self.fusion_net= model0.fusion_net
        
        self.Linear_c = model0.Linear_c
        self.Classifier= model0.Classifier
        
        self.Specific_view = branch(self.low_feature_dim,self.low_feature_dim)
        self.Linear_v = nn.Linear(self.low_feature_dim, self.high_feature_dim)
        self.Classifier_v= nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.high_feature_dim, num_actions),
            nn.Softmax(dim=1)
        )
        
    def forward(self, xs):
        low_fs = []
        view_fs0 = []
        view_out0 = []
        for v in range(self.view):
            x = xs[:, v]
            
            low_f0 = self.encoder(x)
            low_fs.append(low_f0) 
            
            view_f = low_f0.mean(dim=2)
            view_f = self.Specific_view(view_f)
            
            view_f = self.pool2D(view_f).squeeze(-1).squeeze(-1)
            view_f =normalize(self.Linear_v(view_f), dim=1) 
            view_fs0.append(view_f)
            
            view_out=self.Classifier_v(view_f)
            view_out0.append(view_out)
        
        view_fs = torch.stack(view_fs0,dim=1)
        view_out= torch.stack(view_out0,dim=1)
        
        com_all = torch.cat(low_fs, -1)

        com_q,com_qt = self.fusion_net(com_all)
        
        com_fs = torch.stack(low_fs,dim=0)
        com_f = self.crossTrans(com_q,com_fs)
        com_f = self.pool2D(com_f).squeeze(-1).squeeze(-1)
        com_n = normalize(self.Linear_c(com_f), dim=1) 
        # print(com_n.shape)
        com_n = self.bn_h(com_n)
        out = self.Classifier(com_n)
        
        return out ,view_out,com_n, view_fs
    
    
class Encoder(nn.Module):
    def __init__(self,d):
        super(Encoder, self).__init__()
        self.hidden_dim = 128 
        self.model = r3d_18(weights=None)
        
        self.extractor = create_feature_extractor(self.model, return_nodes={"layer3": "low_features"}) 

    def forward(self, inputs): 
        inputs = inputs.permute(0, 2, 1, 3, 4) 
        features_d = self.extractor(inputs)
        features =features_d["low_features"]
        return features # 
    
class GAM(nn.Module):
    def __init__(self,c_dim=256, time_dim=4):
        super(GAM, self).__init__()
        self.inp_dim = c_dim
        self.time_heads = time_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.q = nn.Linear(time_dim, time_dim, bias=False)
        self.k = nn.Linear(time_dim, time_dim, bias=False)
        self.scale = (time_dim/self.time_heads) ** -0.5
        
        self.time_out = nn.Sequential(
                nn.Conv1d(self.inp_dim, self.inp_dim*2, 
                          kernel_size=1, bias=False),
                nn.BatchNorm1d(self.inp_dim * 2),
                nn.GELU(),
                nn.Conv1d(self.inp_dim*2, self.inp_dim,
                          kernel_size=1, bias=False),
                nn.Dropout(0.1),nn.Sigmoid()
            )
        
        self.time_in = nn.Sequential(
            nn.Conv1d(self.inp_dim,
                      self.inp_dim // 2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), 
            nn.BatchNorm1d(self.inp_dim // 2),
            nn.GELU(),
            nn.Conv1d(self.inp_dim // 2, self.inp_dim, 1, bias=False),
            nn.Dropout(0.1),nn.Sigmoid())
        self.ln = LayerNormProxy(self.inp_dim)
    def forward(self, x):
        b, c, t, h, w = x.shape
        residual = x

        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x0 = self.avg_pool(x)
        x = rearrange(x0, '(b t) c h w -> b c (t h w)', t=t)

        q, k = self.q(x), self.k(x)
        v = rearrange(residual, 'b c t h w -> b t c (h w)')
        
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.time_heads), [q, k])
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out0 = einsum('b h i j, b h j d -> b h i d', attn, v)
        out0 = rearrange(out0, 'b t c (h w) -> b c t h w', h=h, w=w)
        
        out = out0 + residual
        
        temporal_embedding = self.avg_pool(out0).squeeze(-1).squeeze(-1)
        temporal_weight = self.time_out(temporal_embedding)+self.time_in(temporal_embedding)
        weight0=temporal_weight[:, :, :, None, None]
        out_t = out0 * weight0 + residual
        
        out=self.ln(out.mean(dim=2))
        out_t=self.ln(out_t.mean(dim=2))

        return out,out_t
    

class DA_Transformer(nn.Module):
    def __init__(self,view,dim_embed=256,  
                 fmap_size=[4,28,28],
                 expansion=2,heads=8, 
                 drop=0.1):

        super().__init__()
        self.depths = 2
        self.proj = nn.Conv2d(dim_embed, dim_embed, 1, 1, 0) 
        self.view=view
        # Simplified stage_spec and related components
        self.use_lpu = True
        # Adjusted layer definitions
        self.local_perception_units = nn.Conv2d(dim_embed, dim_embed,
                                                kernel_size=3, stride=1, padding=1, groups=dim_embed) 
             
        self.layer_norms = LayerNormProxy(dim_embed) 
        self.attns = DAttention(view=self.view,size=fmap_size,
                    n_heads=heads, n_channels=dim_embed,pe='CRPB') 
        self.layer_scales = LayerScale(dim_embed, init_values=1e-4)
        self.mlps = MLPBlock(dim_embed, expansion, drop)
        
    def forward(self, x,x_v):
        x = self.proj(x)

        if self.use_lpu:
            x0 = x
            x = self.local_perception_units(x.contiguous())
            x = x + x0
        x0 = x
        x = self.attns(self.layer_norms(x),x_v)
        x = self.layer_scales(x)+x0
        x1 = x
        x = self.mlps(x)+x1

        return x

class branch(nn.Module):
    def __init__(self,dim_in=256, dim_embed=256,  
                 fmap_size=[28,28], window_size=7, 
                 expansion=2,heads=8, drop=0.1):

        super().__init__()
        self.depths = 2
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()
        self.use_lpu = True
        self.local_perception_units = nn.ModuleList(
            [nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed) 
             for _ in range(self.depths)]
        )
        self.layer_norms = nn.ModuleList([LayerNormProxy(dim_embed) 
                                          for _ in range(self.depths)])
        self.attns = nn.ModuleList([
            LocalAttention(dim_embed, heads, window_size, drop),
            ShiftWindowAttention(dim_embed, heads, window_size, drop,window_size // 2, fmap_size)
            ])
        
        self.layer_scales = nn.ModuleList([LayerScale(dim_embed, init_values=1e-4) 
                                           for _ in range(self.depths)])
        self.mlps = nn.ModuleList([MLPBlock(dim_embed, expansion, drop) 
                                   for _ in range(self.depths)])
        
    def forward(self, x):
        x = self.proj(x)

        for d in range(self.depths):
            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x)
                x = x + x0
            x0 = x
            x = self.attns[d](self.layer_norms[d](x))
            x = self.layer_scales[d](x)+x0
            x1 = x
            x = self.mlps[d](x)+x1

        return x
