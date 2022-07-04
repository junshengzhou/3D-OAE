
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import MODELS
from models.util_models import PointNetFeaturePropagation, Group, Encoder

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list

class DGCNN_Propagation(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1024, 384, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 384),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q


@MODELS.register_module()
class PointTransformer_seg(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propagation_2 = PointNetFeaturePropagation(in_channel= self.trans_dim+3, mlp = [self.trans_dim * 4, self.trans_dim]) 
        self.propagation_1= PointNetFeaturePropagation(in_channel= self.trans_dim+3, mlp = [self.trans_dim * 4, self.trans_dim])
        self.propagation_0 = PointNetFeaturePropagation(in_channel= self.trans_dim+3+16, mlp = [self.trans_dim * 4, self.trans_dim]) 
        self.dgcnn_pro_1 = DGCNN_Propagation(k = self.config.knn)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = self.config.knn)

        self.conv1 = nn.Conv1d(self.trans_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.cls_dim, 1)

        self.build_loss_func()
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def get_loss_seg(self, pred, target):
        pred = pred.contiguous().view(-1, 50)
        target = target.view(-1, 1)[:,0]
        total_loss = F.nll_loss(pred, target)

        return total_loss

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

        print(f'[PointTransformer] Successful Loading the ckpt from {bert_ckpt_path}')


    def forward(self, pts, cls_label):
        B,N,C = pts.shape
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = group_input_tokens
        # transformer
        feature_list = self.blocks(x, pos)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]

        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)    # B, num_class, N

        center_level_0 = pts.transpose(-1, -2).contiguous()                      # 2048 
        f_level_0 = torch.cat([cls_label_one_hot, center_level_0], 1)

        center_level_1 = fps(pts, 512).transpose(-1, -2).contiguous()            # 512
        f_level_1 = center_level_1
        center_level_2 = fps(pts, 256).transpose(-1, -2).contiguous()            # 256
        f_level_2 = center_level_2
        center_level_3 = center.transpose(-1, -2).contiguous()                   # 128

        # init the feature by 3nn propagation
        f_level_3 = feature_list[2]    
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1])  # B, C, 256
        f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list[0])  # B, C, 512
    

        # bottom up
        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)     # B, C, 256
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)     # B, C, 512
        f_level_0 =  self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

        # FC layers
        feat =  F.relu(self.bn1(self.conv1(f_level_0)))
        x = self.drop1(feat)
        x = self.conv2(x)   # B, cls_num, N
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)  # B, N, cls_num
        return x, f_level_3
        

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss