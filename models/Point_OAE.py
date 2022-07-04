from matplotlib.transforms import Transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_

from .util_models import Group
from .util_models import Encoder

from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
import numpy as np

from torchvision import transforms
from datasets import data_transforms
train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

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
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

class TransformerDecoder(nn.Module):
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

    def forward(self, x, return_token_num=0):
        for _, block in enumerate(self.blocks):
            x = block(x)
        
        if return_token_num > 0:
            x = x[:, -return_token_num:]
        return x

@MODELS.register_module()
class Point_OAE(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.out_dim = config.transformer_config.group_size * 3
        self.depth_encoder = config.transformer_config.depth_encoder
        self.depth_decoder = config.transformer_config.depth_decoder
        self.drop_path_rate = config.transformer_config.drop_path_rate 
        self.cls_dim = config.transformer_config.cls_dim 
        self.num_heads = config.transformer_config.num_heads 
        self.group_size = config.transformer_config.group_size
        self.num_group = config.transformer_config.num_group
        self.encoder_dims =  config.transformer_config.encoder_dims
        print_log(f'[Transformer args] {config.transformer_config}', logger = 'Occlusion Auto-Encoder')

        # define the encoder
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        try:
            self.mask_rand = config.mask_rand
        except:
            self.mask_rand = False
        
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # define the learnable tokens
        self.de_mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # pos embedding for each patch 
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        # define the transformer blocks
        dpr_encoder = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth_encoder)]
        dpr_decoder = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth_decoder)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth_encoder,
            drop_path_rate = dpr_encoder,
            num_heads = self.num_heads
        )
        self.encoder_to_decoder = nn.Linear(self.trans_dim, self.out_dim, bias=False)
        self.decoder = TransformerDecoder(
            embed_dim = self.out_dim,
            depth = self.depth_decoder,
            drop_path_rate = dpr_decoder,
            num_heads = self.num_heads
        )
        # layer norm
        self.norm = nn.LayerNorm(self.trans_dim)

        # initialize the learnable tokens
        trunc_normal_(self.de_mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0) # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p =2 ,dim = -1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0] # G
            ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
            mask_num = int(ratio * len(idx))

            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        
        bool_masked_pos = torch.stack(mask_idx).to(center.device) # B G

        return bool_masked_pos

    def _mask_center_all_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        bool_masked_pos = []
        for i in range(center.shape[0]):
            ratio = self.mask_ratio[0]
            num_masks = int(self.num_group * ratio)
            mask = np.hstack([
                np.zeros(self.num_group - num_masks),
                np.ones(num_masks),
            ])
            mask = torch.FloatTensor(mask)
            idx = torch.randperm(mask.shape[0])
            mask = mask[idx].view(mask.size()).bool()
            bool_masked_pos.append(mask)
        bool_masked_pos = torch.cat(bool_masked_pos, dim=0).reshape(center.shape[0], -1).bool().to(center.device)

        return bool_masked_pos

    def _mixup_pc(self, neighborhood, center):
        '''
            neighborhood : B G M 3
            center: B G 3
            ----------------------
            mixup_ratio: /alpha:
                mixup_label = alpha * origin + (1 - alpha) * flip

        '''
        mixup_ratio = torch.rand(neighborhood.size(0))
        mixup_mask = torch.rand(neighborhood.shape[:2]) < mixup_ratio.unsqueeze(-1)
        mixup_mask = mixup_mask.type_as(neighborhood)
        mixup_neighborhood = neighborhood * mixup_mask.unsqueeze(-1).unsqueeze(-1) + neighborhood.flip(0) * (1 - mixup_mask.unsqueeze(-1).unsqueeze(-1))
        mixup_center = center * mixup_mask.unsqueeze(-1) + center.flip(0) * (1 - mixup_mask.unsqueeze(-1))

        return mixup_ratio.to(neighborhood.device), mixup_neighborhood, mixup_center

    def forward_eval(self, pts):
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            # encoder the input cloud blocks
            group_input_tokens = self.encoder(neighborhood)  #  B G N
            group_input_tokens = self.reduce_dim(group_input_tokens)

            # add pos embedding
            pos = self.pos_embed(center)
            # transformer
            x = self.blocks(group_input_tokens, pos)
            x = self.norm(x.mean(1))

            return x
            
    def forward(self, pts, noaug=False, cutmix=False):
        if noaug:
            return self.forward_eval(pts)
            
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        
        if cutmix:
            _, neighborhood, center = self._mixup_pc(neighborhood, center)

        B, G, M, C = neighborhood.shape
        if not noaug:
            # generate mask
            if self.mask_rand:
                bool_masked_pos = self._mask_center_all_rand(center, noaug = noaug) # B G
            else:
                bool_masked_pos = self._mask_center_block(center, noaug = noaug) # B G
            bool_masked_pos = bool_masked_pos.to(center.device, non_blocking=True).flatten(1).to(torch.bool)
            center_masked = center[bool_masked_pos].reshape(B, -1, C)
            neighborhood_masked = neighborhood[bool_masked_pos].reshape(B, -1, M, C)
            center_vis = center[~bool_masked_pos].reshape(B, -1, C)
            neighborhood_vis = neighborhood[~bool_masked_pos].reshape(B, -1, M, C)
        else:
            center_masked = center
            neighborhood_masked = neighborhood

        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood_vis)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)

        # add pos embedding
        pos_vis = self.pos_embed(center_vis)
        pos_masked = self.pos_embed(center_masked)
        # transformer
        x = self.blocks(group_input_tokens, pos_vis)
        x = self.norm(x)

        feat_all = torch.cat([x+pos_vis, self.de_mask_token + pos_masked], dim=1)
        feat_all = self.encoder_to_decoder(feat_all)
        
        output = self.decoder(feat_all, center_masked.shape[1]).reshape(B, center_masked.shape[1], self.group_size, 3)
        out_points = center_masked.unsqueeze(2).repeat(1,1,self.group_size,1) + output
        out_points = out_points.reshape(B, -1, 3)

        masked_shape = (neighborhood_masked + center_masked.unsqueeze(2)).reshape(B,-1,3)
        vis_shape = (neighborhood_vis + center_vis.unsqueeze(2)).reshape(B,-1,3)

        return out_points, masked_shape, vis_shape

@MODELS.register_module()
class PointTransformer_OAE(nn.Module):
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
        self.encoder_dims =  config.encoder_dims

        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
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

        if config.use_dropout:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )
        else:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim, self.cls_dim)
            )
        self.build_loss_func()
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )
        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')

    def forward(self, pts):
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)

        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = group_input_tokens
        # transformer
        x = self.blocks(x, pos)     # B, G, C
        x = self.norm(x.mean(1))    # B, C
        x = self.cls_head_finetune(x)

        return x
