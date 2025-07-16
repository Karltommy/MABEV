

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

@ATTENTION.register_module()
class DeltaFeatureAttention(BaseModule):
    def __init__(self,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 embed_dims = 256,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None
                 ):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.dropout = nn.Dropout(dropout)
        self.delta_mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims)
        )

        self.batch_first = batch_first
        self.norm_cfg = norm_cfg

        self.im2col_step = im2col_step
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims , num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                            num_heads * num_levels * num_points)
        # self.prev_project = nn.Linear(embed_dims*2, embed_dims)
        self.cur_project = nn.Linear(embed_dims*2, embed_dims)
        self.output_project = nn.Linear(embed_dims, embed_dims)
        self.gate_proj = nn.Linear(embed_dims * 2, embed_dims)



    def get_ref_points(self, H, W, bs, device='cuda', dtype=torch.float):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d



    def forward(self,
                prev_query,
                query,
                identity=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs
                ):
        if prev_query is None:
            return query
        if identity is None:
            identity = query
        # if prev_query.dim != query.dim:
        #     return query
        bs, num_query, embed_dims = query.shape
        delta_query = query - prev_query
        # delta_query = delta_query / math.sqrt(embed_dims)
        delta_feat = self.delta_mlp(delta_query)
        # prev_delta_query = torch.cat([prev_query, delta_feat], dim=-1)
        cur_delta_query = torch.cat([query, delta_query], dim=-1)
        # prev_feat = self.prev_project(prev_delta_query)
        # gate = torch.sigmoid(self.cur_project(cur_delta_query))
        cur_feat = self.cur_project(cur_delta_query)
        # mix_delta_query = torch.cat([prev_feat, cur_feat], dim=-1)
        # value = self.mix_project(mix_delta_query)
        # value = query
        value = cur_feat
        _, num_value, _ = value.shape
        value = value.reshape(bs ,
                              num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(delta_feat)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2).contiguous()
        attention_weights = self.attention_weights(delta_feat).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points).contiguous()

        # attention_weights = attention_weights.permute(0, 1, 2, 3, 4) \
        #     .reshape(bs * 1, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        # sampling_offsets = sampling_offsets.permute(0, 1, 2, 3, 4, 5) \
        #     .reshape(bs * 1, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        bev_h = spatial_shapes[0, 1].item()
        bev_w = spatial_shapes[0, 0].item()
        reference_points = self.get_ref_points(bev_h, bev_w, bs)

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
                             + sampling_offsets \
                             / offset_normalizer[None, None, None, :, None, :]

        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:

            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        motion_feat = output
        # cur_delta_query = torch.cat([query, motion_feat], dim=-1)
        # output = self.cur_project(cur_delta_query).permute(1, 2, 0)
        gate = torch.sigmoid(self.gate_proj(torch.cat([query, motion_feat], dim=-1)))
        final_output = gate * query + (1 - gate) * motion_feat
        output = final_output.permute(1, 2, 0)


        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        output = output.view(num_query, embed_dims, bs)
        # output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_project(output)


        return self.dropout(output) + identity



