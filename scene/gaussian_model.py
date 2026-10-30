#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, fov2focal
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F
from scene.model import SHEncoding, MLP

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:
    # 定义一个 GaussianModel 类，用来管理 3D Gaussian Splatting 中所有高斯点的参数。
    # 每个高斯点通常包含：位置 xyz、颜色特征 SH、尺度 scaling、旋转 rotation、不透明度 opacity 等。

    def setup_functions(self):
        # 设置激活函数

        # 通过缩放和旋转矩阵计算每个3D高斯的协方差矩阵
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):

            # 这里先获得RS 需要将缩放向量和旋转四元数都转换成矩阵
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)

            # 计算协方差矩阵 C=RS @ SR
            actual_covariance = L @ L.transpose(1, 2)

            # 由于该矩阵是关于对角线对称的 因此只需要6个参数就可以表达协方差矩阵
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp  # 用来将缩放限制为大于0的数
        self.scaling_inverse_activation = torch.log  # 知道尺度后可以通过该函数获得原始参数

        self.covariance_activation = build_covariance_from_scaling_rotation  # 通过缩放和旋转计算高斯的协方差矩阵

        self.opacity_activation = torch.sigmoid  # 将不透明度限制在0到1内
        self.inverse_opacity_activation = inverse_sigmoid  # 知道不透明度之后通过该函数计算原始参数 _opacity = log(opacity / (1 - opacity))

        self.rotation_activation = torch.nn.functional.normalize  # 旋转是通过四元数表示的，因此需要满足向量模长为1

        self.medium_activation = torch.nn.functional.softplus

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0  # 当前球谐函数的阶数，随着训练不断增大
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  # 球谐函数的最大阶数 越大参数越多 3阶则有48个参数来表达颜色
        self._xyz = torch.empty(0)   # 初始化高斯的中心位置 即高斯的均值向量
        self._features_dc = torch.empty(0)  # 球谐函数的0阶项
        self._features_rest = torch.empty(0)  # 球谐函数的高阶项
        self._scaling = torch.empty(0)  # 高斯的尺度参数，未经过激活的原始参数
        self._rotation = torch.empty(0)  # 高斯的旋转参数，未经normalize的四元数
        self._opacity = torch.empty(0)   # 高斯的不透明度参数，未经sigmoid的原始参数
        self._medium = torch.empty(0)   # 高斯的前向衰减系数
        self._medium_attn = torch.empty(0)

        self._water_state = torch.empty(0)   # 用于保存每个高斯是否属于水体区域
        self._water_votes_0 = torch.empty(0) # 被观测为非水体的次数
        self._water_votes_1 = torch.empty(0) # 被观测为水体的次数
        self._water_votes_2 = torch.empty(0) # 被观测为混合的次数
        self._last_mixed_view_idx = torch.empty(0)  # 每个 Gaussian 最近一次被观测为 mixed 的视角 id

        self._nonwater_ratio = torch.empty(0)
        self._dg = torch.empty(0)

        self.max_radii2D = torch.empty(0)  # 保存每个 Gaussian 在屏幕空间中的最大投影半径
        self.xyz_gradient_accum = torch.empty(0)  # 位置梯度累积量
        self.denom = torch.empty(0)  # 梯度累积的分母
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.direction_encoding = SHEncoding(levels=4, implementation="tcnn")
        self.medium_mlp = MLP(
            in_dim=self.direction_encoding.get_out_dim(),
            num_layers=2,
            layer_width=128,
            out_dim=6,
            activation=nn.Sigmoid(),
            out_activation=None,
            implementation="torch")
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()
        self.setup_functions()   # 各种参数的激活函数和反激活函数
        self.tmp_radii = None

    def capture(self):
        # 返回模型状态
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._medium,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        # 加载模型状态
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._medium,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def get_biexponential_params(self):
        raw = self._medium_attn
        raw_a = raw[:, 0:3]
        raw_b = raw[:, 3:6]
        raw_c = raw[:, 6:9]
        raw_d = raw[:, 9:12]

        a = F.softplus(raw_a)
        b = -(F.softplus(raw_b) + 1e-6)
        c = F.softplus(raw_c)
        d = -(F.softplus(raw_d) + 1e-6)

        return a, b, c, d

    @property  # 用于像访问变量一样访问函数
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_medium(self):
        return self._medium
    
    @property
    def get_exposure(self):
        return self._exposure  # 曝光参数

    @torch.no_grad()
    def update_water_votes(self, gaussian_water_mask, view_idx=None):
        gaussian_water_mask = gaussian_water_mask.detach().to(self._xyz.device).view(-1)

        visible = gaussian_water_mask >= 0

        # 当前视角有效观测投票
        self._water_votes_0 += ((gaussian_water_mask == 0) & visible).float()  # 完全非水体
        self._water_votes_1 += ((gaussian_water_mask == 1) & visible).float()  # 完全水体
        self._water_votes_2 += ((gaussian_water_mask == 2) & visible).float()  # 混合

        # self._last_mixed_view_idx[gaussian_water_mask == 2] = int(view_idx)  # 记录最近被记为混合部分高斯的视角id

    @torch.no_grad()
    def update_water_state_from_view(self, water_threshold=0.5, nonwater_threshold=0.7):
        total_votes = self._water_votes_0 + self._water_votes_1 + self._water_votes_2

        # 计算比例
        water_ratio = self._water_votes_1 / total_votes.clamp_min(1.0)
        nonwater_ratio = self._water_votes_0 / total_votes.clamp_min(1.0)
        mixed_ratio = self._water_votes_2 / total_votes.clamp_min(1.0)

        new_state = self._water_state.clone()
        new_state[(water_ratio >= water_threshold)] = 1
        new_state[(nonwater_ratio >= nonwater_threshold)] = 0
        new_state[(mixed_ratio >= 0.5)] = 2

        self._water_state = new_state

    @torch.no_grad()
    def update_ratio_dg(self, gaussian_nonwater_ratio, gaussian_dg, gaussian_water_mask):
        P = self._xyz.shape[0]
        device = self._xyz.device
        ratio = gaussian_nonwater_ratio.detach().to(device=device,dtype=torch.float32)
        dg = gaussian_dg.detach().to(device=device,dtype=torch.float32)

        water_state = gaussian_water_mask.detach().to(device=device)

        valid = water_state == 2

        valid &= torch.isfinite(ratio)
        valid &= (ratio >= 0.0) & (ratio <= 1.0)

        valid &= torch.isfinite(dg).all(dim=-1)

        updated = valid & (ratio < self._nonwater_ratio)

        self._nonwater_ratio[updated] = ratio[updated]
        self._dg[updated] = dg[updated]

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    # 获取每个高斯的协方差
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    # 增加球谐函数的阶数
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def inverse_softplus(self, x):
        return torch.log(torch.exp(x) - 1.0)

    # 根据SFM给出的初始化点云创建高斯模型
    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()  # 将点云转换成tensor并放到GPU上
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())  # 将点云的RGB颜色转换成球谐函数第0阶的表达形式，并放到GPU上
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()  # 定义球谐函数的参数
        features[:, :3, 0 ] = fused_color  # 球谐函数第0阶的参数
        features[:, 3:, 1:] = 0.0  # 剩下的设置为0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])   # 打印出用于初始化的点云的数量

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)  # 计算每个点到其邻近点的平均距离
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)   # 用距离来初始化每个高斯的尺度 由于优化的尺度是经过exp的 所以这里的尺度要经过log得到原始参数
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")  # 初始化每个高斯的旋转四元数
        rots[:, 0] = 1

        # 初始化每个高斯的不透明度为 0.1 同理原始参数是通过sigmoid得到的，因此需要通过反激活函数处理
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # mediums = self.inverse_softplus(0.01 * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda"))
        # mediums_attn = self.inverse_softplus(0.01 * torch.ones((fused_point_cloud.shape[0], 12), dtype=torch.float, device="cuda"))

        init_a = self.inverse_softplus(0.005 * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda"))
        init_b = self.inverse_softplus(0.01 * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda"))
        init_c = self.inverse_softplus(0.005 * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda"))
        init_d = self.inverse_softplus(0.001 * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda"))
        mediums_attn = torch.cat([init_a, init_b, init_c, init_d], dim=1)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))  # 初始化高斯的位置
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))  # 初始化球谐函数第零阶参数
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))  # 初始化球谐函数剩余阶参数
        self._scaling = nn.Parameter(scales.requires_grad_(True))   # 初始化高斯的尺度
        self._rotation = nn.Parameter(rots.requires_grad_(True))    # 初始化高斯的旋转
        self._opacity = nn.Parameter(opacities.requires_grad_(True))  # 初始化高斯的不透明度

        self._water_state = torch.zeros((self.get_xyz.shape[0],), dtype=torch.float, device="cuda")
        self._water_votes_0 = torch.zeros((self.get_xyz.shape[0],), dtype=torch.float, device="cuda")
        self._water_votes_1 = torch.zeros((self.get_xyz.shape[0],), dtype=torch.float, device="cuda")
        self._water_votes_2 = torch.zeros((self.get_xyz.shape[0],), dtype=torch.float, device="cuda")
        # self._last_mixed_view_idx = torch.full((self.get_xyz.shape[0],),-1,dtype=torch.long,device="cuda")

        self._nonwater_ratio = torch.full((self.get_xyz.shape[0],), float("inf"), dtype=torch.float32, device="cuda")
        self._dg = torch.full((self.get_xyz.shape[0], 3), float("nan"), dtype=torch.float32, device="cuda")

        # self._medium = nn.Parameter(mediums.requires_grad_(True))
        self._medium_attn = nn.Parameter(mediums_attn.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # 初始化高斯投影到2维平面的最大半径
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}  # 建立 image_name 到相机索引的映射
        self.pretrained_exposures = None  # 初始化预训练曝光参数为空
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)  # 创建每个相机一组 3x4 的曝光矩阵
        self._exposure = nn.Parameter(exposure.requires_grad_(True))  # 将曝光矩阵设置为可优化参数

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense  # 保存增密阈值比例，用于判断一个高斯是应该 clone 还是 split
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # 初始化 xyz 梯度累积器 用于判断是否需要增密
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # 初始化计数分母 用于计算平均梯度

        self.direction_encoding.cuda()
        self.medium_mlp.cuda().train()

        # 设置可学习参数，以及他们的学习率和名字
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._medium_attn], 'lr': 1e-4, "name": "medium_attn"},
            # {'params': [self._medium], 'lr': training_args.opacity_lr, "name": "medium"},
        ]

        if self.optimizer_type == "default":
            # 设置Adam优化器
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        # 为曝光参数单独创建一个 Adam 优化器
        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.medium_optimizer = torch.optim.Adam(self.medium_mlp.parameters(), lr=1e-3, eps=1e-15)

        # 为高斯的中心位置（也是均值向量）设置衰减学习率策略
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        # 为曝光参数设置衰减学习率策略
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

        self.medium_mlp_scheduler_args = get_expon_lr_func(lr_init=1e-3, lr_final=1.5e-4,
                                                           lr_delay_mult=0.01,
                                                           max_steps=2000)
    
    # 用定义的学习率衰减策略来更新学习率
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.medium_optimizer.param_groups:
            param_group['lr'] = self.medium_mlp_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # 构造保存 PLY 文件时，每个 vertex 需要包含的属性名列表
    def construct_list_of_attributes(self):
        # nx ny nz 为法线 这里会设置为0
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._medium.shape[1]):
            l.append('medium_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    # 将当前 Gaussian 模型保存为 PLY 文件
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))  # 确保目标文件夹存在，如果不存在则创建

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        medium = self._medium.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]  # 'f4' 表示 float32

        elements = np.empty(xyz.shape[0], dtype=dtype_full)  # 每一个点记录高斯的所有属性
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, medium, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')  # 将结构化数组描述为 PLY 的 vertex 元素
        PlyData([el]).write(path)  # 写入 PLY 文件

    # 重置不透明度，使过高的不透明度被压低
    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.05))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_medium(self):
        medium_new = self.inverse_softplus(torch.min(self.get_medium, torch.ones_like(self.get_medium)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(medium_new, "medium")
        self._medium = optimizable_tensors["medium"]
    
    # 从 PLY 文件加载 Gaussian 模型参数
    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        medium_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("medium_")]
        medium_names = sorted(medium_names, key=lambda x: int(x.split('_')[-1]))

        if len(medium_names) > 0:
            mediums = np.zeros((xyz.shape[0], len(medium_names)))
            for idx, attr_name in enumerate(medium_names):
                mediums[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            # 兼容旧 PLY：没有 medium 字段时初始化
            mediums = np.full((xyz.shape[0], 3), -4.6, dtype=np.float32)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._medium = nn.Parameter(torch.tensor(mediums, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # 用新的 tensor 替换优化器中指定 name 的参数
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}  # 返回替换后的优化器中需要优化的参数
        for group in self.optimizer.param_groups:
            # 寻找与name同名的可学习参数
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)  # 获取该参数在优化器中的状态
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]  # 删除旧参数对应的优化器状态
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]  # 保存替换后的参数
        return optimizable_tensors
    
    # 根据 mask 裁剪优化器中的所有参数
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    # 对现有的高斯进行剪裁
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._medium_attn = optimizable_tensors["medium_attn"]
        # self._medium = optimizable_tensors["medium"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._water_state = self._water_state[valid_points_mask]
        self._water_votes_0 = self._water_votes_0[valid_points_mask]
        self._water_votes_1 = self._water_votes_1[valid_points_mask]
        self._water_votes_2 = self._water_votes_2[valid_points_mask]
        # self._last_mixed_view_idx = self._last_mixed_view_idx[valid_points_mask]

        self._nonwater_ratio = (self._nonwater_ratio[valid_points_mask])
        self._dg = (self._dg[valid_points_mask])

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        '''if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]'''
    
    # 将新增点的参数拼接到优化器已有参数后面
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]  # 找出与当前参数的名字相同的新增参数
            stored_state = self.optimizer.state.get(group['params'][0], None)  # 获取当前参数的优化器状态
            if stored_state is not None:

                # 为新增参数添加全为0的优化器状态
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                # 将新增参数放到原有参数后面加入优化器中
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    # 将新生成的高斯点参数加入模型和优化器中
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_medium_attn, new_scaling, new_rotation, 
                              new_water_state, new_water_votes_0, new_water_votes_1, new_water_votes_2, new_ratio, new_dg):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "medium_attn": new_medium_attn,
        # "medium": new_medium,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # 将新增高斯包含的参数加入优化器
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # 将新增的高斯加入模型
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._medium_attn = optimizable_tensors["medium_attn"]
        # self._medium = optimizable_tensors["medium"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))

        self._water_state = torch.cat((self._water_state, new_water_state), dim=0)
        self._water_votes_0 = torch.cat((self._water_votes_0, new_water_votes_0), dim=0)
        self._water_votes_1 = torch.cat((self._water_votes_1, new_water_votes_1), dim=0)
        self._water_votes_2 = torch.cat((self._water_votes_2, new_water_votes_2), dim=0)
        # self._last_mixed_view_idx = torch.cat((self._last_mixed_view_idx, new_last_mixed_view_idx), dim=0)

        self._nonwater_ratio = torch.cat((self._nonwater_ratio, new_ratio), dim=0)
        self._dg = torch.cat((self._dg, new_dg), dim=0)

        # 新增高斯后充值所有高斯的梯度累积器，梯度计数器，屏幕空间最大半径记录
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    # 分裂大的高斯
    def densify_and_split(self, grads, grad_threshold, scene_extent, extra_mask, N=2):
        # grads: 每个点的梯度指标
        # grad_threshold: 梯度阈值
        # scene_extent: 场景尺度
        # N: 每个选中点分裂成几个新点，默认 2
        n_init_points = self.get_xyz.shape[0]  # 高斯点的数量
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)  # 选择梯度大的高斯
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)  # 进一步选择梯度大且尺度大的高斯

        water_densify_mask = self._water_state != 1
        selected_pts_mask = torch.logical_and(selected_pts_mask, water_densify_mask)

        '''padded_extra_mask = torch.zeros((n_init_points), device="cuda")
        extra_mask = torch.tensor(extra_mask > 0.5, device="cuda")
        padded_extra_mask[:grads.shape[0]] = extra_mask.squeeze()
        selected_pts_mask = torch.logical_or(selected_pts_mask, padded_extra_mask)'''

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)  # 采样新点的标准差为要分裂的高斯点的尺度
        means =torch.zeros((stds.size(0), 3),device="cuda")   # 采样新点的均值为0
        samples = torch.normal(mean=means, std=stds)   # 用正态分布采样新的高斯点
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)  # 将被选中高斯的四元数转换成旋转矩阵
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)  # 新的高斯中心

        split_water_state = self._water_state[selected_pts_mask].repeat(N)
        split_ratio = self._nonwater_ratio[selected_pts_mask].repeat(N)
        split_dg = self._dg[selected_pts_mask].repeat(N, 1)
        split_dg_norm_sq = (split_dg * split_dg).sum(dim=1)

        guided_mask = (
            (split_water_state == 2)
            & torch.isfinite(split_ratio)
            & (split_ratio >= 0.0)
            & (split_ratio <= 1.0)
            & torch.isfinite(split_dg).all(dim=1)
            & torch.isfinite(split_dg_norm_sq)
            & (split_dg_norm_sq > 1e-12)
        )

        random_offset = (new_xyz[guided_mask] - self.get_xyz[selected_pts_mask].repeat(N, 1)[guided_mask])

        parallel_offset = ((random_offset * split_dg[guided_mask]).sum(dim=1, keepdim=True)/ split_dg_norm_sq[guided_mask].unsqueeze(1)) * split_dg[guided_mask]
        perpendicular_position = (new_xyz[guided_mask] - parallel_offset)
        guided_xyz = (split_ratio[guided_mask].unsqueeze(1) * parallel_offset + perpendicular_position + (1.0 - split_ratio[guided_mask].unsqueeze(1)) * split_dg[guided_mask])
        new_xyz[guided_mask] = torch.where(torch.isfinite(guided_xyz).all(dim=1, keepdim=True), guided_xyz, new_xyz[guided_mask])

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))  # 新高斯的尺度要比原高斯更小
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)  # 新高斯继承原高斯的旋转参数
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)  # 新高斯继承原高斯的 DC 颜色特征
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)   # 新高斯继承原高斯的高阶 SH 特征
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)  # 新高斯继承原高斯的不透明度参数
        new_medium_attn = self._medium_attn[selected_pts_mask].repeat(N, 1)
        # new_medium = self._medium[selected_pts_mask].repeat(N,1)
        # new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)  # 新高斯继承临时半径

        new_water_state = self._water_state[selected_pts_mask].repeat(N)
        new_water_votes_0 = torch.zeros((new_water_state.shape[0],), dtype=self._water_votes_0.dtype,device=self._water_votes_0.device)
        new_water_votes_1 = torch.zeros((new_water_state.shape[0],), dtype=self._water_votes_1.dtype,device=self._water_votes_1.device)
        new_water_votes_2 = torch.zeros((new_water_state.shape[0],), dtype=self._water_votes_2.dtype,device=self._water_votes_2.device)

        new_dg = self._dg[selected_pts_mask].repeat(N,1)
        new_ratio= self._nonwater_ratio[selected_pts_mask].repeat(N)

        # 将新增的高斯加入优化器和模型中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_medium_attn, new_scaling, new_rotation, 
                                   new_water_state, new_water_votes_0, new_water_votes_1, new_water_votes_2, new_ratio, new_dg)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # 删除要被分裂的大的高斯
        self.prune_points(prune_filter)
    
    # 对满足条件的小高斯进行克隆
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)  # 选择梯度范数大于等于阈值的高斯
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)  # 进一步筛选尺度较小的高斯

        water_densify_mask = self._water_state != 1
        selected_pts_mask = torch.logical_and(selected_pts_mask, water_densify_mask)
        
        # 复制这些高斯的参数
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_medium_attn = self._medium_attn[selected_pts_mask]
        # new_medium = self._medium[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_water_state = self._water_state[selected_pts_mask]

        new_water_votes_0 = torch.zeros((new_water_state.shape[0],), dtype=self._water_votes_0.dtype,device=self._water_votes_0.device)
        new_water_votes_1 = torch.zeros((new_water_state.shape[0],), dtype=self._water_votes_1.dtype,device=self._water_votes_1.device)
        new_water_votes_2 = torch.zeros((new_water_state.shape[0],), dtype=self._water_votes_2.dtype,device=self._water_votes_2.device)

        new_dg = self._dg[selected_pts_mask]
        new_ratio= self._nonwater_ratio[selected_pts_mask]
        # 将这些高斯加入优化器和模型
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_medium_attn, new_scaling, new_rotation, 
                                   new_water_state, new_water_votes_0, new_water_votes_1, new_water_votes_2, new_ratio, new_dg)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, extra_mask):

        total_votes = self._water_votes_0 + self._water_votes_1 + self._water_votes_2
        water_ratio = self._water_votes_1 / total_votes.clamp_min(1.0)
        selected_pts_mask = torch.logical_or(water_ratio>0.2, total_votes<1)
        self.prune_points(selected_pts_mask)

        grads = self.xyz_gradient_accum / self.denom  # 计算每个高斯的平均梯度
        grads[grads.isnan()] = 0.0  # 如果出现 NaN，替换为 0

        #self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)  # 进行克隆
        self.densify_and_split(grads, max_grad, extent, extra_mask)  # 进行分裂

        prune_mask = (self.get_opacity < min_opacity).squeeze()  # opacity 小于阈值的点会被删除
        if max_screen_size:  # 如果设置了屏幕空间最大尺寸限制
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  # 尺寸过大的也要删除
            # 只要满足以下任一条件，就删除：
            # 1. opacity 太低
            # 2. 屏幕空间太大
            # 3. 世界空间尺度太大
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)  # 进行剪裁
        #tmp_radii = self.tmp_radii
        #self.tmp_radii = None

        self._nonwater_ratio.fill_(float("inf"))
        self._dg.fill_(float("nan"))

        self._water_votes_0.zero_()
        self._water_votes_1.zero_()
        self._water_votes_2.zero_()

        torch.cuda.empty_cache() # 清理 CUDA 缓存，释放未使用显存

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 计算梯度
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # 根据当前相机中心到每个像素的射线预测medium相关参数
    def get_medium_outputs_from_camera(self, camera):

        # 读取图像的宽和高
        W = int(camera.image_width)
        H = int(camera.image_height)
        # 根据视场角和分辨率计算焦距
        fx = fov2focal(camera.FoVx, W)
        fy = fov2focal(camera.FoVy, H)

        # 默认主点在图像中心
        cx = 0.5 * W
        cy = 0.5 * H

        R = camera.world_view_transform[:3, :3]

        # 计算每个主点到每个像素的方向
        y = torch.arange(H).cuda()  # + 0.5
        x = torch.arange(W).cuda()  # + 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # 像素坐标转归一化相机平面坐标
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        directions = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        norms = torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(1e-8)

        # 转到世界坐标系下
        directions = directions / norms
        directions = directions @ R.T

        directions_flat = directions.view(-1, 3)
        directions_encoded = self.direction_encoding(directions_flat)
        outputs_shape = directions.shape[:-1]

        medium_base_out = self.medium_mlp(directions_encoded)

        medium_rgb = self.colour_activation(medium_base_out[..., :3]).view(*outputs_shape, -1)

        medium_bs = self.sigma_activation(medium_base_out[..., 3:6]).view(*outputs_shape, -1)

        return medium_rgb, medium_bs

    def world_to_camera_points(self, camera):
        points_world = self.get_xyz.detach()
        device = points_world.device
        dtype = points_world.dtype

        ones = torch.ones(
        (points_world.shape[0], 1),
        device=device,
        dtype=dtype
        )
        points_world_h = torch.cat([points_world, ones], dim=-1)
        world_view_transform = camera.world_view_transform
        points_camera_h = points_world_h @ world_view_transform
        z = points_camera_h[..., 2:3]
        a, b, c, d = self.get_biexponential_params()
        self._medium = a * torch.exp(b * z) + c * torch.exp(d * z)
