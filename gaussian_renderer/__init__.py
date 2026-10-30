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
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_medium2 import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

# 核心渲染函数
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, medium_rgb : torch.Tensor, medium_bs : torch.Tensor,
           scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建一个和 pc.get_xyz 形状相同的零张量
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration 用于透视投影
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # 计算水平半视场角的正切值
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # 计算垂直半视场角的正切值

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,  # 世界坐标到相机/视图坐标的变换矩阵
        projmatrix=viewpoint_camera.full_proj_transform,   # 完整投影矩阵
        sh_degree=pc.active_sh_degree,                     # 当前启用的球谐函数阶数
        campos=viewpoint_camera.camera_center,             # 相机中心在世界坐标中的位置
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)  # 构建渲染函数

    means3D = pc.get_xyz   # 获取所有 Gaussian 的 3D 中心位置
    means2D = screenspace_points
    opacity = pc.get_opacity  # 获取所有 Gaussian 的 不透明度
    medium_attn = pc.get_medium

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    # 是否在python中就计算出高斯的协方差矩阵
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 知否在python中就预先计算出每个高斯在当前视角下的颜色
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))  # 每个高斯中心到相机中心的向量
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)  # 对方向向量进行单位化
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).  开始渲染
    if separate_sh:
        # rendered_image: 渲染图像，通常 [3,H,W]
        # radii: 每个 Gaussian 的屏幕空间半径，通常 [N]
        # depth_image: 渲染得到的深度图
        rendered_image, out_color_medium, out_medium, radii, depth_image, gaussian_water_mask, gaussian_blurry_mask, gaussian_nearest_nonwater_xy, gaussian_nonwater_count = rasterizer(
            means3D = means3D,
            means2D = means2D,
            medium_rgb=medium_rgb,
            medium_bs=medium_bs,
            medium_attn=medium_attn,
            water_mask=viewpoint_camera.mask,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, out_color_medium, out_medium, radii, depth_image, gaussian_water_mask, gaussian_blurry_mask, gaussian_nearest_nonwater_xy, gaussian_nonwater_count = rasterizer(
            means3D = means3D,
            means2D = means2D,
            medium_rgb=medium_rgb,
            medium_bs=medium_bs,
            medium_attn=medium_attn,
            water_mask=viewpoint_camera.mask,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only) 对渲染图像进行曝光矫正
    if use_trained_exp:
        # 对渲染图应用曝光校正
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out_color_medium = out_color_medium.clamp(0, 1)
    out_medium = out_medium.clamp(0, 1)
    out = {
        "render": rendered_image,  # 渲染得到的 RGB 图像
        "render_medium": out_color_medium,
        "medium": out_medium,
        "viewspace_points": screenspace_points,  
        "visibility_filter" : (radii > 0).nonzero(),  # 可见 Gaussian 的索引
        "radii": radii,  # 每个 Gaussian 在屏幕空间的半径
        "depth" : depth_image,  # 渲染得到的深度图
        "gaussian_water_mask": gaussian_water_mask,
        "gaussian_blurry_mask": gaussian_blurry_mask,
        "gaussian_dg": gaussian_nearest_nonwater_xy,
        "gaussian_nonwater_ratio": gaussian_nonwater_count
        }
    
    return out
