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

import os
import torch
import cv2
import time
from random import randint
from utils.loss_utils import l1_loss, ssim, depth_aware_smooth_loss, anisotropic_total_variation_loss
from gaussian_renderer import render, network_gui
import sys
from lpipsPyTorch import lpips
from torchvision.utils import save_image
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0  # 初始化起始迭代轮次为 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)  # 建立高斯模型 输入球谐函数的阶数以及优化器的种类
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)  # 设置优化器，将高斯的参数放入优化器中
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  # 设置背景颜色
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色变成tensor并放到GPU上

    iter_start = torch.cuda.Event(enable_timing = True)  # 创建 CUDA 事件，用于记录每次迭代开始时间
    iter_end = torch.cuda.Event(enable_timing = True)    # 创建 CUDA 事件，用于记录每次迭代结束时间

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE  # 判断当前是否真正使用 Sparse Adam
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()   # 获取训练相机列表，并复制一份 后续每轮随机取一个相机用于训练
    viewpoint_indices = list(range(len(viewpoint_stack)))  # 创建相机索引列表
    ema_loss_for_log = 0.0  # 初始化用于日志显示的指数滑动平均 loss
    ema_Ll1depth_for_log = 0.0  # 初始化用于日志显示的指数滑动平均深度 loss

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")   # 创建 tqdm 进度条 显示从 first_iter 到 opt.iterations 的训练进度
    first_iter += 1

    # 开始训练循环
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    medium_rgb, medium_bs = gaussians.get_medium_outputs_from_camera(custom_cam)
                    gaussians.world_to_camera_points(custom_cam)
                    # 如果 GUI 传来了自定义相机，则用当前模型渲染该视角
                    net_image = render(custom_cam, gaussians, pipe, background, medium_rgb, medium_bs,
                                       scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                    color_medium, out_medium = net_image["render_medium"], net_image["medium"]
                    image = color_medium + out_medium
                    # image = net_image["render"]
                    # memoryview 创建零拷贝视图，便于网络发送
                    net_image_bytes = memoryview((torch.clamp(image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()  # 记录本轮训练开始的 CUDA 时间点

        gaussians.update_learning_rate(iteration) # 根据当前迭代次数更新 Gaussian 模型中的学习率 包括：xyz 位置学习率和 exposure 学习率

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            # 每1000轮训练增加一次球谐函数的阶数
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            # 如果所有相机都被训练过一次则列表为空 则重新建立相机列表和相机索引
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        
        # 随机抽取一个相机用于训练
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # 如果启用随机背景，则每轮随机生成一个 RGB 背景颜色, 否则使用固定背景 background
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 使用当前相机、当前 Gaussian 模型和背景颜色进行渲染
        medium_rgb, medium_bs = gaussians.get_medium_outputs_from_camera(viewpoint_cam)
        gaussians.world_to_camera_points(viewpoint_cam)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, medium_rgb, medium_bs, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        # 从 render_pkg 中取出关键结果
        # image: 渲染图像 [3,H,W]
        # viewspace_point_tensor: 点在视图空间中的坐标张量，用于统计梯度
        # visibility_filter: 哪些 Gaussian 在当前视角可见
        # radii: 每个 Gaussian 在屏幕空间的半径
        image, color_medium, out_medium, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_medium"], render_pkg["medium"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        render_image = color_medium + out_medium
        render_image = torch.clamp(render_image, 0.0, 1.0)
        # print("mask > 1:", (render_pkg["gaussian_water_mask"] > 0.5).sum().item())

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            render_image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()  # 获取当前相机对应的真实图像，并移动到 GPU
        weight = 1.0 / (render_image.detach() + 1e-3)
        weight = torch.clamp(weight, max=20)
        Ll1 = l1_loss(weight*render_image, weight*gt_image)  # 根据渲染图像和真实图像计算L1 loss
        # 计算 SSIM loss
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(weight*render_image.unsqueeze(0), weight*gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(weight*render_image, weight*gt_image)
        # 计算总损失
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization 利用深度实现正则化
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure

            rgb_image = render_pkg["render"]
            Lsmooth = 0.01 * depth_aware_smooth_loss(
                invDepth,
                rgb_image,
                depth_mask,
                gamma=0.001
            )

            loss += Ll1depth + Lsmooth
            Ll1depth = Ll1depth.item()

            normalize_depth = lambda x: (x - x.min()) / (x.max() - x.min())
            depth = normalize_depth(render_pkg["depth"])
            tv = anisotropic_total_variation_loss(depth.squeeze())
            loss += 0.1 * tv
        else:
            Ll1depth = 0
        
        loss.backward()  # 反向传播

        iter_end.record() # 记录本轮训练结束的 CUDA 时间点

        # 下面的操作不需要梯度, 包括日志记录、保存、增密裁剪、优化器 step 等
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log  # 更新总 loss 的指数滑动平均
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log  # 更新深度 loss 的指数滑动平均

            if iteration % 10 == 0:
                # 在进度条后显示当前平滑 loss 和深度 loss
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)  # 保存当前 Gaussian 模型到指定迭代目录

            # 投票计数
            gaussians.update_water_votes(render_pkg["gaussian_water_mask"])

            # Densification 增加密度并剪裁
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # 记录当前可见点在图像空间中的最大半径，用于后续裁剪过大的点
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)  # 累积当前可见点的视图空间梯度

                next_densify_iter = (iteration // opt.densification_interval + 1) * opt.densification_interval

                # 只有下一个整点确实满足增密条件，才提前开始记录
                if (next_densify_iter > opt.densify_from_iter and next_densify_iter < opt.densify_until_iter and next_densify_iter - iteration <= 20):
                    gaussians.update_ratio_dg(render_pkg["gaussian_nonwater_ratio"], render_pkg["gaussian_dg"],render_pkg["gaussian_water_mask"])

                # 进行增密和剪裁
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.update_water_state_from_view()
                    size_threshold = 10 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, size_threshold, radii, render_pkg["gaussian_blurry_mask"])
                
                # 重置 Gaussian opacity, 防止部分点过早变得过透明或过不透明，改善训练稳定性
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    # gaussians.reset_medium()

            '''if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                gaussians.world_to_camera_points(viewpoint_cam)
                render_pkg2 = render(viewpoint_cam, gaussians, pipe, bg, medium_rgb, medium_bs, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                prune_mask = render_pkg2["gaussian_water_mask"] > 0.5
                gaussians.prune_points(prune_mask)'''

            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    for group in gaussians.medium_optimizer.param_groups:
                        for p in group["params"]:
                            if p is None:
                                continue
                            state = gaussians.medium_optimizer.state[p]
                            if "exp_avg" in state:
                                state["exp_avg"].zero_()
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"].zero_()

            # Optimizer step 更新学习率并清除梯度
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.medium_optimizer.step()
                gaussians.medium_optimizer.zero_grad(set_to_none=True)

                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
            
            # 保存高斯参数和训练状态以及迭代轮次，用于恢复训练
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # gaussians.capture() 捕获模型参数和训练状态
                # iteration 保存当前迭代轮次
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)  # 记录当前迭代耗时

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0    # 初始化的累计 L1 损失
                psnr_test = 0.0  # 初始化的累计 PSNR 指标
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    pipe, background, scaling_modifier, separate_sh, override_color, use_trained_exp = renderArgs

                    if config['name'] == 'test':
                        viewpoint.mask = torch.empty(0, device='cuda', dtype=torch.float32)

                    medium_rgb, medium_bs = scene.gaussians.get_medium_outputs_from_camera(viewpoint)
                    scene.gaussians.world_to_camera_points(viewpoint)
                    rend_res = renderFunc(viewpoint, scene.gaussians, pipe, background, medium_rgb, medium_bs, scaling_modifier=scaling_modifier, 
                                          separate_sh=separate_sh, override_color=override_color, use_trained_exp=use_trained_exp)

                    color_medium, out_medium = rend_res["render_medium"], rend_res["medium"]
                    image = color_medium + out_medium
                    image = torch.clamp(image, 0.0, 1.0)   # 利用高斯模型渲染当前视角下的图像
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)   # 读取真实图像
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                    if config['name'] == 'test':
                        save_dir = os.path.join(scene.model_path, "test_renders", f"iteration_{iteration}")
                        os.makedirs(save_dir, exist_ok=True)
                        image_name = os.path.splitext(os.path.basename(viewpoint.image_name))[0]
                        save_image(image.detach().cpu(), os.path.join(save_dir, f"{idx:05d}_{image_name}.png"))

                        depth = rend_res["depth"].detach().float().cpu().squeeze().numpy()
                        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        cv2.imwrite(os.path.join(save_dir, f"{idx:05d}_{image_name}_depth.png"), cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET))

                        ssims.append(ssim(image, gt_image))
                        lpipss.append(lpips(image, gt_image, net_type='vgg'))
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()  # 计算当前视角 L1 损失
                    psnr_test += psnr(image, gt_image).mean().double()   # 计算当前视角 PSNR

                if config['name'] == 'test':
                    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

                psnr_test /= len(config['cameras'])   # 求当前配置下所有相机的平均 PSNR
                l1_test /= len(config['cameras'])     # 求当前配置下所有相机的平均 L1      
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 12_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # if not args.disable_viewer:
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
