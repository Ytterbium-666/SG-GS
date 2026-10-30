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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
from pathlib import Path

WARNED = False  # 定义一个全局变量 WARNED，用于记录是否已经打印过“大图像自动缩放”的提示

# 定义函数 loadCam，用于把 CameraInfo 转换成真正训练/渲染用的 Camera 对象
def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = Image.open(cam_info.image_path)  # 读取当前相机对应的图像

    image_path = Path(cam_info.image_path)
    mask_path = image_path.parents[2] / "water_masks" / f"{image_path.stem}_water_mask.png"
    mask = Image.open(mask_path)

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512  # 逆深度图
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)  # 16 位 PNG 深度图范围是 0 到 65535

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size  # 图像的分辨率
    if args.resolution in [1, 2, 4, 8]:
        # 计算目标分辨率
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            # 如果 resolution=-1，使用自动缩放策略
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))  # 计算最终的分辨率

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, mask=mask, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    # 这里保存的是相机在世界坐标系下的平移和旋转
    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry