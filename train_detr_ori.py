import cv2
import types
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from copy import copy

from ultralytics import RTDETR

if __name__ == "__main__":
    # 1. 加载模型
    model = RTDETR('rtdetr-n.yaml')

    # torch.backends.cudnn.benchmark = False 

    # 13s aug 0.916
    # 13s no aug 0.861
    # 13s no aug T=5 0.87
    # 13s aug T=5 0.927

    # 3. 开始训练
    # 务必设置 mosaic=0.0，关闭原本的几何增强
    model.train(
        data="coco_dart.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        seed=42,
        cos_lr=True,
        # amp=True,
        # # --- 1. 关闭几何变换 ---
        mosaic=0.0,      # 关闭马赛克
        mixup=0.0,       # 关闭混合
        copy_paste=0.0,  # 关闭复制粘贴

        # # --- 2. 关闭色彩/光照变换 ---
        # hsv_h=0.0,       # 色调
        # hsv_s=0.0,       # 饱和度
        # hsv_v=0.0,       # 亮度

        # # --- 3. 关闭空间随机性 ---
        # degrees=0.0,     # 旋转
        # translate=0.0,   # 平移
        # scale=0.0,       # 缩放 (设为0表示不进行随机缩放，只做 LetterBox)
        # shear=0.0,       # 剪切
        # perspective=0.0, # 透视
        # fliplr=0.0,      # 左右翻转 (你的Dataset目前没有翻转，所以这里也要关)
        # flipud=0.0,      # 上下翻转

        # --- 4. 其他 ---
        rect=False,      # 训练时通常为 False (转为正方形)，这也符合你的 LetterBox 逻辑
        cache=False,     # 关闭缓存，防止干扰
        plots=False,
        # freeze=10,
        workers=32
    )

