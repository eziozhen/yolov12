import cv2
import types
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from copy import copy

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLOVideoDataset
from ultralytics.data.augment import LetterBox, Format, Compose
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel

HISTORY = 4
STRIDE_STEP = 4

class VideoTrainer(DetectionTrainer):
    """
    自定义 Trainer，用于支持视频序列训练。
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        重写构建数据集的方法，将其指向 YOLOVideoDataset
        """
        # 从参数中获取时序配置，如果没传则使用默认值
        # self.args 是 YOLO 传入的所有超参数
        
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)

        # 实例化你的自定义 Dataset
        dataset = YOLOVideoDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",  # 只有训练模式才开启增强
            hyp=self.args,
            rect=False,  # 视频训练通常关闭矩形训练，保证 batch 维度对其
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
            # --- 你的自定义参数 ---
            n_history=HISTORY,
            stride_step=STRIDE_STEP, # 注意不要跟 model stride 混淆，这里改个名区分一下
            history_path="/home/caozhenzhen/xiaomai/ALL_JPEG"
        )
        
        return dataset

if __name__ == "__main__":
    # 1. 加载模型
    model = YOLO("yolov13s.yaml")  # 或 yolov8n.pt 等
    
    # 13s aug 0.916
    # 13s no aug 0.861
    # 13s no aug T=5 0.865

    # 2. 绑定自定义 Trainer
    def train_wrapper(**kwargs):
        trainer = VideoTrainer(overrides=kwargs, _callbacks=model.callbacks)
        trainer.model = model.model
        model.trainer = trainer
        trainer.train()

    model.train = train_wrapper

    nn_model = model.model

    # 备份原版 forward
    nn_model._orig_forward = nn_model.forward

    # 定义极简 Wrapper
    def inject_labels_wrapper(self, batch, *args, **kwargs):
        # -------------------------------------------------------
        # 1. 注入 Labels (给你的 MTI 模块用)
        # -------------------------------------------------------
        if isinstance(batch, dict):
            # 懒加载缓存：只在第一次找 MTI 模块
            if not hasattr(self, '_cached_mti'):
                self._cached_mti = next((m for m in self.modules() if m.__class__.__name__ == 'MTI_Block'), None)
            
            # 注入数据
            if self._cached_mti:
                self._cached_mti.batch_data = batch

        return self._orig_forward(batch, *args, **kwargs)

    nn_model.forward = types.MethodType(inject_labels_wrapper, nn_model)

    # 3. 开始训练
    # 务必设置 mosaic=0.0，关闭原本的几何增强
    model.train(
        data="coco.yaml", 
        epochs=100,
        batch=8,
        imgsz=640,
        # --- 1. 关闭几何变换 ---
        mosaic=0.0,      # 关闭马赛克
        mixup=0.0,       # 关闭混合
        copy_paste=0.0,  # 关闭复制粘贴

        # --- 2. 关闭色彩/光照变换 ---
        hsv_h=0.0,       # 色调
        hsv_s=0.0,       # 饱和度
        hsv_v=0.0,       # 亮度
        
        # --- 3. 关闭空间随机性 ---
        degrees=0.0,     # 旋转
        translate=0.0,   # 平移
        scale=0.0,       # 缩放 (设为0表示不进行随机缩放，只做 LetterBox)
        shear=0.0,       # 剪切
        perspective=0.0, # 透视
        fliplr=0.0,      # 左右翻转 (你的Dataset目前没有翻转，所以这里也要关)
        flipud=0.0,      # 上下翻转

        # --- 4. 其他 ---
        rect=False,      # 训练时通常为 False (转为正方形)，这也符合你的 LetterBox 逻辑
        cache=False,     # 关闭缓存，防止干扰
        plots=False,
        # amp=False,
        # freeze=10,
        workers=16
    )
