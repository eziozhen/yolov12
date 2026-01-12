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
    # setup_seed(42) # 选一个吉利的数字
    # 1. 加载模型
    model = YOLO("yolov13s.yaml")  # 或 yolov8n.pt 等
    
    # torch.backends.cudnn.benchmark = False 

    # 13s aug 0.916
    # 13s no aug 0.861
    # 13s no aug T=5 0.87
    # 13s aug T=5 0.927

    # 2. 绑定自定义 Trainer
    def train_wrapper(**kwargs):
        trainer = VideoTrainer(overrides=kwargs, _callbacks=model.callbacks)

        # 备份原始的预处理方法 (Bound Method)
        _original_preprocess = trainer.preprocess_batch

        def preprocess_batch_hook(self, batch):
            # 1. 先调用原始的预处理 (这一步会将图像转为 Tensor 并移到 GPU)
            # 注意：_original_preprocess 已经绑定了 self，直接传 batch 即可
            batch = _original_preprocess(batch)

            # 2. 在这里直接将数据注入到 MTI 模块
            # 此时 batch 肯定存在且格式正确
            if hasattr(self.model, 'modules'):
                for m in self.model.modules():
                    if m.__class__.__name__ == 'MTI_Block':
                        # 注入数据 (Box, Class, Batch_idx 等)
                        m.batch_data = batch
            
            return batch

        # 将这个 Hook 绑定回 trainer 实例
        # 这样训练时就会执行我们的逻辑
        trainer.preprocess_batch = types.MethodType(preprocess_batch_hook, trainer)

        trainer.model = model.model
        model.trainer = trainer
        trainer.train()

    model.train = train_wrapper


    # 3. 开始训练
    # 务必设置 mosaic=0.0，关闭原本的几何增强
    model.train(
        data="coco.yaml", 
        epochs=200,
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
