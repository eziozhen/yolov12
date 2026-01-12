import cv2
import types
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from copy import copy

from ultralytics import RTDETR
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLOVideoDataset
from ultralytics.data.augment import LetterBox, Format, Compose
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel

HISTORY = 4
STRIDE_STEP = 4

class VideoTrainer(RTDETRTrainer):
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

def custom_predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
    """
    Modified predict function to handle Video Sequence (B*T input -> B output).
    Keeps the original logic structure intact.
    """
    y, dt, embeddings = [], [], []  # outputs
    
    # --- 1. Original Backbone + Encoder Loop ---
    for m in self.model[:-1]:  # except the head part
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
        if embed and m.i in embed:
            embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
            if m.i == max(embed):
                return torch.unbind(torch.cat(embeddings, 1), dim=0)
    
    # --- 2. Head Inference with Video Logic ---
    head = self.model[-1]
    
    # Gather encoder outputs (features) normally
    x_in = [y[j] for j in head.f] 

    # [NEW] Video Sequence Handling
    # Check if we are training and if inputs (B*T) mismatch targets (B)
    if self.training and batch is not None:
        # batch['gt_groups'] length is the true Batch Size (B)
        bs_gt = len(batch.get('gt_groups', []))
        # Feature map batch size is (B * T)
        bs_feat = x_in[0].shape[0]

        # If features contain history frames but targets don't:
        if bs_gt > 0 and bs_feat > bs_gt and bs_feat % bs_gt == 0:
            n_frames = bs_feat // bs_gt
            # Reshape (B*T, C, H, W) -> (B, T, C, H, W) and take the last frame (Keyframe)
            # You can change '[:, -1]' to other fusion logic if needed
            x_in = [feat.view(bs_gt, n_frames, *feat.shape[1:])[:, -1] for feat in x_in]

    # Pass the (potentially sliced) features to the head
    x = head(x_in, batch)  # head inference
    return x

def custom_loss(self, batch, preds=None):
    if not hasattr(self, "criterion"):
        self.criterion = self.init_criterion()

    img = batch["img"]
    bs_input = len(img) # B * T

    # 1. 获取 n_history
    n_history = HISTORY + 1
    if n_history == 0:
        # 自动推导兜底策略：如果 batch 是 5 的倍数且 > 16 (根据你的实际情况调整)
        if bs_input >= 5 and bs_input % 5 == 0:
             n_history = 5 # 假设是 5 (4历史+1当前)
        else:
             n_history = 1

    # 计算实际的 Batch Size (B)，例如 40 // 5 = 8
    bs = bs_input // n_history if n_history > 1 else bs_input

    # 2. 直接获取标签 (你的 Dataset 已经保证了它们对应 B 个样本)
    # 不需要筛选！不需要过滤！
    batch_idx = batch["batch_idx"]

    # 3. 计算 gt_groups
    # 这里很关键：我们需要告诉 CDN 有多少组 GT。
    # batch_idx 的范围已经是 0 ~ (B-1) 了 (例如 0~7)
    # 所以我们直接统计每个 index 有多少个 GT 即可
    gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]

    # 4. 构建 targets
    targets = {
        "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
        "bboxes": batch["bboxes"].to(device=img.device),
        "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
        "gt_groups": gt_groups, # 长度为 B (8)
    }
    # 调用 predict
    preds = self.predict(img, batch=targets) if preds is None else preds
    
    dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
    
    if dn_meta is None:
        dn_bboxes, dn_scores = None, None
    else:
        dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
        dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

    dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
    dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

    loss = self.criterion(
        (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
    )

    # ====================================================================
    # ★★★ MTI-Net 辅助 Loss 计算 (新增部分) ★★★
    # ====================================================================
    mti_loss = torch.tensor(0.0, device=img.device)
    
    # self 就是 Model 实例，直接遍历
    for m in self.modules():
        if hasattr(m, 'loss_data') and m.loss_data is not None:
            f_repaired, f_orig, mask = m.loss_data

            if mask.sum() > 0:
                # 公式: || (Repaired - StopGrad[Orig]) * Mask ||^2
                target = f_orig.detach()
                diff = (f_repaired - target) * mask
                
                # 归一化 MSE
                current_loss = (diff ** 2).sum() / (mask.sum() + 1e-6)
                mti_loss += current_loss
            
            # ★★★ 算完后清空，防止显存泄漏 ★★★
            m.loss_data = None

    # 给权重 (例如 0.005)
    mti_loss_weighted = mti_loss * 0.005
    
    total_loss = sum(loss.values()) + mti_loss_weighted

    loss_items = torch.as_tensor(
        [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]] + [mti_loss_weighted.detach()], 
        device=img.device
    )

    return total_loss, loss_items


if __name__ == "__main__":
    # setup_seed(42) # 选一个吉利的数字
    # 1. 加载模型
    model = RTDETR("rtdetr-n1.yaml")  # 或 yolov8n.pt 等
    
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

        # 3. [KEY STEP] Monkey Patch the Model Methods
        # Bind the custom functions to the model instance
        # model.model.predict = types.MethodType(custom_predict, model.model)
        # model.model.loss = types.MethodType(custom_loss, model.model)

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
