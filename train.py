import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import LetterBox, Format, Compose
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from copy import copy

import torch
import torch.nn as nn

CONST_T = 1


class TimeAwareBN(nn.BatchNorm2d):
    """
    时序感知的 BatchNorm。
    核心逻辑：
    1. 输入 (B*T, C, H, W)。
    2. 将其拆分为 '历史帧' (History) 和 '当前帧' (Current)。
    3. '当前帧': 正常通过 BN (Train模式)，计算均值方差，并更新全局 running_stats。
    4. '历史帧': 通过 BN (Eval模式)，利用刚刚更新的全局 running_stats 进行归一化，但不更新统计量。
    5. 拼接返回，保证梯度都能回传。
    """
    def __init__(self, original_bn, seq_len):
        # 初始化父类，继承原 BN 的所有参数 (num_features, eps, momentum, etc.)
        super().__init__(
            num_features=original_bn.num_features,
            eps=original_bn.eps,
            momentum=original_bn.momentum,
            affine=original_bn.affine,
            track_running_stats=original_bn.track_running_stats
        )
        # 复制参数权重 (Weight & Bias) 和 统计量 (Running Mean/Var)
        self.load_state_dict(original_bn.state_dict())
        self.seq_len = seq_len

    def forward(self, x):
        # 如果不是训练模式，或者不需要统计，直接走标准逻辑
        if not self.training or not self.track_running_stats:
            return super().forward(x)

        BT, C, H, W = x.shape
        T = self.seq_len

        # 安全检查：如果 batch size 对不上（比如最后一点尾数），降级为普通 BN
        if BT % T != 0:
            return super().forward(x)

        B = BT // T

        # 1. 拆解数据
        # 假设输入顺序是 [Seq0_t0, Seq0_t1... Seq0_curr, Seq1_t0...]
        x_reshaped = x.view(B, T, C, H, W)
        
        # 历史帧: (B, T-1, C, H, W) -> 展平
        x_hist = x_reshaped[:, :-1, ...].reshape(B * (T - 1), C, H, W)
        
        # 当前帧: (B, C, H, W)
        x_curr = x_reshaped[:, -1, ...]

        # 2. 处理当前帧 (Train Mode)
        # 这一步会：计算 x_curr 的均值方差用于归一化，并更新 running_mean/var
        # 这是你想要的关键点：统计量只被“干净”的当前帧更新
        out_curr = super().forward(x_curr)

        # 3. 处理历史帧 (Eval Mode)
        # 这一步会：使用 running_mean/var (刚才可能被更新过) 来归一化历史帧
        # 关键点：不会再次更新 running_stats，避免了重复数据的污染
        self.eval() # 临时切换到 eval
        out_hist = super().forward(x_hist)
        self.train() # 切回 train 状态

        # 4. 拼装回原始顺序 (B*T, ...)
        # 此时 out_hist 和 out_curr 都在计算图中，梯度都可以回传！
        out_reshaped = torch.cat([
            out_hist.view(B, T - 1, C, H, W), 
            out_curr.unsqueeze(1)
        ], dim=1)
        
        return out_reshaped.view(BT, C, H, W)


class StackLetterBox(LetterBox):
    """
    支持多帧堆叠图片 (Channel > 3) 的 LetterBox。
    原理：Resize 可以直接对多通道做，但 Padding (copyMakeBorder) 需要拆分成 3通道一组单独做。
    """
    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # --- 1. 计算缩放比例 (保持原版逻辑) ---
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # --- 2. 计算 Padding (保持原版逻辑) ---
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        # --- 3. 执行 Resize (保持原版逻辑) ---
        # cv2.resize 支持多通道，通常不需要修改
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        
        # --- 4. 执行 Padding (★核心修改★) ---
        # 原始代码直接调用 cv2.copyMakeBorder，多通道会崩。
        # 我们这里判断通道数，如果 > 4，就切分处理。
        
        # 获取通道数，如果是单通道灰度图，len可能只有2，C设为1
        C = img.shape[2] if len(img.shape) == 3 else 1
        
        if C > 4:
            # === 多通道切分方案 (CV方法) ===
            padded_slices = []
            # 每次处理 3 个通道 (OpenCV 最喜欢的数量)
            for i in range(0, C, 3):
                # 切片：例如 [:, :, 0:3], [:, :, 3:6] ...
                slice_img = img[:, :, i : min(i + 3, C)]
                
                # 对切片进行 Padding
                # value=(114, 114, 114) 会根据切片的通道数自动截取，
                # 比如切片只有1通道，它就只取 114；有3通道就取 (114,114,114)
                padded_slice = cv2.copyMakeBorder(
                    slice_img, top, bottom, left, right, 
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
                
                # 如果切片是单通道，copyMakeBorder 可能会返回 (H, W) 而不是 (H, W, 1)
                # 为了 concat，必须保持 3D 维度
                if len(padded_slice.shape) == 2:
                    padded_slice = padded_slice[..., None]
                    
                padded_slices.append(padded_slice)
            
            # 将 Pad 好的切片拼回去
            img = np.concatenate(padded_slices, axis=2)
            
        else:
            # === 标准方案 (保持原版逻辑) ===
            # 通道数 <= 4，直接调用，速度最快
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )

        # --- 5. 更新 Labels (保持原版逻辑) ---
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            # 调用父类的 _update_labels，因为这部分逻辑是通用的
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

# === 1. 自定义 Dataset (修复版) ===
class VideoIndexDataset(YOLODataset):
    def __init__(self, *args, seq_len=5, frame_interval=2, history_root=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.frame_interval = frame_interval
        self.history_root = Path(history_root)

        # ★★★ 关键修复：覆盖默认的 transforms ★★★
        # 我们自己在 load_image 里做好了 LetterBox Resize
        # 所以这里只需要一个 Format 来负责 (H,W,C)->(C,H,W) 和 BBox 格式化
        # 这样避免了标准管道里的 Resize/Mosaic 破坏我们的时序堆叠
        self.transforms = Compose([
            # 这里的 LetterBox 会接收 (img, labels) 并同时修改它们
            StackLetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=False),
            
            # Format 负责最后的格式转换 (BBox xywh -> xyxy 等)
            Format(
                bbox_format="xywh", 
                normalize=True, 
                return_mask=False, 
                return_keypoint=False,
                batch_idx=True,
                mask_ratio=4,
                mask_overlap=True
            )
        ])

    def load_image(self, i):
        """
        必需返回 3 个值: (image, (h0, w0), (h, w))
        image 格式必须是 (H, W, Channels)，其中 Channels = T * 3
        """
        im_file = self.im_files[i]
        path_obj = Path(im_file)
        
        # 1. 解析文件名
        try:
            name_parts = path_obj.stem.split('_') 
            frame_idx = int(name_parts[-1]) 
            video_prefix = "_".join(name_parts[:-1]) 
        except:
            return self._load_single_frame_fallback(im_file)

        frames = []
        # 注意：使用 self.imgsz 而不是 self.args.imgsz
        target_size = self.imgsz 
        
        # 用于记录当前帧（Anchor）的尺寸信息，供 Label 缩放使用
        h0, w0 = 0, 0
        h, w = 0, 0
        
        # 初始化回溯逻辑
        # 我们先读当前帧，确保它是存在的
        curr_im = cv2.imread(im_file)
        curr_im = cv2.cvtColor(curr_im, cv2.COLOR_BGR2RGB)
        if curr_im is None: raise FileNotFoundError(f"Missing current frame: {im_file}")
        
        # 记录原始尺寸 (h0, w0)
        h0, w0 = curr_im.shape[:2]
        
        # 缓存最近有效帧，用于 Padding
        last_valid_frame = curr_im
        
        # 临时列表：倒序存储 [Current, t-2, t-4...]
        collected_frames = [curr_im]

        # 2. 回溯历史 (t=1 到 seq_len-1)
        for t in range(1, self.seq_len):
            offset = t * self.frame_interval
            target_idx = frame_idx - offset
            
            im = None
            if target_idx >= 0:
                p = str(self.history_root / f"{video_prefix}_{target_idx:05d}{path_obj.suffix}")
                raw_im = cv2.imread(p)
                raw_im = cv2.cvtColor(raw_im, cv2.COLOR_BGR2RGB)
                if raw_im is not None:
                    if raw_im.shape[:2] != (h0, w0):
                        raw_im = cv2.resize(raw_im, (w0, h0))
                    im = raw_im

            if im is not None:
                last_valid_frame = im
                collected_frames.append(im)
            else:
                # Padding: 没读到就用上一个有效的填充
                collected_frames.append(last_valid_frame)

        # 3. 整理输出
        # 反转回正序: [t-8, t-6, t-4, t-2, Current]
        collected_frames.reverse()
        
        # ★★★ 关键修复：在 Channel 维度堆叠 (H, W, 3) list -> (H, W, T*3) ★★★
        # 这样 YOLO 的 Format transform 就能自动把它变成 (T*3, H, W)
        img_stack = np.concatenate(collected_frames, axis=2) 
        # img_stack = np.concatenate([curr_im] * self.seq_len, axis=2) 
        
        return img_stack, (h0, w0), (h0, w0)

    def _load_single_frame_fallback(self, f):
        """兜底逻辑：文件名解析失败时调用"""
        im = cv2.imread(f)
        h0, w0 = im.shape[:2]
        # 复制 T 次
        img_stack = np.concatenate([im] * self.seq_len, axis=2)
        return img_stack, (h0, w0), (h0, w0)

class VideoValidator(DetectionValidator):
    def preprocess(self, batch):
        batch = super().preprocess(batch)
        
        imgs = batch['img']
        B, C_total, H, W = imgs.shape
        T = CONST_T
        
        # 1. 仅展平图片 (B*T, 3, H, W) 供 Backbone 使用
        batch['img'] = imgs.view(B, T, 3, H, W).view(B * T, 3, H, W)
        # 2. ★★★ 核心修复：用 None 填充 ratio_pad ★★★
        # 这里的长度必须是 B (对应预测结果数量)，而不是 B*T
        # 因为我们的模型会自动把 B*T 的输入 Slice 成 B 个输出
        # 验证器只关心这 B 个输出
        batch['ratio_pad'] = [None] * B
            
        # 注意：这里 DO NOT 扩充 ori_shape 或 batch_idx
        # 因为预测结果只有 B 个，元数据也应该只有 B 个
        
        return batch

# === 2. Trainer (注入标签) ===
class VideoTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return VideoIndexDataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
            augment=False, rect=False, stride=int(self.stride),
            data=self.data, classes=self.args.classes,
            # 参数配置
            seq_len=CONST_T, 
            frame_interval=4, 
            history_root="/home/caozhenzhen/xiaomai/ALL_JPEG" # <--- 确保路径正确
        )

    # ★★★ 新增：注册 Validator ★★★
    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'mti_loss'
        return VideoValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args),
            _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        saved_ratio_pad = batch.get('ratio_pad')
        
        # # 注入标签
        # if hasattr(self.model, 'model'):
        #      for m in self.model.modules():
        #         if m.__class__.__name__ == 'MTI_Block':
        #             m.batch_labels = batch['bboxes']

        if hasattr(self.model, 'model'):
             for m in self.model.modules():
                if m.__class__.__name__ == 'MTI_Block':
                    # 我们需要 batch_idx 来知道框属于哪张图
                    # 我们需要 bboxes 来知道框在哪里
                    m.batch_data = batch # 直接把整个 batch 字典给它

        imgs = batch['img']
        B, C_total, H, W = imgs.shape
        T = CONST_T
        
        # 1. 展平图片 (B*T, 3, H, W)
        batch['img'] = imgs.view(B, T, 3, H, W).view(B * T, 3, H, W)
        
        # 2. ★★★ 修复：绝对不要修改 batch_idx ★★★
        # 我们的模型输出 B 个预测结果，对应 batch 中的 B 个样本。
        # 原生的 batch_idx (0,0,1,2...) 正是对应这 B 个样本的。
        # 之前乘以 T 是错误的，导致 Loss 找不到对应的 Target。
        
        # 3. ★★★ 修复：不要扩充元数据 ★★★
        # im_file, ori_shape 等保持 B 的长度即可。
        
        # 4. 修复 ratio_pad (防止 val 报错)
        # batch['ratio_pad'] = [None] * B
        if saved_ratio_pad is not None and len(saved_ratio_pad) >= B:
             batch['ratio_pad'] = saved_ratio_pad[:B]

        return batch

def replace_bn_with_time_aware(module, seq_len):
    """
    递归替换模型中的 BatchNorm2d 为 TimeAwareBN
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # 创建自定义 BN，继承原参数
            new_bn = TimeAwareBN(child, seq_len)
            # 替换
            setattr(module, name, new_bn)
        else:
            # 递归
            replace_bn_with_time_aware(child, seq_len)

if __name__ == "__main__":
    model = YOLO("yolov12s.yaml") 
    # 0.868   T = 1
    
    # train88  yolov13s
    # train94  yolov12s aug
    
    # model.load("runs/detect/train94/weights/best.pt")

    # replace_bn_with_time_aware(model.model, CONST_T)

    # def train_wrapper(**kwargs):
    #     trainer = VideoTrainer(overrides=kwargs, _callbacks=model.callbacks)
    #     trainer.model = model.model
    #     model.trainer = trainer
    #     trainer.train()

    # model.train = train_wrapper

    # print(model.model)

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

