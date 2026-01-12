"""
=======================================================================================================
DART (Dense Association & Restoration Tracking) - v3.0
高级视频数据清洗、轨迹补全与自动标注系统
=======================================================================================================

【系统设计理念】
本系统旨在利用“时序一致性”解决目标检测中的漏检、抖动和ID频繁切换问题。
通过引入“人工锚点”和“贝叶斯漂移计算”，实现了从粗糙检测到完美标注的自动化飞跃。

【核心数据流与优先级策略】
系统的输入分为两个层级，处理逻辑完全不同：
1. 第一优先级 (绝对真值)：Config.GT_DIR ("predict")
   - 定义：人工精心标注的 XML 文件。
   - 权重：Score = 2.0 (最高置信度)。
   - 行为：
     - SOT 阶段：强制重置跟踪器位置，对齐真值。
     - 漂移阶段：Drift = 0 (作为最强的时序锚点)。
     - 融合阶段：跳过计算，直接输出，不接受任何算法修改。
2. 第二优先级 (模型检测)：Config.YOLO_DIR ("predict_au")
   - 定义：YOLO 等模型的初步检测结果。
   - 权重：Score = 原始检测置信度 (通常 0.5~0.9)。
   - 行为：
     - SOT 阶段：执行“防抖逻辑”，仅在偏差过大时重置，否则信任平滑轨迹。
     - 融合阶段：参与 DART 加权融合与裁判。

【全链路处理流程详解】

阶段一：混合驱动的数据增强 (Hybrid SOT Generation)
-------------------------------------------------------------------------------------------------------
目标：生成稠密、无漏检的候选框流。
逻辑：
1. **双向遍历**：分别进行正向 (Forward, t->t+1) 和反向 (Backward, t->t-1) 跟踪。
2. **SOT 智能更新策略**：
   - 遇到【人工真值】：无条件信任。强制将 SOT 跟踪器“瞬移”到真值位置，消除之前累积的误差。
   - 遇到【模型检测】：执行防抖 (Anti-Jitter)。
     - 计算 SOT 预测框与 YOLO 检测框的中心距离。
     - 如果距离 < 10px：认为物体静止或微动，检测器在抖。保持 SOT 轨迹，忽略 YOLO 的微小跳变。
     - 如果距离 > 10px：认为物体发生了真实位移或 SOT 跟丢。重置 SOT 到 YOLO 位置。
   - 遇到【漏检 (无框)】：利用 SOT 的惯性预测填补空缺 (Inpainting)。

阶段二：双流 ByteTrack 关联 (Dual-Stream Association)
-------------------------------------------------------------------------------------------------------
目标：建立长时序 ID，解决遮挡后的身份恢复。
逻辑：
- 将阶段一生成的两组稠密检测框（正向/反向）分别输入 ByteTrack。
- 得到两条独立的轨迹链：Forward Tracks 和 Backward Tracks。

阶段三：漂移计算 (Temporal Drift Calculation)
-------------------------------------------------------------------------------------------------------
目标：量化每一帧预测框的“不可信度” (Uncertainty)。
数学模型：假设误差服从维纳过程 (Wiener Process)，方差随时间线性积累。
- **锚点 (Anchor)**：Score > 0.8 的框。人工真值 (Score=2.0) 是最强锚点。
- **Drift (d)**：当前帧距离最近锚点的时间步数。
  - Forward Drift (df): 距离“上一个”锚点的帧数。
  - Backward Drift (db): 距离“下一个”锚点的帧数。

阶段四：自适应融合与裁判 (Adaptive Fusion & Rescue)
-------------------------------------------------------------------------------------------------------
目标：合并双向结果，生成最终标注。
策略矩阵：
1. **人工帧 (Is Manual)**：
   - 动作：直接透传 (Pass-through)。不进行融合，保证真值 100% 还原。
2. **普通帧 - 分歧小 (Low Divergence, IoU > 0.7)**：
   - 假设：正反向都在跟踪同一物体，存在精度误差。
   - 动作：**贝叶斯加权融合**。
     Box_final = (Box_f * w_f + Box_b * w_b)
     其中权重 w ~ 1/Drift。谁离锚点近，谁的权重就大，结果就越偏向谁。
3. **普通帧 - 分歧大 (High Divergence, IoU < 0.7)**：
   - 假设：一方跟丢了，产生了幻觉。
   - 动作：**基于 Drift 的裁判救援**。
     - 如果 df < 15 且 db 很打：说明正向刚经过锚点，正向是对的。-> 选 Forward。
     - 如果 db < 15 且 df 很大：说明反向刚经过锚点，反向是对的。-> 选 Backward。
     - 如果都很大：判定为不可靠，丢弃。

阶段五：完美帧挖掘 (Perfect Frame Mining)
-------------------------------------------------------------------------------------------------------
- 逻辑：计算全图的平均散度 (Average Divergence)。
- 动作：如果一张图中所有物体的正反向预测几乎完全重合 (Div < 0.05)，说明该帧标注质量极高，自动保存图片作为“完美样本”。

=======================================================================================================
"""

import os
import cv2
import glob
import numpy as np
import xml.etree.ElementTree as ET
import pickle
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from multiprocessing import Process, Manager

# -----------------------------------------------------------
# 引入 ByteTrack
# -----------------------------------------------------------
from byte_tracker.tracker.byte_tracker import BYTETracker
# try:
#     from byte_tracker.tracker.byte_tracker import BYTETracker
# except ImportError:
#     print("【警告】未找到 ByteTrack 模块！正在使用占位类。")
#     class BYTETracker:
#         def __init__(self, args, frame_rate=30): pass
#         def update(self, output_results, img_info, img_size): return []

# -----------------------------------------------------------
# 全局配置参数
# -----------------------------------------------------------
class Config:
    # [关键路径配置]
    IMG_DIR = "/home/caozhenzhen/xiaomai/ALL_JPEG"                # 图片文件夹
    GT_DIR = "/home/caozhenzhen/xiaomai/yolov5-pytorch-main/VOCdevkit/VOC2007/Annotations"              # 【优先级1】人工真值文件夹 (绝对锚点)
    YOLO_DIR = "/home/caozhenzhen/xiaomai/yolov5-pytorch-main-modify/predict_low_conf/xml"         # 【优先级2】YOLO检测文件夹 (自动检测)

    OUT_DIR = "dart_single/predict_final_dart"  # 最终输出结果
    VIS_DIR_S1 = "dart_single/vis_stage1_nms"

    # ================= [新增缓存配置] =================
    USE_CACHE = True                # 是否启用缓存
    CACHE_DIR = "/home/caozhenzhen/xiaomai/yolov5-pytorch-main/dart_cache_new"        # 缓存文件存放目录
    FORCE_REBUILD = False           # 设为 True 可强制重新运行 CSRT (忽略已有缓存)

    # [DART 完美帧筛选]
    SAVE_PERFECT_FRAMES = True
    BEST_FRAMES_DIR = "dart_single/dart_perfect_samples"
    BEST_FRAMES_XML_DIR = "dart_single/dart_perfect_samples_xml"
    PERFECT_FRAME_DIV = 0.1        # 代表正向跟踪和反向跟踪结果之间的“散度”（Divergence）。0.05 意味着正反向轨迹的 IoU（交并比）必须高于 0.95。

    # [核心] 算法阈值
    MAX_DIVERGENCE = 0.3            # 1-IoU > 0.3 视为严重分歧
    JITTER_TOLERANCE = 50.0         # [防抖] 像素偏差小于此值，不重置 SOT
    TRUST_LIMIT = 30.0              # [救援] 距离锚点多少帧以内视为可信

    # 数据增强参数
    ANCHOR_CONF_THRES = 0.6         # 模型分 > 0.8 视为锚点
    MANUAL_SCORE = 2.0              # 人工标注的强制分数 (远超模型分)

    # SOT & NMS
    SOT_TYPE = "CSRT"               
    MATCH_THRES = 0.4
    NMS_THRES = 0.45

    # ByteTrack 参数
    TRACK_THRESH = 0.05
    TRACK_BUFFER = 60      
    MATCH_THRESH = 0.8 
    MOT20 = False

    # 可视化
    VISUALIZE = False
    VIS_DIR = "dart_single/vis_dart"

# -----------------------------------------------------------
# 基础工具函数
# -----------------------------------------------------------
class TrackerArgs:
    def __init__(self):
        self.track_thresh = Config.TRACK_THRESH
        self.track_buffer = Config.TRACK_BUFFER
        self.match_thresh = Config.MATCH_THRESH
        self.mot20 = Config.MOT20

def nms_center_distance(boxes, scores, dist_thres=30):
    """
    [中心点距离去重]
    不管 IoU 是多少，只要两个框的中心点距离小于 dist_thres (像素)，
    就认为它们是同一个物体，保留分数高的那个。
    
    参数:
        dist_thres: 像素阈值。建议设置为物体平均宽度的 1/3。
                    例如 4K 图麦穗设 30-40，1080P 图设 15-20。
    """
    if len(boxes) == 0: return []
    
    # 计算所有框的中心点
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 计算当前最高分框(i)与其他所有剩余框的距离
        xx = (cx[order[1:]] - cx[i]) ** 2
        yy = (cy[order[1:]] - cy[i]) ** 2
        dist = np.sqrt(xx + yy)
        
        # 找出距离足够远(>阈值)的框，保留下来进入下一轮
        # 距离近的(<阈值)在这里就被过滤掉了
        inds = np.where(dist > dist_thres)[0]
        
        # 更新 order (注意 inds 是相对于 order[1:] 的索引，所以要 +1)
        order = order[inds + 1]
        
    return keep

def nms_containment_safe(boxes, scores, iomin_thres=0.8, score_gap=0.6):
    """
    稳妥的包含去重 (Safe IoMin NMS)
    
    参数:
        iomin_thres: 包含程度阈值 (默认0.8)。如果小框 80% 面积在大框内...
        score_gap:   分数级差 (默认0.6)。只有当 小框分数 < 大框分数 * 0.6 时，才删除。
                     防止误删那些"虽然被包含，但置信度很高"的真实物体。
    """
    if len(boxes) == 0: return []
    
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 计算当前最高分框(i)与其他框的 IoMin
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        # IoMin = 交集 / 较小框的面积
        min_areas = np.minimum(areas[i], areas[order[1:]])
        ovr = inter / (min_areas + 1e-6)
        
        # 找出满足"包含关系"的框的索引
        contained_inds = np.where(ovr > iomin_thres)[0]
        
        # [核心稳妥逻辑]
        # 在这些被包含的框中，只删除那些分数"显著低于"当前框的
        # scores[order[1:][idx]] 取出被包含框的分数
        # scores[i] 是当前大框的分数
        
        # 如果 小框分数 < 大框分数 * score_gap，则认为是垃圾，需要删除
        # 反之，如果小框分数很高，我们保留它 (不放入 delete_mask)
        
        # 这里我们需要找出"应该被保留的索引"
        # 原始 order[1:] 中，除了被安全删除的，其他的都留到下一轮
        
        # 1. 默认保留所有
        mask = np.ones(order.size - 1, dtype=bool)
        
        # 2. 遍历被包含的框，检查分数
        for idx in contained_inds:
            score_small = scores[order[idx + 1]] # +1 是因为 order 切片了
            score_large = scores[i]
            
            # 如果分数悬殊，杀掉 (mask=False)
            if score_small < (score_large * score_gap):
                mask[idx] = False
            # 否则保留 (mask=True)，即"免死金牌"
        
        # 更新 order，只留下未处理的 + 免死的
        order = order[1:][mask]

    return keep

def refine_boxes_smart_repulsion(boxes, scores, iou_ceiling=1.0, ratio_thr=4.0): # 阈值改为1.0，允许所有重叠进入
    """
    [智能排斥修正 v2.0 - 包含切割版]
    不仅处理边缘重叠，也处理"大框套小框"。
    如果大框包含了小框，大框会根据小框的位置进行"躲避性收缩"，
    从而保留大框中未重叠的有效区域。
    """
    if len(boxes) == 0: return boxes, scores
    
    n = len(boxes)
    new_boxes = boxes.copy()
    
    # 标记框是否有效
    valid_mask = np.ones(n, dtype=bool)
    
    # 计算面积 (用于判断谁大谁小)
    area = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
    
    for i in range(n):
        if not valid_mask[i]: continue
        
        for j in range(i + 1, n):
            if not valid_mask[j]: continue
            
            b1 = new_boxes[i]
            b2 = new_boxes[j]
            
            # 1. 计算重叠区域
            xx1 = max(b1[0], b2[0]); yy1 = max(b1[1], b2[1])
            xx2 = min(b1[2], b2[2]); yy2 = min(b1[3], b2[3])
            w_inter = xx2 - xx1
            h_inter = yy2 - yy1
            
            # 无重叠，跳过
            if w_inter <= 0 or h_inter <= 0: continue
            
            # 2. 决定谁是"弱者" (需要收缩的一方)
            # 逻辑：
            # A. 分数优先：分低的缩
            # B. 分数接近(差值<0.1)：面积大的缩 (默认为大框是虚胖)
            s1 = scores[i]
            s2 = scores[j]
            
            victim = -1
            winner = -1
            
            if s1 > s2 + 0.1: # b1 强
                victim = j; winner = i
            elif s2 > s1 + 0.1: # b2 强
                victim = i; winner = j
            else:
                # 分数差不多，让大的那个缩
                if area[j] > area[i]:
                    victim = j; winner = i
                else:
                    victim = i; winner = j
            
            # 获取坐标
            b_vic = new_boxes[victim]
            b_win = new_boxes[winner]
            
            # 3. [核心] 智能收缩方向判断 (基于中心点)
            # 我们看 "赢家" 在 "输家" 的哪个方位，输家就往反方向躲
            
            cx_vic = (b_vic[0] + b_vic[2]) / 2
            cy_vic = (b_vic[1] + b_vic[3]) / 2
            cx_win = (b_win[0] + b_win[2]) / 2
            cy_win = (b_win[1] + b_win[3]) / 2
            
            # 计算相对位置偏差
            dx = cx_win - cx_vic
            dy = cy_win - cy_vic
            
            # 判断是横向排斥还是纵向排斥
            # 比如麦穗通常是左右排列，横向偏差通常更大，或者是重叠形状决定的
            
            # 如果横向距离 > 纵向距离，或者横向重叠比例更小，优先切横向
            # 这里我们简单策略：切那个"更容易切开"的边
            
            # 限制：不能把框切没了。保留至少 20% 的尺寸。
            min_w = (b_vic[2] - b_vic[0]) * 0.2
            min_h = (b_vic[3] - b_vic[1]) * 0.2

            cur_w = b_vic[2] - b_vic[0]
            cur_h = b_vic[3] - b_vic[1]

            def is_thin_strip(w, h):
                # 防止除以0，加上极小值
                return max(w, h) / (min(w, h) + 1e-6) > ratio_thr

            if abs(dx) > abs(dy): 
                # --- 横向关系 (左右) ---
                if dx < 0: 
                    # 赢家在左边 -> 输家(大框)切掉左边，保留右边
                    # 新的 x1 = 赢家的 x2
                    new_x1 = b_win[2]
                    new_w = b_vic[2] - new_x1
                    # 安全检查
                    if new_w > min_w:
                        if is_thin_strip(new_w, cur_h):
                            valid_mask[victim] = False 
                        else:
                            new_boxes[victim, 0] = new_x1 
                else:
                    # 赢家在右边 -> 输家(大框)切掉右边，保留左边
                    # 新的 x2 = 赢家的 x1
                    new_x2 = b_win[0]
                    new_w = new_x2 - b_vic[0]
                    if new_w > min_w:
                        if is_thin_strip(new_w, cur_h):
                            valid_mask[victim] = False
                        else:
                            new_boxes[victim, 2] = new_x2
                        
            else:
                # --- 纵向关系 (上下) ---
                if abs(s1 - s2) > 0.1:
                    if dy < 0:
                        # 赢家在上面 -> 输家切掉上面，保留下面
                        new_y1 = b_win[3]
                        new_h = b_vic[3] - new_y1
                        if new_h > min_h:
                            if is_thin_strip(cur_w, new_h):
                                valid_mask[victim] = False
                            else:
                                new_boxes[victim, 1] = new_y1
                    else:
                        # 赢家在下面 -> 输家切掉下面，保留上面
                        new_y2 = b_win[1]
                        new_h = new_y2 - b_vic[1]
                        if new_h > min_h:
                            if is_thin_strip(cur_w, new_h):
                                valid_mask[victim] = False
                            else:
                                new_boxes[victim, 3] = new_y2
                else:
                    if area[i] < area[j]:
                        valid_mask[i] = False
                    else:
                        valid_mask[j] = False
            if valid_mask[victim]:
                # 更新面积 (因为框变小了，下一轮比较需要用新面积)
                area[victim] = (new_boxes[victim, 2] - new_boxes[victim, 0]) * \
                            (new_boxes[victim, 3] - new_boxes[victim, 1])

    return new_boxes[valid_mask], scores[valid_mask], valid_mask

def iou_batch(bb_test, bb_gt):
    """ 批量计算 IoU """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    denom = ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) +
              (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return wh / (denom + 1e-6)

def nms_numpy(boxes, scores, threshold):
    if len(boxes) == 0: return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep

def read_xml(path):
    if not os.path.exists(path): return np.empty((0, 5))
    tree = ET.parse(path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        b = obj.find('bndbox')
        score_node = obj.find('score')
        score = float(score_node.text) if score_node is not None else 1.0
        boxes.append([float(b.find('xmin').text), float(b.find('ymin').text), 
                      float(b.find('xmax').text), float(b.find('ymax').text), score])
    return np.array(boxes) if boxes else np.empty((0, 5))

def write_xml(tracks, img_path, out_path, shape):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = os.path.basename(img_path)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(shape[1])
    ET.SubElement(size, "height").text = str(shape[0])
    for t in tracks:
        x1, y1, x2, y2 = t['box']
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "wheat"
        ET.SubElement(obj, "track_id").text = str(t['id'])
        ET.SubElement(obj, "score").text = f"{t['score']:.2f}"
        if 'div' in t:
            ET.SubElement(obj, "uncertainty").text = f"{t['div']:.4f}"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(x1))
        ET.SubElement(bnd, "ymin").text = str(int(y1))
        ET.SubElement(bnd, "xmax").text = str(int(x2))
        ET.SubElement(bnd, "ymax").text = str(int(y2))
    tree = ET.ElementTree(root)
    # ET.indent(tree, space="  ", level=0)
    tree.write(out_path)

# -----------------------------------------------------------
# 核心组件 1: SOT 生成器 (实现优先级逻辑)
# -----------------------------------------------------------
class SOTGenerator:
    def __init__(self):
        pass 
    
    def run(self, frame_list, direction="Forward"):
        iter_list = frame_list if direction == "Forward" else frame_list[::-1]
        results = {}
        trackers = [] # 结构: {tracker, score, is_alive, drift, last_box}
        
        # =========================================================
        # [极简参数配置]
        # =========================================================
        SCALE = 0.5         # 降采样加速
        MAX_TTL = 60        # 高召回：允许盲跑 2 秒
        
        # [内置策略阈值]
        RESET_SCORE_THRES = 0.5  # YOLO分 > 0.7 时强制重置 (防止飘移)
        BORDER_MARGIN = 15       # 边缘禁区 (防止边缘堆积)
        SCORE_DECAY = 0.95       # 盲跑衰减 (解决大框套小框，让旧框变弱)
        MATCH_THRES = 0.3        # 宽松匹配
        
        pbar = tqdm(iter_list, desc=f"Stage 1: {direction}", leave=False)
        
        for i, item in enumerate(pbar):
            # 1. 图像处理
            img_high = cv2.imread(item['path'])
            h_high, w_high = img_high.shape[:2]
            w_small, h_small = int(w_high * SCALE), int(h_high * SCALE)
            img_small = cv2.resize(img_high, (w_small, h_small), interpolation=cv2.INTER_LINEAR)

            # 2. 读取数据
            gt_path = os.path.join(Config.GT_DIR, f"{item['stem']}.xml")
            yolo_path = os.path.join(Config.YOLO_DIR, f"{item['stem']}.xml")
            dets = np.empty((0, 5))
            is_manual = False
            
            if os.path.exists(gt_path):
                dets = read_xml(gt_path)
                if len(dets) > 0: dets[:, 4] = Config.MANUAL_SCORE
                is_manual = True
            elif os.path.exists(yolo_path):
                dets = read_xml(yolo_path)
            
            # -------------------------------------------------------
            # Step 1: 预测更新 (含边缘查杀 & 分数衰减)
            # -------------------------------------------------------
            active_idxs = []
            pred_boxes = []
            
            for idx, trk in enumerate(trackers):
                trk['no_match_count'] += 1
                trk['drift'] += 1  # 【修改点】盲跑一帧，不确定度+1
                
                # [策略] 分数衰减：解决大框套小框的核心
                # 盲跑的框分数会越来越低，在 NMS 和冲突检查中处于劣势
                trk['score'] *= SCORE_DECAY

                if trk['no_match_count'] > MAX_TTL:
                    trk['is_alive'] = False; continue

                ok, bbox_small = trk['tracker'].update(img_small)
                
                if ok:
                    x = bbox_small[0] / SCALE; y = bbox_small[1] / SCALE
                    w = bbox_small[2] / SCALE; h = bbox_small[3] / SCALE
                    x2 = x + w; y2 = y + h
                    
                    # [策略] 边缘查杀：只要碰边，直接死
                    if (x > BORDER_MARGIN and y > BORDER_MARGIN and 
                        x2 < w_high - BORDER_MARGIN and y2 < h_high - BORDER_MARGIN):
                        
                        box = [x, y, x2, y2, trk['score']]
                        trk['last_box'] = box
                        pred_boxes.append(box)
                        active_idxs.append(idx)
                        trk['is_alive'] = True
                    else:
                        trk['is_alive'] = False # 撞墙死
                else:
                    trk['is_alive'] = False

            # -------------------------------------------------------
            # Step 2: 匹配与重置 (高分重置策略)
            # -------------------------------------------------------
            used_det = set()
            if len(active_idxs) > 0 and len(dets) > 0:
                iou = iou_batch(np.array(pred_boxes)[:,:4], dets[:,:4])
                r_ind, c_ind = linear_sum_assignment(-iou)
                
                for r, c in zip(r_ind, c_ind):
                    if iou[r, c] > MATCH_THRES:
                        idx = active_idxs[r]; d_box = dets[c]
                        
                        # [策略] 重置判断
                        # 1. 如果是人工真值 -> 必须重置
                        # 2. 如果是高分检测(>0.7) -> 强制重置 (避免SOT吸附背景)
                        should_reset = is_manual or (d_box[4] > RESET_SCORE_THRES)
                        
                        if should_reset:
                            t_obj = cv2.TrackerCSRT_create() 
                            wb = d_box[2]-d_box[0]; hb = d_box[3]-d_box[1]
                            ibox = (int(d_box[0]*SCALE), int(d_box[1]*SCALE), int(wb*SCALE), int(hb*SCALE))
                            t_obj.init(img_small, ibox)
                            trackers[idx]['tracker'] = t_obj
                            trackers[idx]['last_box'] = d_box # 坐标对齐
                            trackers[idx]['drift'] = 0 # 【修改点】对齐了真值/高分，Drift 归零
                        
                        # 无论是否重置，分数都要回血，计数清零
                        trackers[idx]['score'] = d_box[4]
                        trackers[idx]['no_match_count'] = 0 
                        used_det.add(c)

            # -------------------------------------------------------
            # Step 3: 冲突查杀 & 初始化 (解决嵌套/重叠)
            # -------------------------------------------------------
            # 找出没匹配上的"嫌疑SOT" (可能是飘了，也可能是被套住的旧框)
            suspicious_trackers = []
            for idx, trk in enumerate(trackers):
                if trk.get('is_alive', True) and trk['no_match_count'] > 0 and 'last_box' in trk:
                    suspicious_trackers.append((idx, trk['last_box'][:4]))

            for di in range(len(dets)):
                if di not in used_det:
                    d_box = dets[di]
                    x1, y1, x2, y2 = d_box[:4]

                    # [策略] 边缘禁止出生
                    if (x1 < BORDER_MARGIN or y1 < BORDER_MARGIN or 
                        x2 > w_high - BORDER_MARGIN or y2 > h_high - BORDER_MARGIN):
                        continue 

                    # [策略] 冲突查杀：新 YOLO 来了，挡路的旧 SOT 必须死
                    # 这能解决"大框套小框"：如果新框是大框，它会覆盖旧的小框，若有重叠则杀旧
                    if len(suspicious_trackers) > 0:
                        this_det = np.expand_dims(d_box[:4], 0)
                        other_boxes = np.array([x[1] for x in suspicious_trackers])
                        ious = iou_batch(this_det, other_boxes).flatten()
                        
                        # 只要沾边 (>0.1) 且没匹配上，就认定是干扰项
                        conflict_indices = np.where(ious > 0.1)[0]
                        for c_idx in conflict_indices:
                            real_idx = suspicious_trackers[c_idx][0]
                            trackers[real_idx]['is_alive'] = False 

                    # 初始化
                    t_obj = cv2.TrackerCSRT_create()
                    w_d = x2 - x1; h_d = y2 - y1
                    init_box = (int(x1*SCALE), int(y1*SCALE), int(w_d*SCALE), int(h_d*SCALE))
                    t_obj.init(img_small, init_box)
                    
                    trackers.append({
                        'tracker': t_obj, 'score': d_box[4],
                        'is_alive': True, 'no_match_count': 0, 'last_box': d_box,
                        'drift': 0 # 【修改点】新生的框，Drift为0
                    })

            # -------------------------------------------------------
            # Step 4: 清理 & 输出
            # -------------------------------------------------------
            # 物理删除
            new_trackers = []
            for t in trackers:
                if t.get('is_alive', True):
                    new_trackers.append(t)
                else:
                    del t['tracker']
            trackers = new_trackers
            
            out_candidates = []
            if is_manual:
                for d in dets:
                    out_candidates.append(list(d) + [0])
            else:
                for t in trackers:
                    # [策略] 过滤低分：盲跑太久分太低的，不输出给 NMS，减少干扰
                    if 'last_box' in t and t['score'] > 0.15: 
                        box = list(t['last_box'][:5]) + [t['drift']]
                        out_candidates.append(np.array(box))
            results[item['frame_id']] = np.array(out_candidates) if out_candidates else np.empty((0, 6))

            pbar.set_postfix({"Active": len(trackers)})

        return results

# -----------------------------------------------------------
# 主流水线
# -----------------------------------------------------------
class DARTPipeline:
    def __init__(self):
        if not os.path.exists(Config.OUT_DIR):
            os.makedirs(Config.OUT_DIR)
        if Config.VISUALIZE and not os.path.exists(Config.VIS_DIR):
            os.makedirs(Config.VIS_DIR)
        if Config.SAVE_PERFECT_FRAMES and not os.path.exists(Config.BEST_FRAMES_DIR):
            os.makedirs(Config.BEST_FRAMES_DIR)
        if Config.SAVE_PERFECT_FRAMES and not os.path.exists(Config.BEST_FRAMES_XML_DIR):
            os.makedirs(Config.BEST_FRAMES_XML_DIR)

    def run_sot_process(self, frame_list, direction, return_dict):
        gen = SOTGenerator()
        res = gen.run(frame_list, direction)
        return_dict[direction] = res

    def prepare_dense_detections(self, vname, frame_list):
        fwd_cache_path = os.path.join(Config.CACHE_DIR, f"{vname}_fwd.pkl")
        bwd_cache_path = os.path.join(Config.CACHE_DIR, f"{vname}_bwd.pkl")
        
        fwd_res = None
        bwd_res = None
        
        # 1. 尝试读取缓存
        if Config.USE_CACHE and not Config.FORCE_REBUILD and os.path.exists(fwd_cache_path) and os.path.exists(bwd_cache_path):
            print(f"  [Cache] Found cache for {vname}, loading...")
            try:
                with open(fwd_cache_path, 'rb') as f:
                    fwd_res = pickle.load(f)
                with open(bwd_cache_path, 'rb') as f:
                    bwd_res = pickle.load(f)
                print(f"  [Cache] Loaded successfully! Skipping CSRT & ByteTrack.")
            except Exception as e:
                print(f"  [Cache] Error loading cache: {e}. Re-running...")
        
        if fwd_res is None or bwd_res is None:
            # 2. 缓存不存在 (或强制重跑)，运行原始耗时逻辑
            print(f"  [Processing] Running Stage 1 & 2 (This may take time)...")
            
            manager = Manager()
            return_dict = manager.dict()
            p1 = Process(target=self.run_sot_process, args=(frame_list, "Forward", return_dict))
            # p2 = Process(target=self.run_sot_process, args=(frame_list, "Backward", return_dict))
            p1.start()
            # p2.start()
            p1.join()
            # p2.join()
            fwd_res = return_dict["Forward"]
            bwd_res = {}

            # gen = SOTGenerator()
            # fwd_res = gen.run(frame_list, "Forward")
            # bwd_res = gen.run(frame_list, "Backward")
            # 3. 保存结果到缓存
            if Config.USE_CACHE:
                if not os.path.exists(Config.CACHE_DIR):
                    os.makedirs(Config.CACHE_DIR)
                print(f"  [Cache] Saving results to {cache_path}...")
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'fwd_res': fwd_res,
                        'bwd_res': bwd_res
                    }, f)

        
        if Config.VISUALIZE and not os.path.exists(Config.VIS_DIR_S1):
            os.makedirs(Config.VIS_DIR_S1)

        dense_dets = {}
        all_fids = sorted(frame_list, key=lambda x: x['frame_id'])
        
        for item in tqdm(all_fids, desc="Stage 1: Merging & NMS", leave=False):
            fid = item['frame_id']
            cands_f = fwd_res.get(fid, np.empty((0,6)))
            # cands_b = bwd_res.get(fid, np.empty((0,6)))
            
            # if len(cands_f) > 0 and len(cands_b) > 0:
            #     all_cands = np.vstack((cands_f, cands_b))
            # elif len(cands_f) > 0:
            #     all_cands = cands_f
            # elif len(cands_b) > 0:
            #     all_cands = cands_b
            # else: all_cands = np.empty((0,6))
            
            all_cands = cands_f
            
            if len(all_cands) > 0:
                conf_mask = all_cands[:, 4] > 0.1
                all_cands = all_cands[conf_mask]
                
                keep_con = nms_containment_safe(all_cands[:,:4], all_cands[:,4], iomin_thres=0.8, score_gap=0.6)
                all_cands = all_cands[keep_con]
                
                keep = nms_numpy(all_cands[:,:4], all_cands[:,4], Config.NMS_THRES)
                dense_dets[fid] = all_cands[keep]
            else:
                dense_dets[fid] = np.empty((0,6))

            if Config.VISUALIZE:
                # 注意：频繁读取图片和写入会降低速度，调试完建议注释掉
                img = cv2.imread(item['path'])

                # 1. (可选) 画出被 NMS 过滤掉的框 (灰色)，方便看 CSRT 的原始结果
                for box in all_cands:
                    x1, y1, x2, y2, s = box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1)

                # 2. 画出最终保留的框 (绿色)
                for box in dense_dets[fid]:
                    x1, y1, x2, y2, s = box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # 标记分数
                    cv2.putText(img, f"{s:.2f}", (int(x1), int(y1)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 标记信息
                cv2.putText(img, f"Frame: {fid} | Count: {len(dense_dets[fid])}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                save_path = os.path.join(Config.VIS_DIR_S1, f"{item['stem']}.jpg")
                cv2.imwrite(save_path, img)

        return dense_dets

    def run_tracking(self, frame_list, detections, direction="Forward"):
        tracker = BYTETracker(TrackerArgs(), frame_rate=30)
        results = defaultdict(list) 
        iter_list = frame_list if direction == "Forward" else frame_list[::-1]
        img = cv2.imread(frame_list[0]['path'])
        h, w = img.shape[:2]

        for item in tqdm(iter_list, desc=f"Stage 2: Tracking {direction}", leave=False):
            fid = item['frame_id']
            dets = detections.get(fid, np.empty((0,6)))
            if dets.shape[1] > 5:
                dets = dets[:, :5]
            targets = []
            if len(dets) > 0:
                targets = tracker.update(dets, [h, w], [h, w])
            
            for t in targets:
                tlwh = t.tlwh
                box = np.array([tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]])
                results[t.track_id].append({
                    'frame_id': fid, 'box': box, 'score': t.score
                })
        return results

    def calculate_drift(self, track_dict, direction="Forward"):
        drift_map = {} 
        for tid, nodes in track_dict.items():
            nodes.sort(key=lambda x: x['frame_id'])
            # Score > 0.8 视为锚点 (Manual帧因 Score=2.0 自动成为锚点)
            anchors = [i for i, n in enumerate(nodes) if n['score'] > Config.ANCHOR_CONF_THRES]
            node_drifts = {}
            
            if not anchors:
                for n in nodes: node_drifts[n['frame_id']] = 100.0
            else:
                for i, n in enumerate(nodes):
                    if direction == "Forward":
                        past_anchors = [a for a in anchors if a <= i]
                        dist = (i - past_anchors[-1]) if past_anchors else 100.0
                    else:
                        future_anchors = [a for a in anchors if a >= i]
                        dist = (future_anchors[0] - i) if future_anchors else 100.0
                    node_drifts[n['frame_id']] = float(dist)
            drift_map[tid] = node_drifts
        return drift_map

    def calculate_consistency_rate(self, raw_boxes, final_boxes, threshold=0.7):
        """
        计算 CSRT 原始框(Raw) 在 最终结果(Final) 中的保留率。
        使用 IoMin (交集/小框面积) 来容忍框的收缩。
        """
        N = len(raw_boxes)
        M = len(final_boxes)
        
        if N == 0: return 1.0 if M == 0 else 0.0 # 都没有是一致，否则不一致
        if M == 0: return 0.0 # 有原始框但没结果，保留率为0
        
        # 1. 扩展维度以进行广播计算 (N, 1, 4) vs (1, M, 4)
        # raw: [x1, y1, x2, y2]
        raw = np.expand_dims(raw_boxes[:, :4], 1) 
        final = np.expand_dims(final_boxes[:, :4], 0)
        
        # 2. 批量计算交集 (Intersection)
        xx1 = np.maximum(raw[..., 0], final[..., 0])
        yy1 = np.maximum(raw[..., 1], final[..., 1])
        xx2 = np.minimum(raw[..., 2], final[..., 2])
        yy2 = np.minimum(raw[..., 3], final[..., 3])
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        inter = w * h
        
        # 3. 批量计算自身面积
        area_raw = (raw[..., 2] - raw[..., 0]) * (raw[..., 3] - raw[..., 1])
        area_final = (final[..., 2] - final[..., 0]) * (final[..., 3] - final[..., 1])
        
        # 4. 计算 IoMin (交集 / 较小面积) -> 容忍收缩
        # 如果 CSRT 框很大，Final 框很小但在里面，IoMin 会很大 -> 匹配成功
        min_area = np.minimum(area_raw, area_final)
        io_min_matrix = inter / (min_area + 1e-6) # 形状 (N, M)
        
        # 5. 统计匹配率
        # axis=1 表示对每一个 raw 框，找它在所有 final 框里的最高分
        best_matches = np.max(io_min_matrix, axis=1) 
        
        # 只要最高分超过阈值，就算"被留下了"
        matched_count = np.sum(best_matches > threshold)
        
        return matched_count / N

    def is_x_axis_continuous(self, final_objs, img_width):
        """
        通过 X 轴投影覆盖率判定该帧是否异常
        """
        if len(final_objs) < 10: return False

        # 1. 获取所有框在 X 轴上的线段 [xmin, xmax]
        intervals = []
        for obj in final_objs:
            intervals.append([obj['box'][0], obj['box'][2]])
        
        # 按左端点排序
        intervals.sort(key=lambda x: x[0])

        # 2. 合并重叠线段 (Merge Intervals)
        merged = []
        if not intervals: return False
        
        curr = intervals[0]
        for i in range(1, len(intervals)):
            if intervals[i][0] <= curr[1]: # 有重叠
                curr[1] = max(curr[1], intervals[i][1])
            else: # 断开了
                merged.append(curr)
                curr = intervals[i]
        merged.append(curr)

        # 3. 分析合并后的区间
        # 按区间长度排序，找最长的（主列）
        merged.sort(key=lambda x: x[1] - x[0], reverse=True)
        main_interval = merged[0]
        main_len = main_interval[1] - main_interval[0]

        # 阈值设置：基于图像宽度或物体平均宽度
        # 假设麦穗平均宽度约 100-150px (4K图)
        MAX_GAP_ALLOWED = img_width * 0.05 # 允许最大 5% 宽度的断裂
        MIN_MAIN_COL_WIDTH = img_width * 0.15 # 主列至少要占 15% 宽度

        # 判定 A: 主列是否太窄
        if main_len < MIN_MAIN_COL_WIDTH:
            return False

        # 判定 B: 是否存在严重的侧边干扰 (Gap 检查)
        # 如果有多个区间，检查主区间与其他区间之间的空隙
        if len(merged) > 1:
            for i in range(1, len(merged)):
                other = merged[i]
                # 计算两个区间之间的距离
                gap = 0
                if other[0] > main_interval[1]:
                    gap = other[0] - main_interval[1]
                elif main_interval[0] > other[1]:
                    gap = main_interval[0] - other[1]
                
                # 如果存在一个较大的干扰块且距离主列很远，说明该帧不干净
                if (other[1] - other[0]) > (img_width * 0.02) and gap > MAX_GAP_ALLOWED:
                    return False

        return True

    # ================= [新增] YOLO 保留率计算 =================
    def calculate_yolo_retention(self, raw_boxes, final_objs, iou_thresh=0.25):
        """
        计算 YOLO 原始检测框在最终结果中的保留率。
        """
        if raw_boxes is None or len(raw_boxes) == 0: return 1.0 # YOLO 没框，默认保留 100%
        if not final_objs: return 0.0 # YOLO 有框但 DART 没框，保留率 0%
            
        raw_locs = raw_boxes[:, :4]
        final_locs = np.array([x['box'] for x in final_objs])
        
        # 批量计算 IoU (Row: Final, Col: Raw) -> 输出 Shape (N_final, N_raw)
        iou_matrix = iou_batch(final_locs, raw_locs) 
        
        # 对每一个 Raw 框（列），找被 Final 匹配的最大 IoU
        max_ious_per_raw = np.max(iou_matrix, axis=0) 
        
        # 统计被保留的比例
        retained_count = np.sum(max_ious_per_raw > iou_thresh)
        
        return retained_count / len(raw_boxes)
    # ==========================================================

    def dart_fusion(self, vname, frame_list, fwd_tracks, bwd_tracks, dense_dets):
        # 1. 准备漂移数据
        fwd_drift = self.calculate_drift(fwd_tracks, "Forward")
        # bwd_drift = self.calculate_drift(bwd_tracks, "Backward")
        
        h, w = cv2.imread(frame_list[0]['path']).shape[:2]
        
        # 2. 建立索引
        fwd_by_frame = defaultdict(list)
        for tid, nodes in fwd_tracks.items():
            for n in nodes: fwd_by_frame[n['frame_id']].append({**n, 'tid': tid})
        bwd_by_frame = defaultdict(list)
        for tid, nodes in bwd_tracks.items():
            for n in nodes: bwd_by_frame[n['frame_id']].append({**n, 'tid': tid})
            
        vw = None
        if Config.VISUALIZE:
             vw = cv2.VideoWriter(os.path.join(Config.VIS_DIR, f"{vname}_dart_final.mp4"),
                                  cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        print(f"  > Stage 4: Fusion & Rescue for {vname}...")
        
        # [新增] 综合不确定性计算函数 (0.0~1.0, 越低越好)
        def get_uncertainty(iou_score, conf_score, drift):
            # 单向版：没有 iou_score (设为0)，主要靠 drift 和 conf
            # 给一个基础惩罚 0.1，因为没有双向验证
            return 0.1 + 0.4 * (1.0 - conf_score) + 0.5 * (min(drift, 30.0) / 30.0)

        # 【改动 3：定义困难度计算函数】-----------------------------------------
        def calculate_hardness(final_objs, raw_dets):
            if not final_objs: return 0.0
            # 原始没框，全是救回来的 -> 极难
            if raw_dets is None or len(raw_dets) == 0: return float(len(final_objs))
            
            fin_boxes = np.array([x['box'] for x in final_objs])
            # 计算 IoU
            iou_matrix = iou_batch(fin_boxes, raw_dets[:, :4])
            max_ious = np.max(iou_matrix, axis=1) # 每个最终框对应的最大IoU
            
            # 权重：漏检(IoU<0.1)=1.0, 精修(0.1<IoU<0.8)=0.5, 简单(IoU>0.8)=0
            n_rescue = np.sum(max_ious < 0.1)
            n_refine = np.sum((max_ious >= 0.1) & (max_ious < 0.8))
            return (n_rescue * 1.0) + (n_refine * 0.5)
        # ---------------------------------------------------------------------

        gt_fids = set()
        for item in frame_list:
            gt_path = os.path.join(Config.GT_DIR, f"{item['stem']}.xml")
            if os.path.exists(gt_path):
                gt_fids.add(item['frame_id'])
        gt_fids_list = sorted(list(gt_fids))
        
        print(gt_fids_list)
        
        MIN_GAP_FRAMES = 1
        last_anchor_fid = -999

        for item in tqdm(frame_list, desc="Fusing", leave=False):
            fid = item['frame_id']
            
            # --- Manual 帧 (保持不变) ---
            if fid in gt_fids:
                last_anchor_fid = fid
                # (略: 保持原有人工帧处理逻辑)
                continue

            # --- DART 融合逻辑 (改写为求并集) ---
            f_objs = fwd_by_frame[fid]
            # b_objs = bwd_by_frame[fid]
            final_objs = []
            
            # used_f = set(); used_b = set()
            
            # # A. 处理交集 (Intersection) -> 高精度
            # if f_objs and b_objs:
            #     f_boxes = np.array([x['box'] for x in f_objs])
            #     b_boxes = np.array([x['box'] for x in b_objs])
            #     iou = iou_batch(f_boxes, b_boxes)
            #     r_ind, c_ind = linear_sum_assignment(-iou)
                
            #     for r, c in zip(r_ind, c_ind):
            #         current_iou = iou[r, c]

            #         # 获取基本信息
            #         box_f = f_objs[r]['box']
            #         box_b = b_objs[c]['box']
                    
            #         # ---------------------------------------------------------
            #         # 1. 计算面积与 IoMin (只算一次)
            #         # ---------------------------------------------------------
            #         w_f = box_f[2] - box_f[0]; h_f = box_f[3] - box_f[1]
            #         area_f = w_f * h_f
                    
            #         w_b = box_b[2] - box_b[0]; h_b = box_b[3] - box_b[1]
            #         area_b = w_b * h_b
                    
            #         # 计算交集面积
            #         xx1 = max(box_f[0], box_b[0]); yy1 = max(box_f[1], box_b[1])
            #         xx2 = min(box_f[2], box_b[2]); yy2 = min(box_f[3], box_b[3])
            #         w_inter = max(0, xx2 - xx1); h_inter = max(0, yy2 - yy1)
            #         area_inter = w_inter * h_inter
                    
            #         # IoMin
            #         min_area = min(area_f, area_b)
            #         io_min = area_inter / (min_area + 1e-6)
                    
            #         # ---------------------------------------------------------
            #         # 2. 融合决策
            #         # ---------------------------------------------------------
            #         # 准入条件：IoU合格 OR 包含关系显著
            #         if current_iou > 0.5 or io_min > 0.8:
            #             df = fwd_drift[f_objs[r]['tid']].get(fid, 100.0)
            #             db = bwd_drift[b_objs[c]['tid']].get(fid, 100.0)
                        
            #             # [紧致化融合 Tight Fusion]
            #             # 直接使用上面算好的面积
            #             ratio = area_f / (area_b + 1e-6)
                        
            #             if ratio > 1.3: 
            #                 final_box = box_b # F太大(可能是泥土)，强制选B
            #             elif ratio < 0.7: 
            #                 final_box = box_f # B太大，强制选F
            #             else:
            #                 # 大小差不多，维纳过程加权
            #                 w_f = 1.0 / (df + 1e-6); w_b = 1.0 / (db + 1e-6)
            #                 final_box = (w_f * box_f + w_b * box_b) / (w_f + w_b)
                        
            #             final_score = max(f_objs[r]['score'], b_objs[c]['score'])
            #             min_drift = min(df, db)
                        
            #             # [关键修复] 计算不确定性时，承认"包含"也是一种高度一致
            #             # 如果是 IoU=0.2 但 IoMin=0.9，我们应该认为一致性是 0.9
            #             effective_consistency = max(current_iou, io_min)
                        
            #             unc = get_uncertainty(effective_consistency, final_score, min_drift)
                        
            #             # 奖励：如果一致性极高，直接给完美分
            #             if effective_consistency > 0.9: unc = 0.05
                        
            #             final_objs.append({
            #                 'box': final_box, 
            #                 'score': final_score, 
            #                 'id': f_objs[r]['tid'], 
            #                 'div': unc
            #             })
            #             used_f.add(r); used_b.add(c)

            # B. 处理差集 (Union) -> 高召回
            # 也就是"单侧补全"，只保留靠谱的
            
            # Forward 剩余
            for i, obj in enumerate(f_objs):
                # if i not in used_f:
                df = fwd_drift[obj['tid']].get(fid, 100.0)
                # 筛选：漂移不能太离谱 (<30帧)
                if df < Config.TRUST_LIMIT:
                    # 单侧意味着 IoU=0 (一致性极差)，所以不确定性会偏高
                    # 但如果 drift 很小 (刚出现/刚跟丢)，我们给它打折
                    base_unc = get_uncertainty(0.0, obj['score'], df)
                    if df <= 10: base_unc = 0.1 # 免死金牌：刚出现的算准的
                    
                    final_objs.append({
                        'box': obj['box'], 'score': obj['score'], 
                        'id': obj['tid'], 'div': base_unc
                    })

            # # Backward 剩余
            # for i, obj in enumerate(b_objs):
            #     if i not in used_b:
            #         db = bwd_drift[obj['tid']].get(fid, 100.0)
            #         if db < Config.TRUST_LIMIT:
            #             base_unc = get_uncertainty(0.0, obj['score'], db)
            #             if db <= 10: base_unc = 0.1
                        
            #             final_objs.append({
            #                 'box': obj['box'], 'score': obj['score'], 
            #                 'id': obj['tid'], 'div': base_unc
            #             })

            # C. 最终去重 (Final NMS)
            # 因为求了并集，可能存在正反向没匹配上但物理重叠的框，需要清理
            if len(final_objs) > 0:
                # 准备数据
                boxes = np.array([x['box'] for x in final_objs])
                scores = np.array([1.0 - x['div'] for x in final_objs]) # 分数 = 确定性
                
                # --- 第一道防线：包含去重 (杀大框套小框) ---
                keep_con = nms_containment_safe(boxes, scores, iomin_thres=0.75, score_gap=0.6)
                final_objs = [final_objs[k] for k in keep_con]
                
                # --- [新增] 第二道防线：中心点距离去重 (杀错位重叠) ---
                if len(final_objs) > 0:
                    boxes = np.array([x['box'] for x in final_objs])
                    scores = np.array([1.0 - x['div'] for x in final_objs])
                    
                    # 阈值 30px (针对4K图)。如果在1080P下请改为 15。
                    # 只要中心点距离小于这个值，不管长什么样，杀掉分低的。
                    keep_dist = nms_center_distance(boxes, scores, dist_thres=30)
                    final_objs = [final_objs[k] for k in keep_dist]

                # -----------------------------------------------------
                # Step 2: [新增] 智能分数排斥修正 (Repulsion)
                # -----------------------------------------------------
                if len(final_objs) > 0:
                    # 重新提取坐标和分数
                    boxes = np.array([x['box'] for x in final_objs])
                    scores = np.array([1.0 - x['div'] for x in final_objs])
                    
                    # 执行排斥：修改 boxes 的坐标，让它们互不侵犯
                    refined_boxes, refined_scores, valid_mask = refine_boxes_smart_repulsion(boxes, scores, iou_ceiling=0.5)

                    final_objs = [obj for i, obj in enumerate(final_objs) if valid_mask[i]]
                    # 因为坐标变了，需要更新 final_objs
                    # 注意：这里我们原地修改 box 坐标，保留其他属性
                    for i, obj in enumerate(final_objs):
                        obj['box'] = refined_boxes[i]

                # --- 第三道防线：标准 NMS (杀普通重叠) ---
                if len(final_objs) > 0:
                    boxes = np.array([x['box'] for x in final_objs])
                    scores = np.array([1.0 - x['div'] for x in final_objs])
                    
                    keep = nms_numpy(boxes, scores, Config.NMS_THRES) # 建议 0.5
                    final_objs = [final_objs[k] for k in keep]

            # if len(final_objs) > 0:
            #     final_objs = self.clean_edges_conservative(final_objs, w)

            # 输出 XML
            if final_objs:
                # h, w = cv2.imread(item['path']).shape[:2]

                for obj in final_objs:
                    b = obj['box']
                    # 限制 x1, y1 不小于 0
                    b[0] = max(0.0, b[0])
                    b[1] = max(0.0, b[1])
                    # 限制 x2, y2 不超过图像宽高
                    b[2] = min(float(w), b[2])
                    b[3] = min(float(h), b[3])
                    obj['box'] = b

                # write_xml(final_objs, item['path'], os.path.join(Config.OUT_DIR, f"{item['stem']}.xml"), (h,w,3))

            # # 完美帧筛选 (基于不确定性)
            # if Config.SAVE_PERFECT_FRAMES and final_objs:
            #     # 1. 准备数据 (转为 numpy 数组)
            #     final_boxes_arr = np.array([x['box'] for x in final_objs])
            #     raw_boxes_arr = dense_dets.get(fid, np.empty((0,5)))

            #     # 2. 计算内部质量 (不确定性)
            #     divs = [x['div'] for x in final_objs]
            #     avg_div = sum(divs) / len(divs)
                
            #     # 3. [调用] 计算外部一致性 (保留率)
            #     # 直接一行代码搞定，不需要自己写循环
            #     consistency_rate = self.calculate_consistency_rate(raw_boxes_arr, final_boxes_arr, threshold=0.9)
                
            #     # 4. 联合筛选
            #     # 条件: 融合质量高 (avg_div < 0.2) 且 CSRT框保留率高 (rate > 0.95)
            #     if (len(final_objs) > 5) and (avg_div < 0.1) and (consistency_rate > 0.98):
            #         img = cv2.imread(item['path'])
                    
            #         # 画框
            #         for o in final_objs:
            #             x1,y1,x2,y2 = map(int, o['box'])
            #             # 绿色=好, 红色=差
            #             clr = (0,255,0) if o['div'] < 0.15 else (0,0,255)
            #             cv2.rectangle(img, (x1,y1), (x2,y2), clr, 2)
            #             cv2.putText(img, f"{o['score']:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)
                    
            #         # 打印信息 Q=Quality, C=Consistency
            #         info = f"Q:{1-avg_div:.2f} C:{consistency_rate:.2f}"
            #         cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
            #         # 保存
            #         if 300 < item['frame_id'] < frame_list[-1]['frame_id'] - 300:
            #             cv2.imwrite(os.path.join(Config.BEST_FRAMES_DIR, f"{item['stem']}_perfect.jpg"), img)
            #             write_xml(final_objs, item['path'], os.path.join(Config.BEST_FRAMES_XML_DIR, f"{item['stem']}.xml"), (h, w, 3))


            if Config.SAVE_PERFECT_FRAMES and final_objs and dense_dets.get(fid) is not None:
                
                # 1. 基础时空过滤 (Timeline & Spacing)
                # -------------------------------------------------
                dist_prev = fid - last_anchor_fid
                dist_next = 9999
                for g_fid in gt_fids_list:
                    if g_fid > fid:
                        dist_next = g_fid - fid
                        break
                
                is_timeline_safe = 300 < fid < (frame_list[-1]['frame_id'] - 300)
                # 间隔保持 15 帧 (0.5s)
                is_spacing_safe = (dist_prev >= 15) and (dist_next >= 15)

                if is_timeline_safe and is_spacing_safe:
                    
                    yolo_xml_path = os.path.join(Config.YOLO_DIR, f"{item['stem']}.xml")
                    raw_yolo = read_xml(yolo_xml_path)
                    retention = self.calculate_yolo_retention(raw_yolo, final_objs, iou_thresh=0.25)
                    
                    if retention < 0.913:
                        # 丢弃该帧：说明 DART 过滤掉了太多 YOLO 认为存在的物体
                        continue 
                    # ==========================================
                    
                    raw_high_conf = raw_yolo[raw_yolo[:, 4] > 0.6] 
                    
                    fin_boxes = np.array([x['box'] for x in final_objs])
                    
                    # 默认为 Rescue (假设全没匹配上)
                    obj_ious = np.zeros(len(final_objs))
                    
                    if len(raw_high_conf) > 0 and len(fin_boxes) > 0:
                        # 计算每个最终框 vs 原始检测框的最大 IoU
                        iou_matrix = iou_batch(fin_boxes, raw_high_conf)
                        obj_ious = np.max(iou_matrix, axis=1) 
                    
                    # 3. 统计关键指标
                    # -------------------------------------------------
                    # [核心] Rescue: IoU < 0.1 且 DART 很确信 (div < 0.15)
                    # 我们不仅要求是救援框，还要求这个救援框是靠谱的，不是飘来的
                    n_valid_rescue = 0
                    for i, o in enumerate(final_objs):
                        if obj_ious[i] < 0.1 and o['div'] < 0.2:
                            n_valid_rescue += 1

                    # Refine: 0.1 <= IoU < 0.85 (大幅修正)
                    n_refine = np.sum((obj_ious >= 0.1) & (obj_ious < 0.85))
                    
                    # 整体质量
                    avg_div = sum(x['div'] for x in final_objs) / len(final_objs)
                    max_div = max(x['div'] for x in final_objs)

                    # 4. 最终严苛筛选 (Strict Filter)
                    # -------------------------------------------------
                    # 条件 A: 必须有至少 1 个靠谱的救援框 (红色)
                    # 条件 B: 或者有 5 个以上被大幅修正的框 (黄色) -> 这种情况也很有价值，证明原模型位置歪了
                    # 条件 C: 整体不能太乱 (avg_div < 0.15)
                    
                    has_high_value = (n_valid_rescue >= 1) or (n_refine >= 2)
                    is_clean = (avg_div < 0.30) and (max_div < 0.35) and (len(final_objs) >= 10)

                    is_spatially_consistent = self.is_x_axis_continuous(final_objs, w)
                    
                    is_stable = (avg_div < 0.30) and (retention > 0.85)

                    if (has_high_value or is_stable) and is_clean and is_spatially_consistent:
                        # ---> 满足条件，保存！ <---
                        
                        base_name = f"{item['stem']}"
                        img_save = cv2.imread(item['path'])
                        
                        # 绘制用于人工复核的图片
                        for i, o in enumerate(final_objs):
                            x1, y1, x2, y2 = map(int, o['box'])
                            iou = obj_ious[i]
                            
                            # 配色方案
                            if iou < 0.1: 
                                color = (0, 0, 255); thickness = 2 # 🔴 Rescue
                            elif iou < 0.85: 
                                color = (0, 215, 255); thickness = 2 # 🟡 Refine
                            else: 
                                color = (0, 255, 0); thickness = 1 # 🟢 Easy
                                
                            cv2.rectangle(img_save, (x1, y1), (x2, y2), color, thickness)

                        # 在图上标记为什么选这张图
                        trigger_reason = "Rescue" if n_valid_rescue >= 1 else "Refine"
                        info = f"Type:{trigger_reason} Rescue:{n_valid_rescue} Refine:{n_refine}"
                        cv2.putText(img_save, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        cv2.imwrite(os.path.join(Config.BEST_FRAMES_DIR, f"{base_name}.jpg"), img_save)
                        write_xml(final_objs, item['path'], os.path.join(Config.BEST_FRAMES_XML_DIR, f"{base_name}.xml"), (h, w, 3))
                        
                        # 更新锚点，强制冷却 15 帧
                        last_anchor_fid = fid

            # =========================================================================

            # 可视化
            if vw:
                img = cv2.imread(item['path'])
                for o in final_objs:
                    x1,y1,x2,y2 = map(int, o['box'])
                    clr = (0, int(255*(1-min(o['div'],1))), int(255*min(o['div'],1)))
                    cv2.rectangle(img, (x1,y1), (x2,y2), clr, 2)
                    cv2.putText(img, f"ID:{o['id']}", (x1,y1-5), 0, 0.6, clr, 2)
                vw.write(img)

        if vw:
            vw.release()

    # -------------------------------------------------------------------------
    # [新增] 缓存核心逻辑：尝试读取，读不到就跑算法并保存
    # -------------------------------------------------------------------------
    def get_tracks_with_cache(self, vname, frames):
        
        # (A) Stage 1: 混合 SOT 生成 (耗时大户)
        dense_dets = self.prepare_dense_detections(vname, frames)
        
        # (B) Stage 2: 双向 ByteTrack
        fwd_tracks = self.run_tracking(frames, dense_dets, "Forward")
        # bwd_tracks = self.run_tracking(frames, dense_dets, "Backward")
        bwd_tracks = {}

        return fwd_tracks, bwd_tracks, dense_dets

    # -------------------------------------------------------------------------
    # [修改] 主流程 process 函数
    # -------------------------------------------------------------------------
    def process(self):
        # file_names = ['DJI_0347', 'DJI_0350', 'DJI_0353', 'DJI_0354', 'DJI_0356', 'DJI_0357', 'DJI_0358',
        #             'DJI_0385', 'DJI_0386', 'DJI_0387', 'DJI_0388', 'DJI_0389', 'DJI_0391', 'DJI_0392',
        #             'DJI_0393', 'DJI_0394', 'DJI_0395', 'DJI_0396', 'DJI_0418', 'DJI_0419', 'DJI_0420',
        #             'DJI_0421', 'DJI_0447', 'DJI_0448', 'DJI_0449', 'DJI_0450', 'DJI_0451', 'DJI_0452',
        #             'DJI_0453', 'DJI_0454', 'DJI_0455', 'DJI_0456', 'DJI_0457', 'DJI_0458', 'DJI_0459',
        #             'DJI_0460', 'DJI_0476']

        # train
        file_names = ['DJI_0449', 'DJI_0389', 'DJI_0394', 'DJI_0350', 'DJI_0420',
                'DJI_0387', 'DJI_0358', 'DJI_0386', 'DJI_0395', 'DJI_0391',
                'DJI_0356', 'DJI_0354', 'DJI_0347', 'DJI_0385', 'DJI_0419',
                'DJI_0392', 'DJI_0396', 'DJI_0456', 'DJI_0447', 'DJI_0460',
                'DJI_0453', 'DJI_0451', 'DJI_0459', 'DJI_0448', 'DJI_0455']

        files = sorted(glob.glob(os.path.join(Config.IMG_DIR, "*.jpg")))
        video_dict = defaultdict(list)
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            import re
            match = re.search(r'(\d+)$', stem)
            if match:
                fid = int(match.group(1))
                vname = stem[:-len(match.group(1))].rstrip('_') or "default"
                if vname[:-6] not in file_names:
                    continue
                video_dict[vname].append({"path": f, "frame_id": fid, "stem": stem})

        for vname, frames in video_dict.items():
            frames.sort(key=lambda x: x['frame_id'])
            print(f"\n>>> Processing Video: {vname} ({len(frames)} frames)")

            # [修改] 调用缓存逻辑获取轨迹
            fwd_tracks, bwd_tracks, dense_dets = self.get_tracks_with_cache(vname, frames)

            # [修改] 直接进行融合 (Stage 3 & 4)
            self.dart_fusion(vname, frames, fwd_tracks, bwd_tracks, dense_dets)

    def process_ori(self):
        file_names = ['DJI_0347', 'DJI_0350', 'DJI_0353', 'DJI_0354', 'DJI_0356', 'DJI_0357', 'DJI_0358',
                    'DJI_0385', 'DJI_0386', 'DJI_0387', 'DJI_0388', 'DJI_0389', 'DJI_0391', 'DJI_0392',
                    'DJI_0393', 'DJI_0394', 'DJI_0395', 'DJI_0396', 'DJI_0418', 'DJI_0419', 'DJI_0420',
                    'DJI_0421', 'DJI_0449', 'DJI_0450', 'DJI_0476']
        files = sorted(glob.glob(os.path.join(Config.IMG_DIR, "*.jpg")))
        video_dict = defaultdict(list)
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            import re
            match = re.search(r'(\d+)$', stem)
            if match:
                fid = int(match.group(1))
                vname = stem[:-len(match.group(1))].rstrip('_') or "default"
                if vname[:-6] not in file_names:
                    continue
                video_dict[vname].append({"path": f, "frame_id": fid, "stem": stem})

        for vname, frames in video_dict.items():
            frames.sort(key=lambda x: x['frame_id'])
            print(f"\n>>> Processing Video: {vname[:-6]} ({len(frames)} frames)")
            
            dense_dets = self.prepare_dense_detections(frames)
            fwd_tracks = self.run_tracking(frames, dense_dets, "Forward")
            bwd_tracks = self.run_tracking(frames, dense_dets, "Backward")
            self.dart_fusion(vname, frames, fwd_tracks, bwd_tracks, dense_dets)

if __name__ == "__main__":
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    p = DARTPipeline()
    p.process()
