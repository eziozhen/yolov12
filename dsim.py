import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# =============================================================================
# [配置区域]
# =============================================================================
class Config:
    # ------------------- 路径配置 -------------------
    ROOT_DIR = "/home/caozhenzhen/xiaomai/yolov5-pytorch-main/"  # 修改为你的路径
    
    IMAGE_DIR_NAME = "VOCdevkit/VOC2007/JPEGImages"
    LABEL_DIR_NAME = "VOCdevkit/VOC2007/Annotations"
    TXT_FILE_NAME = "VOCdevkit/VOC2007/ImageSets/Main/val.txt"
    OUTPUT_DIR_NAME = "output_dynamic" # 输出文件夹改个名区分一下
    
    # ------------------- [1] 缺苗判定参数 -------------------
    # 判定阈值：物理空隙 > 平均株高 * 此倍数
    # 1.5倍：只要能塞下1.5棵树，就算缺苗。
    # 这是一个比较保守但准确的数值，既不敏感也不迟钝。
    GAP_RATIO = 1.5
    
    # ------------------- [2] 头尾对齐参数 -------------------
    # 如果一行的起点比全场平均起点晚了多少像素，算缺苗？
    # 建议设为 1.5 倍平均株距
    HEAD_TAIL_TOLERANCE = 1.5
    
    # ------------------- [3] 碰撞检测 (防误报) -------------------
    # 虚拟盒子缩放
    VIRTUAL_BOX_SCALE = 0.7
    
    # ------------------- 颜色 -------------------
    LANE_COLORS = [
        (255, 0, 0), (0, 200, 0), (0, 0, 255), (200, 200, 0), 
        (0, 200, 200), (200, 0, 200), (100, 100, 100)
    ]
    MISSING_COLOR = (0, 0, 255)

    @classmethod
    def get_path(cls, *args):
        return os.path.normpath(os.path.join(cls.ROOT_DIR, *args))

class ForcedAlignmentDetector:
    def __init__(self):
        pass
        
    def parse_xml(self, xml_path):
        try:
            tree = ET.parse(xml_path)
        except:
            return []
        root = tree.getroot()
        size_node = root.find('size')
        width = int(size_node.find('width').text)
        height = int(size_node.find('height').text)
        boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if not bndbox: continue
            boxes.append([
                float(bndbox.find('xmin').text),
                float(bndbox.find('ymin').text),
                float(bndbox.find('xmax').text),
                float(bndbox.find('ymax').text)
            ])
        return np.array(boxes), width, height

    def find_optimal_angle(self, boxes):
        """矫正角度"""
        if len(boxes) < 2: return 0.0
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        bias = np.mean(centers, axis=0)
        def cost(a):
            ar = np.radians(a)
            c, s = np.cos(ar), np.sin(ar)
            rot_x = (centers[:, 0] - bias[0])*c - (centers[:, 1] - bias[1])*s
            # 使用差分和最小化，让行更紧凑
            sx = np.sort(rot_x)
            diffs = np.diff(sx)
            return np.sum(diffs[:int(len(diffs)*0.8)])
        res = minimize_scalar(cost, bounds=(-60, 60), method='bounded')
        return res.x

    def get_rotated_data(self, boxes, angle):
        """获取旋转属性"""
        if len(boxes) == 0: return []
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        img_center = np.mean(centers, axis=0)
        
        angle_rad = np.radians(angle)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        
        props = []
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box
            w = xmax - xmin
            h = ymax - ymin
            
            # 旋转中心
            c_orig = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
            rot_c = np.dot(c_orig - img_center, R.T) + img_center
            
            # 旋转上下物理边缘
            p_top = np.dot(np.array([c_orig[0], ymin]) - img_center, R.T) + img_center
            p_bottom = np.dot(np.array([c_orig[0], ymax]) - img_center, R.T) + img_center
            
            props.append({
                'id': i,
                'orig_box': box,
                'w': w,
                'h': h,
                'rot_x': rot_c[0],
                'rot_y': rot_c[1],
                'rot_top': p_top[1], 
                'rot_bottom': p_bottom[1]
            })
        return props

    def calculate_global_angle(self, boxes):
        """
        [核心升级] 使用霍夫变换 (Hough Transform) 计算全局垄行倾斜角。
        不管苗怎么缺，只要整体是成行的，这个方法就能把角度算准。
        """
        if len(boxes) < 5: return 0.0
        
        # 1. 创建一张空的“点图”
        # 映射到一个 1000x1000 的虚拟画布上，方便计算
        canvas_size = 1000
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        
        # 归一化坐标到画布
        xs = boxes[:, 0::2].mean(axis=1) # center x
        ys = boxes[:, 1::2].mean(axis=1) # center y
        
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        
        if max_x == min_x or max_y == min_y: return 0.0
        
        # 把点画上去
        for x, y in zip(xs, ys):
            nx = int((x - min_x) / (max_x - min_x) * (canvas_size - 10)) + 5
            ny = int((y - min_y) / (max_y - min_y) * (canvas_size - 10)) + 5
            # 画大一点的圆，让霍夫变换更容易捕捉到“线”
            cv2.circle(mask, (nx, ny), 4, 255, -1)
            
        # 2. 霍夫线变换
        # 精度：1度，1像素；阈值：看点的数量，至少要有10%的点共线
        threshold = max(5, int(len(boxes) * 0.1))
        lines = cv2.HoughLines(mask, 1, np.pi / 180, threshold)
        
        if lines is None: return 0.0
        
        # 3. 统计所有检测到的线的角度
        angles = []
        for line in lines:
            rho, theta = line[0]
            # theta 是弧度，0是垂直线，pi/2是水平线
            # 我们要的是偏离水平的角度
            # 转换为角度 (0-180)
            deg = np.degrees(theta)
            
            # 霍夫变换出来的水平线角度通常接近 90 度
            # 我们将其转换为相对于水平线 (0度) 的偏离角 (-90 到 90)
            diff = deg - 90
            
            # 过滤掉太离谱的角度（比如检测到了竖线），只看水平附近的 (-45 到 45)
            if abs(diff) < 45:
                angles.append(diff)
                
        if not angles: return 0.0
        
        # 取中位数，消除噪点干扰
        return np.median(angles)

    def assign_lanes_forced(self, raw_boxes):
        """
        [强力分行版] 解决两行合并为一行的问题
        
        核心逻辑：
        1. 按照 Y 轴坐标对所有框进行排序。
        2. 动态计算“换行阈值”：使用全图苗高度的 0.6 倍。
           (如果两个框 Y 轴距离超过 0.6 个苗高，判定为换行)
        3. 单次扫描切分，不再使用复杂的聚类算法，避免参数敏感。
        """
        if len(raw_boxes) == 0: return {}
        
        # 1. 准备数据：计算中心点和高度
        # data =List of [id, cx, cy, w, h]
        # 注意：这里我们假设 raw_boxes 格式是 [xmin, ymin, xmax, ymax]
        # 如果你的格式不同，请相应调整
        items = []
        for i, box in enumerate(raw_boxes):
            xmin, ymin, xmax, ymax = box[:4]
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            items.append({'id': i, 'cx': cx, 'cy': cy, 'w': w, 'h': h, 
                          'rot_x': cx, 'rot_y': cy}) # 兼容后续逻辑的键名

        # 2. 计算全局统计量 (用于确定切分阈值)
        all_heights = [p['h'] for p in items]
        if not all_heights: return {}
        
        # 使用 75分位数高度作为参考 (大苗标准)
        median_h = np.percentile(all_heights, 75)
        
        # [核心参数] 换行阈值
        # 经验值：0.6 倍苗高。
        # 同一行的苗，Y中心抖动很难超过 0.6个高度。
        # 如果超过了，大概率是下一行。
        split_threshold = median_h * 0.6

        # 3. 按 Y 轴中心排序 (从上到下)
        items.sort(key=lambda x: x['cy'])
        
        lanes = {}
        current_lane_id = 1
        
        if not items: return {}
        
        # 初始化第一行
        lanes[current_lane_id] = [items[0]]
        last_y = items[0]['cy']
        
        # 4. 扫描并切分
        for i in range(1, len(items)):
            curr = items[i]
            diff = curr['cy'] - last_y
            
            if diff > split_threshold:
                # 距离太远，判定为新的一行
                current_lane_id += 1
                lanes[current_lane_id] = []
                # 重置基准Y (新行的第一个苗)
                last_y = curr['cy'] 
            else:
                # 距离很近，判定为同一行
                # 更新 last_y：使用滑动平均或直接用当前的，这里用当前的可以让行稍微弯曲
                # 但为了防止漂移，最好还是跟这一行的均值比，或者只在换行时重置。
                # 简单且鲁棒的方法：只有换行才重置 last_y？
                # 不，为了适应稍微倾斜的行，我们应该更新 last_y，但不能更新太快。
                # 这里采用：如果归入当前行，就不更新 last_y (以行首为准)，
                # 或者更新 last_y (允许行倾斜)。
                # 考虑到 user 说 raw_boxes 是 rectify 过的 (Angle=0)，行是平的。
                # 所以我们 **不更新 last_y**，始终和这一行的“平均水平”或“行首”比较。
                
                # 修正策略：始终和当前行的【平均Y】比较可能更稳，
                # 但简单起见，既然排好序了，和上一个归入该行的苗比较也行，
                # 只是要注意 diff 的累积。
                
                # 最稳妥方案：
                # 只要 (curr_y - current_lane_avg_y) < threshold。
                # 这里简化：如果 diff (和上一个 sorted item) 很大，就断开。
                pass # 逻辑已经在 if 里处理了

            lanes[current_lane_id].append(curr)
            
            # [关键微调] 
            # 如果我们不更新 last_y，那么 split_threshold 需要大一点 (行宽的一半)。
            # 如果我们更新 last_y = curr['cy']，那么 split_threshold 就是“行间隙”。
            # 这种情况下，更新 last_y 容易导致把紧挨着的两行粘连。
            # 
            # 最优解：**不要更新 last_y**。
            # last_y 应该代表“当前行的重心”。
            # 随着新苗加入，动态更新当前行的重心。
            current_lane_objs = lanes[current_lane_id]
            lane_ys = [p['cy'] for p in current_lane_objs]
            last_y = np.mean(lane_ys) # 始终和当前行的平均线比较

        return lanes

    def check_collision(self, virtual_box, all_boxes):
        """物理碰撞检测"""
        vx1, vy1, vx2, vy2 = virtual_box
        for box in all_boxes:
            bx1, by1, bx2, by2 = box
            xx1 = max(vx1, bx1); yy1 = max(vy1, by1)
            xx2 = min(vx2, bx2); yy2 = min(vy2, by2)
            if xx2 > xx1 and yy2 > yy1: return True
        return False

    def calculate_lane_paces(self, lanes):
        """
        [新增] 计算每一行的局部节奏 (Local Pace)
        解决"忽快忽慢"和"株高不等于株距"的问题
        """
        lane_paces = {}
        all_diffs = []
        
        # 1. 先算每一行的中位间距
        for lid, items in lanes.items():
            if len(items) < 3: 
                lane_paces[lid] = None 
                continue
                
            items.sort(key=lambda x: x['rot_y'])
            ys = np.array([p['rot_y'] for p in items])
            diffs = np.diff(ys)
            
            # 过滤掉明显的离群值（比如巨大的缺苗空档）
            # 只取 0.5倍 到 1.5倍 中位数的间距参与统计
            raw_med = np.median(diffs)
            valid_diffs = diffs[(diffs > raw_med * 0.5) & (diffs < raw_med * 1.5)]
            
            if len(valid_diffs) > 0:
                local_pace = np.median(valid_diffs)
                lane_paces[lid] = local_pace
                all_diffs.extend(valid_diffs)
            else:
                lane_paces[lid] = None

        # 2. 计算一个全局兜底值
        if len(all_diffs) > 0:
            global_fallback = np.median(all_diffs)
        else:
            global_fallback = 50.0 
            
        # 3. 填补那些没算出来的行
        for lid in lanes.keys():
            if lane_paces.get(lid) is None:
                lane_paces[lid] = global_fallback
                
        return lane_paces

    def process(self, raw_boxes, img_w, img_h):
        """
        [全域互斥 + 优先级版]
        
        解决用户痛点：
        1. 防止缺苗框互相重叠 (Vertical Collision of Missing Boxes)。
        2. 引入两轮扫描机制 (Two-Pass Strategy)：
           - Pass 1: 优先计算“补中 (Body)”，确保行内缺苗先占坑。
           - Pass 2: 后计算“补头/尾 (Head/Tail)”，防止边缘延伸出的红叉覆盖了正常的缺苗。
        3. 实时更新障碍物列表：生成的缺苗立刻变为障碍物，后续检测必须避开它。
        """
        if len(raw_boxes) < 2: return [], [], 0
        
        # 1. 基础数据与分行
        angle = 0.0 
        props = self.get_rotated_data(raw_boxes, angle)
        lanes = self.assign_lanes_forced(raw_boxes)
        
        final_missing = []
        labels = np.zeros(len(raw_boxes), dtype=int)
        
        # --- [步骤 A] 准备统计数据 ---
        all_widths = [p['w'] for p in props]
        all_heights = [p['h'] for p in props]
        if not all_widths: return [], [], 0

        # 标准尺寸 (P75 大苗标准)
        p75_w = np.percentile(all_widths, 75)
        p75_h = np.percentile(all_heights, 75)
        virtual_size = max(p75_w, p75_h)
        size_tolerance = 0.85
        
        # 统计全田边界 (用于补头/尾)
        lane_starts = []
        lane_ends = []
        for lid, items in lanes.items():
            if not items: continue
            items.sort(key=lambda x: x['rot_x'])
            lane_starts.append(items[0]['rot_x'] - items[0]['w'] / 2)
            lane_ends.append(items[-1]['rot_x'] + items[-1]['w'] / 2)
            
        if not lane_starts: return [], [], 0
        
        global_start_x = np.percentile(lane_starts, 25)
        global_end_x = np.percentile(lane_ends, 75)
        
        # 抖动探测参数
        nudge_range = virtual_size * 0.25
        nudge_offsets = [0, -nudge_range * 0.5, nudge_range * 0.5, -nudge_range, nudge_range]

        # 动态障碍物列表 (存放新生成的红叉)
        generated_obstacles = []

        # --- [辅助函数] 严格校验位置是否合法 ---
        def is_location_valid(box):
            # 1. 越界检查
            margin = 2
            if (box[0] < -margin or box[1] < -margin or 
                box[2] > img_w + margin or box[3] > img_h + margin):
                return False
            
            # 2. 撞真苗检查
            if self.check_collision(box, raw_boxes):
                return False
                
            # 3. 撞新苗检查
            for ob in generated_obstacles:
                if not (box[2] < ob[0] or box[0] > ob[2] or box[3] < ob[1] or box[1] > ob[3]):
                    return False
            return True

        # --- [内部函数] 填空逻辑 (含平移避让) ---
        def fill_gap_between(start_edge, end_edge, base_y):
            current_edge = start_edge
            
            while end_edge - current_edge >= virtual_size * size_tolerance:
                virtual_cx = current_edge + virtual_size / 2
                
                placed_success = False
                final_vbox = None
                
                # 遍历抖动位置
                for offset_y in nudge_offsets:
                    test_y = base_y + offset_y
                    
                    vw, vh = virtual_size, virtual_size
                    scale = 0.9
                    # 初始尝试框
                    vbox = [virtual_cx - vw*scale/2, test_y - vh*scale/2, 
                            virtual_cx + vw*scale/2, test_y + vh*scale/2]
                    
                    # === 尝试 1: 直接检查当前位置 ===
                    if is_location_valid(vbox):
                        placed_success = True
                        final_vbox = vbox
                        break
                    
                    # === 尝试 2: 如果撞了真苗，尝试上下平移避让 ===
                    # 找出撞到了哪个框，尝试躲避它
                    collision_found = False
                    shifted_vbox = None
                    
                    for rbox in raw_boxes:
                        # 简单的 AABB 碰撞判断
                        if (vbox[0] < rbox[2] and vbox[2] > rbox[0] and 
                            vbox[1] < rbox[3] and vbox[3] > rbox[1]):
                            
                            # 发生了碰撞，分析方向
                            rbox_ymin, rbox_ymax = rbox[1], rbox[3]
                            vbox_ymin, vbox_ymax = vbox[1], vbox[3]
                            
                            # Case A: 底部碰撞 (虚拟框的底部 戳进了 真实框的顶部)
                            # 判定：虚拟框中心比真实框中心更靠上，且下边缘重叠
                            if (vbox_ymin + vbox_ymax)/2 < (rbox_ymin + rbox_ymax)/2:
                                # 策略：向上移动 (Shift UP)
                                # 目标：让虚拟框底部 = 真实框顶部 - 1px (留点缝隙)
                                shift_dist = vbox_ymax - rbox_ymin + 1
                                # 限制最大移动距离 (不能为了躲避跑太远)
                                if shift_dist < vh * 0.5:
                                    shifted_vbox = [vbox[0], vbox[1] - shift_dist, 
                                                    vbox[2], vbox[3] - shift_dist]
                                    collision_found = True
                            
                            # Case B: 顶部碰撞 (虚拟框的顶部 戳进了 真实框的底部)
                            # 判定：虚拟框中心比真实框中心更靠下，且上边缘重叠
                            else:
                                # 策略：向下移动 (Shift DOWN)
                                # 目标：让虚拟框顶部 = 真实框底部 + 1px
                                shift_dist = rbox_ymax - vbox_ymin + 1
                                if shift_dist < vh * 0.5:
                                    shifted_vbox = [vbox[0], vbox[1] + shift_dist, 
                                                    vbox[2], vbox[3] + shift_dist]
                                    collision_found = True
                            
                            # 我们只处理第一个碰到的物体进行避让尝试
                            break 
                    
                    # 如果刚才计算出了一个避让后的新框，必须再次校验它是否合法
                    # (防止往上躲的时候撞到了上面的框，或者跑出了边界)
                    if collision_found and shifted_vbox is not None:
                        if is_location_valid(shifted_vbox):
                            placed_success = True
                            final_vbox = shifted_vbox
                            break
                
                # --- 结算 ---
                if placed_success and final_vbox is not None:
                    final_cy = (final_vbox[1] + final_vbox[3]) / 2
                    final_missing.append([virtual_cx, final_cy])
                    
                    generated_obstacles.append(final_vbox)
                    current_edge += virtual_size 
                else:
                    current_edge += virtual_size * 0.5
        
        # --- [步骤 B] 优先级执行策略 ---
        lane_meta = {}
        for lid, items in lanes.items():
            if not items: continue
            for p in items: labels[p['id']] = lid
            items.sort(key=lambda x: x['rot_x'])
            
            if len(items) >= 2:
                row_mean_y = np.median([p['rot_y'] for p in items])
            else:
                row_mean_y = items[0]['rot_y']
            
            lane_meta[lid] = {'items': items, 'mean_y': row_mean_y}

        # PASS 1: Body
        for lid, meta in lane_meta.items():
            items = meta['items']
            for k in range(len(items) - 1):
                curr = items[k]
                next_box = items[k+1]
                curr_right = curr['rot_x'] + curr['w'] / 2
                next_left = next_box['rot_x'] - next_box['w'] / 2
                local_y = (curr['rot_y'] + next_box['rot_y']) / 2
                fill_gap_between(curr_right, next_left, local_y)

        # PASS 2: Head/Tail
        for lid, meta in lane_meta.items():
            items = meta['items']
            row_mean_y = meta['mean_y']
            # Head
            first_left = items[0]['rot_x'] - items[0]['w'] / 2
            if first_left > global_start_x + virtual_size * 0.5:
                fill_gap_between(global_start_x, first_left, row_mean_y)
            # Tail
            last_right = items[-1]['rot_x'] + items[-1]['w'] / 2
            if last_right < global_end_x - virtual_size * 0.5:
                fill_gap_between(last_right, global_end_x, row_mean_y)

        return final_missing, labels, angle

def imread_safe(path):
    try: return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except: return None

def imwrite_safe(path, img):
    try: cv2.imencode('.jpg', img)[1].tofile(path)
    except: pass

def main():
    txt_path = Config.get_path(Config.TXT_FILE_NAME)
    xml_dir = Config.get_path(Config.LABEL_DIR_NAME)
    img_dir = Config.get_path(Config.IMAGE_DIR_NAME)
    out_dir = Config.get_path(Config.OUTPUT_DIR_NAME)
    
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    if not os.path.exists(txt_path): return

    with open(txt_path, 'r', encoding='utf-8') as f:
        file_list = [line.strip() for line in f if line.strip()]
        
    print(f"Processing {len(file_list)} images (Forced Alignment + Head/Tail)...")
    
    detector = ForcedAlignmentDetector()
    
    for file_name in tqdm(file_list):
        stem = os.path.splitext(os.path.basename(file_name))[0]
        xml_path = os.path.join(xml_dir, stem + ".xml")
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
            temp = os.path.join(img_dir, stem + ext)
            if os.path.exists(temp):
                image_path = temp
                break
        
        if not os.path.exists(xml_path) or not image_path: continue
        
        raw_boxes, width, height = detector.parse_xml(xml_path)
        img = imread_safe(image_path)
        if len(raw_boxes) == 0 or img is None: continue
        
        missing_pts, labels, angle = detector.process(raw_boxes, width, height)
        
        # 绘图
        for i, box in enumerate(raw_boxes):
            lid = labels[i]
            color = Config.LANE_COLORS[lid % len(Config.LANE_COLORS)]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # 可选：显示它被分配到了哪一行
            # cv2.putText(img, str(lid), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for pt in missing_pts:
            cx, cy = int(pt[0]), int(pt[1])
            w, h = 35, 35
            # 画红色方框表示缺苗
            cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), Config.MISSING_COLOR, 2)
            # 画个叉
            cv2.line(img, (cx-10, cy-10), (cx+10, cy+10), Config.MISSING_COLOR, 2)
            cv2.line(img, (cx+10, cy-10), (cx-10, cy+10), Config.MISSING_COLOR, 2)

        info = f"Forced Align | Miss: {len(missing_pts)}"
        cv2.rectangle(img, (0, 0), (300, 30), (0,0,0), -1)
        cv2.putText(img, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        save_path = os.path.join(out_dir, stem + "_forced.jpg")
        imwrite_safe(save_path, img)

    print("Done.")

if __name__ == "__main__":
    main()