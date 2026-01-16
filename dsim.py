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
    TXT_FILE_NAME = "VOCdevkit/VOC2007/ImageSets/Main/trainval.txt"
    OUTPUT_DIR_NAME = "output_dynamic_new" # 输出文件夹改个名区分一下
    
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
        split_threshold = median_h * 0.8

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

    def process(self, raw_boxes, img_w, img_h):
        """
        [绝对隔离版 - 四面强制留空]
        
        核心升级：
        1. 膨胀碰撞检测 (Inflated Collision Check)：
           在检查位置是否合法时，使用一个“向四周膨胀 15%”的隐形框去检测。
           只要隐形框碰到了任何东西，就视为位置太挤，拒绝放置。
           这保证了最终的红框四周都有 15% 的物理留空。
        
        2. 漂移控制：
           基准线：np.polyfit 拟合的中心直线。
           漂移量：允许上下浮动 30% (nudge_range) 来适应微小的地形起伏。
        """
        if len(raw_boxes) < 2: return [], [], 0
        
        # 1. 基础流程
        angle = 0.0 
        props = self.get_rotated_data(raw_boxes, angle)
        lanes = self.assign_lanes_forced(raw_boxes)
        
        final_missing = []
        labels = np.zeros(len(raw_boxes), dtype=int)
        
        # --- 尺寸统计 (P75 高度) ---
        all_heights = [p['h'] for p in props]
        if not all_heights: return [], [], 0
        global_p75_h = np.percentile(all_heights, 95)

        # 统计边界
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
        
        generated_obstacles = []

        # --- [核心修改] 膨胀版碰撞检测 ---
        def is_location_valid(box, margin_pixels):
            """
            box: [xmin, ymin, xmax, ymax] (实际要画的框)
            margin_pixels: 强制留空的像素距离 (四面都要留)
            """
            # 1. 越界检查 (检查实际框是否出界)
            # 允许框贴着图片边缘，但不允许出去
            if (box[0] < 0 or box[1] < 0 or 
                box[2] > img_w or box[3] > img_h):
                return False

            # 2. 构造“膨胀框” (Inflated Box)
            # 我们用这个大一圈的框去探测障碍物
            expanded_box = [
                box[0] - margin_pixels, # 左扩
                box[1] - margin_pixels, # 上扩
                box[2] + margin_pixels, # 右扩
                box[3] + margin_pixels  # 下扩
            ]
            
            # 3. 撞真苗检查 (用膨胀框去撞)
            # 如果膨胀框撞到了真苗，说明真苗在安全距离内 -> 拒绝
            if self.check_collision(expanded_box, raw_boxes):
                return False
                
            # 4. 撞新苗检查 (用膨胀框去撞)
            # 如果膨胀框撞到了其他红框，说明离得太近 -> 拒绝
            for ob in generated_obstacles:
                if not (expanded_box[2] < ob[0] or expanded_box[0] > ob[2] or 
                        expanded_box[3] < ob[1] or expanded_box[1] > ob[3]):
                    return False
            
            return True

        # --- 填空函数 ---
        def fill_gap_between(start_edge, end_edge, slope, intercept, virtual_size):
            current_edge = start_edge
            
            # [定义安全距离]
            # 四周都要留出 20% 的空隙
            safety_margin = virtual_size * 0.20
            
            # X轴步进策略：
            # 必须留出：左margin + 框 + 右margin
            step_width = virtual_size + 2 * safety_margin
            
            # 循环条件：剩余空间必须 > step_width
            while end_edge - current_edge >= step_width:
                
                # 计算 X 中心：
                # 当前边缘 + 左安全距离 + 半个框宽
                virtual_cx = current_edge + safety_margin + virtual_size / 2
                
                # 再次检查右边界：
                # 框的右边 + 右安全距离 必须 < 终点
                if virtual_cx + virtual_size/2 + safety_margin > end_edge:
                    break
                
                # 计算 Y 中心：基于线性拟合
                target_y = slope * virtual_cx + intercept
                
                placed_success = False
                final_vbox = None
                
                # [漂移设置]
                # 允许上下浮动 30% 的高度去寻找不碰的位置
                nudge_range = virtual_size * 0.3
                nudge_offsets = [0, -nudge_range * 0.5, nudge_range * 0.5, -nudge_range, nudge_range]
                
                for offset_y in nudge_offsets:
                    test_y = target_y + offset_y
                    
                    vw, vh = virtual_size, virtual_size
                    # 虚拟框本身
                    vbox = [virtual_cx - vw/2, test_y - vh/2, 
                            virtual_cx + vw/2, test_y + vh/2]
                    
                    # 1. 严格检查 (传入 safety_margin)
                    if is_location_valid(vbox, safety_margin):
                        placed_success = True
                        final_vbox = vbox
                        break
                    
                    # 2. 智能避让 (平移)
                    # 避让逻辑依然尝试寻找空位
                    collision_found = False
                    shifted_vbox = None
                    for rbox in raw_boxes:
                        # 这里用 Expanded Box 逻辑太复杂，我们先检测直接碰撞，再微调
                        # 为了简化计算，避让依然基于原始框，但校验时使用严格标准
                        if (vbox[0] < rbox[2] and vbox[2] > rbox[0] and 
                            vbox[1] < rbox[3] and vbox[3] > rbox[1]):
                            rbox_ymin, rbox_ymax = rbox[1], rbox[3]
                            vbox_ymin, vbox_ymax = vbox[1], vbox[3]
                            if (vbox_ymin + vbox_ymax)/2 < (rbox_ymin + rbox_ymax)/2: # Bottom Hit
                                # 往上躲，躲多远？躲到不接触为止 + 安全距离
                                shift_dist = vbox_ymax - rbox_ymin + 1 + safety_margin
                                if shift_dist < vh * 0.6: # 允许最大移动 60%
                                    shifted_vbox = [vbox[0], vbox[1] - shift_dist, vbox[2], vbox[3] - shift_dist]
                                    collision_found = True
                            else: # Top Hit
                                shift_dist = rbox_ymax - vbox_ymin + 1 + safety_margin
                                if shift_dist < vh * 0.6:
                                    shifted_vbox = [vbox[0], vbox[1] + shift_dist, vbox[2], vbox[3] + shift_dist]
                                    collision_found = True
                            break 
                    
                    if collision_found and shifted_vbox is not None:
                        # 避让后的新框，必须通过严格检查
                        if is_location_valid(shifted_vbox, safety_margin):
                            placed_success = True
                            final_vbox = shifted_vbox
                            break
                
                if placed_success and final_vbox is not None:
                    final_missing.append(final_vbox)
                    generated_obstacles.append(final_vbox)
                    # 成功后，推进：框宽 + 2倍安全距离 (保证下一个框从安全区外开始算)
                    current_edge += step_width
                else:
                    # 失败后，只推进半个框宽试探
                    current_edge += virtual_size * 0.5
        
        # --- 准备每行的拟合方程 ---
        lane_meta = {}
        for lid, items in lanes.items():
            if not items: continue
            for p in items: labels[p['id']] = lid
            items.sort(key=lambda x: x['rot_x'])
            
            # 拟合直线
            xs = [p['rot_x'] for p in items]
            ys = [p['rot_y'] for p in items]
            if len(xs) >= 2:
                slope, intercept = np.polyfit(xs, ys, 1)
            else:
                slope, intercept = 0.0, ys[0]
            
            # 尺寸计算
            row_heights = [p['h'] for p in items]
            row_median_h = np.max(row_heights)
            final_row_size = (global_p75_h * 0.8) + (row_median_h * 0.2)
            # final_row_size = global_p75_h
            
            lane_meta[lid] = {
                'items': items, 
                'slope': slope, 
                'intercept': intercept, 
                'virtual_size': final_row_size 
            }

        # PASS 1: Body
        for lid, meta in lane_meta.items():
            items = meta['items']
            v_size = meta['virtual_size']
            slope = meta['slope']
            intercept = meta['intercept']
            for k in range(len(items) - 1):
                curr = items[k]
                next_box = items[k+1]
                fill_gap_between(
                    curr['rot_x'] + curr['w'] / 2, 
                    next_box['rot_x'] - next_box['w'] / 2, 
                    slope, intercept, v_size
                )

        # PASS 2: Head/Tail
        for lid, meta in lane_meta.items():
            items = meta['items']
            v_size = meta['virtual_size']
            slope = meta['slope']
            intercept = meta['intercept']
            
            first_left = items[0]['rot_x'] - items[0]['w'] / 2
            if first_left > global_start_x + v_size * 0.5:
                fill_gap_between(global_start_x, first_left, slope, intercept, v_size)
            
            last_right = items[-1]['rot_x'] + items[-1]['w'] / 2
            if last_right < global_end_x - v_size * 0.5:
                fill_gap_between(last_right, global_end_x, slope, intercept, v_size)

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
            xmin, ymin, xmax, ymax = map(int, pt)
            # cx, cy = int(pt[0]), int(pt[1])
            # w, h = 35, 35
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), Config.MISSING_COLOR, 3)
            cv2.line(img, (xmin, ymin), (xmax, ymax), Config.MISSING_COLOR, 2)
            cv2.line(img, (xmin, ymax), (xmax, ymin), Config.MISSING_COLOR, 2)
            # # 画红色方框表示缺苗
            # cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), Config.MISSING_COLOR, 2)
            # # 画个叉
            # cv2.line(img, (cx-10, cy-10), (cx+10, cy+10), Config.MISSING_COLOR, 2)
            # cv2.line(img, (cx+10, cy-10), (cx-10, cy+10), Config.MISSING_COLOR, 2)

        info = f"Forced Align | Miss: {len(missing_pts)}"
        cv2.rectangle(img, (0, 0), (300, 30), (0,0,0), -1)
        cv2.putText(img, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        save_path = os.path.join(out_dir, stem + ".jpg")
        imwrite_safe(save_path, img)

    print("Done.")

if __name__ == "__main__":
    main()