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
        return np.array(boxes)

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
        [直方图优化版 - 高斯平滑 + 显著性寻峰]
        解决：
        1. 两行合一行 -> 降低最小间距限制，利用显著性区分。
        2. 一行分两色 -> 使用高斯模糊抹平波峰顶端的微小抖动。
        """
        if len(raw_boxes) == 0: return {}
        
        # 1. 霍夫变换算角度并旋转
        # angle = self.calculate_global_angle(raw_boxes)
        angle = 0.0
        props = self.get_rotated_data(raw_boxes, angle)

        if not props: return {}
        
        # 2. 准备投影直方图
        all_ys = [p['rot_y'] for p in props]
        all_hs = [p['h'] for p in props]
        avg_h = np.median(all_hs) if all_hs else 30
        
        min_y, max_y = min(all_ys), max(all_ys)
        # 放大分辨率，让直方图更细腻 (1像素 -> 2个bin)
        resolution_scale = 2.0 
        height_range = int((max_y - min_y + avg_h * 2) * resolution_scale)
        y_hist = np.zeros(height_range + 100, dtype=np.float32)
        
        offset = (-min_y + avg_h) * resolution_scale
        
        for p in props:
            # 累加区间：从 Top 到 Bottom
            y_start = int((p['rot_top'] * resolution_scale) + offset)
            y_end = int((p['rot_bottom'] * resolution_scale) + offset)
            
            y_start = max(0, min(y_start, len(y_hist)-1))
            y_end = max(0, min(y_end, len(y_hist)-1))
            
            # 权重优化：用框的面积或者宽度，让大苗更有话语权
            weight = p['w']
            y_hist[y_start:y_end] += weight
            
        # 3. [关键优化] 高斯平滑
        # sigma 决定了“模糊”程度。
        # 太小 -> 一行变两行；太大 -> 两行并一行。
        # 经验值：sigma = 苗高的 0.25 倍左右最合适
        sigma = (avg_h * resolution_scale) * 0.25
        y_hist_smooth = gaussian_filter1d(y_hist, sigma)
        
        # 4. [关键优化] 寻找波峰
        # distance: 两个峰之间的最小距离。设为 0.6 倍株高，防止把紧挨着的两行漏掉
        min_dist = (avg_h * resolution_scale) * 0.6
        # prominence: 突起程度。滤除那些像杂草一样的平缓小包
        min_prominence = np.max(y_hist_smooth) * 0.15
        
        peaks, _ = find_peaks(y_hist_smooth, distance=min_dist, prominence=min_prominence)
        
        # 5. [兜底策略] 峰值合并 (Post-Merge)
        # 如果 find_peaks 还是把一行分成了两个很近的峰，我们手动合一下
        final_peak_centers = []
        if len(peaks) > 0:
            current_merge_group = [peaks[0]]
            
            for i in range(1, len(peaks)):
                prev_p = peaks[i-1]
                curr_p = peaks[i]
                
                # 如果两个峰距离小于 0.7 倍株高，认为是同一个宽行的两个肩
                if (curr_p - prev_p) < (avg_h * resolution_scale * 0.7):
                    current_merge_group.append(curr_p)
                else:
                    # 结算上一组
                    final_peak_centers.append(np.mean(current_merge_group))
                    current_merge_group = [curr_p]
            # 结算最后一组
            final_peak_centers.append(np.mean(current_merge_group))
        else:
            final_peak_centers = [np.mean(all_ys) * resolution_scale + offset]

        # 还原回真实坐标
        lane_centers = [(pk - offset) / resolution_scale for pk in final_peak_centers]
        lane_centers = np.array(lane_centers)
        
        lanes = {i: [] for i in range(len(lane_centers))}
        
        # 6. 分配逻辑：寻找最近的波峰
        # 增加一个拒识阈值：如果离哪个峰都太远(行间杂草)，就丢弃
        valid_dist_limit = avg_h * 1.2
        
        for p in props:
            y = p['rot_y']
            dists = np.abs(lane_centers - y)
            nearest_idx = np.argmin(dists)
            
            if dists[nearest_idx] < valid_dist_limit:
                lanes[nearest_idx].append(p)
        
        # 7. 过滤空行或极少行
        final_lanes = {}
        idx_cnt = 0
        sorted_ids = sorted(lanes.keys())
        for lid in sorted_ids:
            if len(lanes[lid]) >= 3: # 至少3棵苗才算一行
                final_lanes[idx_cnt] = lanes[lid]
                idx_cnt += 1
                
        return final_lanes

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

    def process(self, raw_boxes):
        """
        [智能试探填充版]
        
        响应用户：
        1. 解决“碰撞检测太死板”的问题。
        2. 引入“抖动探测” (Nudging)：在尝试放置虚拟苗时，不仅试探行中心，
           还会尝试向上、向下微调坐标。只要能在微调范围内找到一个不碰撞的位置，就判定放置成功。
        3. 保持迭代式推进：放下一个后，从新框右边继续。
        """
        if len(raw_boxes) < 2: return [], [], 0
        
        # 1. 锁定 Angle = 0
        angle = 0.0 
        props = self.get_rotated_data(raw_boxes, angle)
        
        # 2. 分行
        lanes = self.assign_lanes_forced(raw_boxes)
        
        final_missing = []
        labels = np.zeros(len(raw_boxes), dtype=int)
        
        # --- [步骤 A] 确定“标准虚拟苗”尺寸 ---
        all_widths = [p['w'] for p in props]
        all_heights = [p['h'] for p in props]
        if not all_widths: return [], [], 0

        # 使用 75 分位数构建标准框 (大苗标准)
        p75_w = np.percentile(all_widths, 75)
        p75_h = np.percentile(all_heights, 75)
        
        # 虚拟框尺寸：取宽高的最大值构建正方形，保证占位足够
        virtual_size = max(p75_w, p75_h)
        
        # --- [步骤 B] 定义抖动范围 ---
        # 允许上下浮动的范围。比如允许偏移 25% 的苗高。
        # 如果在这个范围内能找到空位，就算成功。
        nudge_range = virtual_size * 0.25 
        # 定义尝试的偏移量：[0, -5, +5, -10, +10...]
        # 优先试中间，不行试两边
        nudge_offsets = [0, -nudge_range * 0.5, nudge_range * 0.5, -nudge_range, nudge_range]

        # 3. 逐行处理
        for lid, items in lanes.items():
            if len(items) < 2: continue
            
            for p in items: labels[p['id']] = lid
            items.sort(key=lambda x: x['rot_x'])
            
            # --- [步骤 C] 迭代式填充 + 智能试探 ---
            for k in range(len(items) - 1):
                curr = items[k]
                next_box = items[k+1]
                
                # 起始边缘 (左苗的右边)
                current_edge = curr['rot_x'] + curr['w'] / 2
                # 终点边缘 (右苗的左边)
                gap_end = next_box['rot_x'] - next_box['w'] / 2
                
                # 行的基准中心 Y
                base_row_y = (curr['rot_y'] + next_box['rot_y']) / 2
                
                # 循环填空
                while gap_end - current_edge >= virtual_size:
                    
                    # 尝试放置的 X 中心
                    virtual_cx = current_edge + virtual_size / 2
                    
                    # [核心修改] 抖动探测 (Nudging Check)
                    placed_success = False
                    best_y = base_row_y
                    
                    for offset_y in nudge_offsets:
                        # 尝试一个新的 Y 坐标
                        test_y = base_row_y + offset_y
                        
                        # 构建测试框
                        vw, vh = virtual_size, virtual_size
                        # 这里把测试框稍微缩小一点点(比如0.9倍)，防止边缘刚好擦边导致的误判
                        # 物理上我们允许稍微有一点点挨着
                        scale = 0.9 
                        vbox = [virtual_cx - vw*scale/2, test_y - vh*scale/2, 
                                virtual_cx + vw*scale/2, test_y + vh*scale/2]
                        
                        # 碰撞检测
                        if not self.check_collision(vbox, raw_boxes):
                            # 找到了！这个位置不碰撞
                            placed_success = True
                            best_y = test_y
                            break # 只要找到一个能放的位置，就停止尝试
                    
                    if placed_success:
                        # 记录这个微调后的最佳位置
                        final_missing.append([virtual_cx, best_y])
                        
                        # 成功占位：墙往右推
                        current_edge += virtual_size
                    else:
                        # [重要] 如果所有位置都试过了还是碰撞
                        # 说明这里有个障碍物（杂草/误检框）挡得严严实实
                        # 我们不能死循环，必须跳过这个障碍物
                        # 策略：稍微往右移动一点点（比如 1/3 个身位），看看能不能避开
                        # 这样可以防止死锁，也能尝试在障碍物后面找空隙
                        current_edge += virtual_size * 0.5

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
        
        raw_boxes = detector.parse_xml(xml_path)
        img = imread_safe(image_path)
        if len(raw_boxes) == 0 or img is None: continue
        
        missing_pts, labels, angle = detector.process(raw_boxes)
        
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