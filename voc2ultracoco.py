import os
import shutil
import xml.etree.ElementTree as ET
import glob
import random
import yaml
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 原始 VOC 数据路径
VOC_ROOT = "VOCdevkit_new/VOC2007"  
XML_PATH = os.path.join(VOC_ROOT, "Annotations")
# IMG_PATH = os.path.join(VOC_ROOT, "JPEGImages")
IMG_PATH = os.path.join("/home/caozhenzhen/xiaomai/", "ALL_JPEG")
SPLIT_PATH = os.path.join(VOC_ROOT, "ImageSets", "Main")

# 2. 目标输出路径 (符合 Ultralytics 结构)
OUTPUT_ROOT = "wheat_yolo_dataset_new"

# 3. 划分比例 (如果没有现成的 txt 文件)
SPLIT_RATIO = 0.8  
# ===========================================

def parse_xml(xml_file):
    """解析 VOC XML 文件，返回归一化所需的宽高等信息"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    filename = root.find("filename").text
    
    # 容错：处理文件名后缀问题
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        basename = os.path.splitext(os.path.basename(xml_file))[0]
        filename = basename + ".jpg" # 默认补全 jpg

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        boxes.append([name, xmin, ymin, xmax, ymax])
        
    return filename, width, height, boxes

def convert_box_to_yolo(size, box):
    """
    将 VOC (xmin, ymin, xmax, ymax) 转换为 YOLO (x_center, y_center, w, h) 并归一化
    size: (width, height)
    box: (xmin, xmax, ymin, ymax)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_batch(image_ids, mode, category_map):
    """
    mode: 'train' 或 'val'
    """
    # 建立目录结构: images/train, labels/train
    save_img_dir = os.path.join(OUTPUT_ROOT, "images", mode)
    save_label_dir = os.path.join(OUTPUT_ROOT, "labels", mode)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    print(f"正在处理 {mode} 集，共 {len(image_ids)} 张图片...")
    
    for img_id_str in tqdm(image_ids):
        xml_file = os.path.join(XML_PATH, f"{img_id_str}.xml")
        
        if not os.path.exists(xml_file):
            continue

        # 1. 解析 XML
        filename, width, height, boxes = parse_xml(xml_file)
        
        # 2. 复制图片 (源路径容错)
        src_img_path = os.path.join(IMG_PATH, filename)
        if not os.path.exists(src_img_path):
            # 尝试用 ID 寻找图片
            potential_exts = ['.jpg', '.png', '.jpeg', '.bmp']
            found = False
            for ext in potential_exts:
                temp_path = os.path.join(IMG_PATH, f"{img_id_str}{ext}")
                if os.path.exists(temp_path):
                    src_img_path = temp_path
                    filename = f"{img_id_str}{ext}" # 更新真正的文件名
                    found = True
                    break
            if not found:
                print(f"警告: 找不到图片 ID {img_id_str}，跳过")
                continue

        # 复制文件
        dst_img_path = os.path.join(save_img_dir, filename)
        shutil.copy2(src_img_path, dst_img_path)

        # 3. 生成 YOLO 格式的 TXT 标签
        # txt文件名必须与图片名一致（除了后缀）
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(save_label_dir, txt_filename)

        with open(txt_path, 'w') as f:
            for name, xmin, ymin, xmax, ymax in boxes:
                if name not in category_map:
                    continue
                
                cls_id = category_map[name]
                # 注意传入 convert 的顺序
                yolo_box = convert_box_to_yolo((width, height), (xmin, xmax, ymin, ymax))
                
                # 写入: class_id x_center y_center w h
                line = f"{cls_id} {' '.join([f'{x:.6f}' for x in yolo_box])}\n"
                f.write(line)

def get_all_categories():
    """扫描所有XML获取类别名称，生成索引映射"""
    classes = set()
    xml_files = glob.glob(os.path.join(XML_PATH, "*.xml"))
    print("正在扫描类别...")
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            for obj in tree.findall("object"):
                classes.add(obj.find("name").text)
        except:
            pass
    
    # 排序，保证 ID 顺序一致. YOLO ID 从 0 开始
    sorted_classes = sorted(list(classes))
    category_map = {name: i for i, name in enumerate(sorted_classes)}
    print(f"检测到类别 (ID映射): {category_map}")
    return category_map, sorted_classes

def generate_yaml(class_names):
    """自动生成 data.yaml 文件"""
    yaml_content = {
        'path': os.path.abspath(OUTPUT_ROOT),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = os.path.join(OUTPUT_ROOT, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"\n配置已生成: {yaml_path}")

def main():
    if os.path.exists(OUTPUT_ROOT):
        print(f"提示: 输出目录 {OUTPUT_ROOT} 已存在。")
    
    # 1. 获取类别
    category_map, class_list = get_all_categories()

    # 2. 划分数据集
    train_txt = os.path.join(SPLIT_PATH, "train.txt")
    val_txt = os.path.join(SPLIT_PATH, "val.txt")

    train_ids = []
    val_ids = []

    if os.path.exists(train_txt) and os.path.exists(val_txt):
        print("使用现有的 ImageSets 划分。")
        with open(train_txt, 'r') as f:
            train_ids = [x.strip() for x in f.readlines() if x.strip()]
        with open(val_txt, 'r') as f:
            val_ids = [x.strip() for x in f.readlines() if x.strip()]
    else:
        print("未发现划分文件，执行随机划分...")
        all_xmls = glob.glob(os.path.join(XML_PATH, "*.xml"))
        all_ids = [os.path.splitext(os.path.basename(x))[0] for x in all_xmls]
        
        random.shuffle(all_ids)
        split_idx = int(len(all_ids) * SPLIT_RATIO)
        train_ids = all_ids[:split_idx]
        val_ids = all_ids[split_idx:]

    # 3. 执行转换
    process_batch(train_ids, 'train', category_map)
    process_batch(val_ids, 'val', category_map)

    # 4. 生成 yaml 配置文件
    generate_yaml(class_list)

    print("\n转换全部完成！")
    print(f"请在训练时使用: model.train(data='{os.path.abspath(OUTPUT_ROOT)}/data.yaml', ...)")

if __name__ == "__main__":
    main()