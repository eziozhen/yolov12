import os
import shutil

def fast_copy_files(src_dir, dst_dir, target_ids, black_list):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    files = os.listdir(src_dir)
    print(f"扫描到 {len(files)} 个文件，开始处理...")

    count = 0
    
    for filename in files:
        if not filename.endswith('.xml'):
            continue

        # 1. 直接切片获取 ID (例如 "DJI_0347")
        # 假设格式固定，取前8位即可
        dji_id = filename[:8]

        # 2. 判断是否在列表
        if dji_id in target_ids and dji_id not in black_list:
            # 3. 生成新文件名
            # 原名: DJI_0347_frame_DJI_0347_frame_00329.xml
            # 逻辑: ID(8位) + "_frame_"(7位) = 15位
            # 直接从第 15 位开始截取，就是后面的部分
            new_filename = filename[15:] 

            # 简单的安全检查：确保切切片后还是以 DJI 开头，防止切错了文件
            if not new_filename.startswith("DJI_"):
                print(f"[警告] 文件名格式可能不对，跳过: {filename}")
                continue

            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, new_filename)

            shutil.copy2(src_path, dst_path)
            print(f"[OK] {filename} -> {new_filename}")
            count += 1

    print(f"处理完成，共拷贝 {count} 个文件。")

# ================= 配置 =================

source_folder = 'dart_perfect_samples_xml'
target_folder = 'dart_perfect_samples_xml_cleaned'

# 在这里填入你要的 ID
allow_list = ['DJI_0449', 'DJI_0389', 'DJI_0394', 'DJI_0350',
              'DJI_0420', 'DJI_0387', 'DJI_0358', 'DJI_0386',
              'DJI_0395', 'DJI_0391', 'DJI_0356', 'DJI_0354',
              'DJI_0347', 'DJI_0385', 'DJI_0419', 'DJI_0392',
              'DJI_0396', 'DJI_0456', 'DJI_0447', 'DJI_0460',
              'DJI_0453', 'DJI_0451', 'DJI_0459', 'DJI_0448',
              'DJI_0455']

black_list = ['DJI_0447', 'DJI_0391', 'DJI_0392', 'DJI_0394', 'DJI_0395', 'DJI_0396']
# DJI_0447

if __name__ == '__main__':
    # 获取绝对路径，防止路径错误
    base_dir = os.getcwd()
    src = os.path.join(base_dir, source_folder)
    dst = os.path.join(base_dir, target_folder)
    
    if os.path.exists(src):
        fast_copy_files(src, dst, allow_list, black_list)
    else:
        print("找不到源文件夹")