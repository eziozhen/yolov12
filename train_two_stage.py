from ultralytics import YOLO

def run_two_stage_training():
    # ==========================================
    # 阶段一：Warm-up (只用纯金数据)
    # 目标：让模型先学会“什么是对的”，建立高 Precision
    # ==========================================
    print(">>> 开始阶段一：Gold Data Warm-up...")
    model = YOLO('yolov12s.yaml')
    
    results_1 = model.train(
    data="coco_ori.yaml", 
    epochs=100,
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

    # 获取阶段一最好的权重路径
    # 通常在 dart_training/stage1_gold/weights/best.pt
    stage1_weight = f"{results_1.save_dir}/weights/best.pt"
    print(f">>> 阶段一完成。最佳权重已保存至: {stage1_weight}")

    # ==========================================
    # 阶段二：Boost (混合数据)
    # 目标：引入困难样本 (Silver)，提升 Recall
    # ==========================================
    print(">>> 开始阶段二：Full Data Fine-tuning...")
    
    # 关键点：加载阶段一训练好的权重，而不是从头开始
    model_stage2 = YOLO(stage1_weight)
    
    results_2 = model_stage2.train(
    data="coco_dart.yaml", 
    epochs=150,
    batch=16,
    imgsz=640,
    seed=42,
    lr0=0.002,             # 设置为原来的 1/5 甚至 1/10
    lrf=0.01,              # 最终学习率倍率也相应调整
    optimizer='SGD',
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

    print(">>> 全部训练完成！最终模型在 dart_training/stage2_full/weights/best.pt")

if __name__ == '__main__':
    run_two_stage_training()