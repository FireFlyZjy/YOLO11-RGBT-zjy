import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # ============================================================================
    # 模型配置
    # ============================================================================
    # RGBT (4通道: RGB + IR) 使用的模型 YAML
    # MODEL_YAML = "ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion-SE.yaml"     # 例: 带SE注意力的双模态模型
    # RGBRGB6C (6通道: RGB 两次拼接) 使用的模型 YAML
    MODEL_YAML = "ultralytics/cfg/models/26-RGBT/yolo26-RGBRGB6C-midfusion.yaml"     # 只需将yaml中ch设成6, 红外部分改为SilenceChannel[3,6]

    # ============================================================================
    # 预训练权重
    # ============================================================================
    PRETRAINED = "weights/yolo26n.pt"                                                 # 预训练权重路径, 注释掉则不加载

    # ============================================================================
    # 数据集
    # ============================================================================
    # DATA = R'ultralytics/cfg/datasets/flir.yaml'                                   # FLIR 数据集 (RGBT 3类)
    DATA = R'ultralytics/cfg/datasets/LLVIP_zjy.yaml'                                # LLVIP 数据集 (RGBRGB6C 1类)

    # ============================================================================
    # 模态配置
    # ============================================================================
    USE_SIMOTM = "RGBRGB6C"                                                          # 模态组合方式:
                                                                                     #   "RGBT"     - 4通道 [RGB + IR]
                                                                                     #   "RGBRGB6C" - 6通道 [RGB + RGB]
                                                                                     #   "SimOTMBBS" - SimOTM+BBS
                                                                                     #   "RGBT_IR"  - IR only
                                                                                     #   "RGBRGB"   - 6通道备选
                                                                                     #   "Gray"     - 灰度
    CHANNELS = 6                                                                     # 输入通道数, RGBT=4, RGBRGB6C=6

    # ============================================================================
    # 训练参数
    # ============================================================================
    EPOCHS = 100                                                                     # 训练轮数 (epoch), 越大模型收敛越好但耗时越长
    BATCH = 4                                                                        # 批次大小 (batch size), 越大显存占用越高, 小显存可设2
    IMGSZ = 640                                                                      # 输入图像尺寸, 训练时缩放到此大小
    WORKERS = 0                                                                      # 数据加载线程数, Windows下建议0避免bug
    DEVICE = "0"                                                                     # GPU设备号, "0"表示第一张卡, "cpu"表示使用CPU
    OPTIMIZER = "SGD"                                                                # 优化器: "SGD"(推荐), "Adam", "AdamW"等
    CLOSE_MOSAIC = 10                                                                # 最后N个epoch关闭mosaic增强, 稳定训练后期
    CACHE = False                                                                    # 是否将数据集缓存到内存, True加速但耗内存
    # LR0 = 0.002                                                                    # 初始学习率 (注释掉则使用YOLO默认值)
    # RESUME = ''                                                                    # 恢复训练: 设为last.pt路径则从断点续训
    # AMP = False                                                                    # 关闭自动混合精度 (默认开启, 可加速训练)
    # FRACTION = 0.2                                                                 # 使用数据集的比例, <1.0表示只取部分数据
    # PAIRS_RGB_IR = ['visible', 'infrared']                                         # 双模态文件配对规则:
                                                                                     #   ['visible','infrared'] - FLIR格式
                                                                                     #   ['rgb', 'ir']          - 其他格式
                                                                                     #   ['images', 'images_ir'] - 其他格式
                                                                                     #   ['images', 'image']    - 其他格式
    # VAL = True                                                                     # 训练完成后自动验证 (默认开启)

    # ============================================================================
    # 训练结果保存路径
    # ============================================================================
    PROJECT = R'C:\Users\Patrick\Desktop\DeepLearning\Code\YOLOv11to26-RGBT\runs\LLVIP\26dual-demo'  # 项目保存根目录
    NAME = 'yolo26n-RGBRGB6C-midfusion'                                              # 本次运行的名称, 结果保存在 PROJECT/NAME/

    # ============================================================================
    # 训练开始 (可注释掉 .load() 以不加载预训练权重)
    # ============================================================================
    model = YOLO(MODEL_YAML).load(PRETRAINED)

    # ============================================================================
    # 训练
    # ============================================================================
    model.train(
        data=DATA,
        cache=CACHE,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        close_mosaic=CLOSE_MOSAIC,
        workers=WORKERS,
        device=DEVICE,
        optimizer=OPTIMIZER,
        # lr0=LR0,                   # 取消注释以使用自定义学习率
        # resume=RESUME,             # 取消注释以恢复训练
        # amp=AMP,                   # 取消注释以关闭自动混合精度
        # fraction=FRACTION,         # 取消注释以限制数据使用比例
        # pairs_rgb_ir=PAIRS_RGB_IR, # 取消注释以自定义配对规则
        # val=VAL,                   # 取消注释以控制是否验证
        use_simotm=USE_SIMOTM,
        channels=CHANNELS,
        project=PROJECT,
        name=NAME,
    )
