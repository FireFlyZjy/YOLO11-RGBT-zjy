import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR

"""
模型信息打印 & 训练模板脚本

功能说明:
    1. 加载 YAML 模型配置，打印模型结构、参数量、FLOPs 等详细信息
    2. 保留了一份完整的训练配置模板（注释状态），方便快速复制使用

使用方式:
    - 直接运行: 仅打印模型信息，不进行训练
    - 取消注释 train() 部分: 可直接作为训练脚本使用，按需修改参数

model.info() 参数说明:
    model.info(verbose=True, detailed=True)
    - verbose=True : 打印每层的详细参数（输入输出shape、参数量等）
    - detailed=True: 打印更详细的层信息（层类型、参数分布等）
    返回值: (n_layers, n_params, n_gradients, flops)
"""

if __name__ == '__main__':
    # ============================================================================
    # 模型信息打印
    # 加载 YAML 配置，打印模型的详细结构、参数量(M)、FLOPs(G)
    # 用于快速查看模型结构，不进行训练
    # ============================================================================
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')           # 模型 YAML 配置文件，可替换为其他模型

    # 也可以直接加载训练好的权重文件查看信息
    # model = YOLO('runs/dota8/dota8-yolo11n-RGBT-midfusion-obb-e300-16-27/weights/best.pt')

    # 打印模型详细信息
    # verbose=True  : 逐层打印参数
    # detailed=True : 打印额外细节
    model.info(verbose=True, detailed=True)

    # ============================================================================
    # 训练配置模板（以下全部注释，需要训练时取消注释并按需修改）
    # 这是一个 RGBT 4通道 midfusion 的训练示例
    # ============================================================================

    # model.load('yolov8n.pt')                                        # 加载预训练权重（可选）
    # model.train(
    #     data=R'ultralytics/cfg/datasets/M3FD.yaml',                 # 数据集 YAML 路径
    #     cache=False,                                                 # 是否缓存到内存，True 加速但耗内存
    #     imgsz=640,                                                   # 输入图像尺寸
    #     epochs=300,                                                  # 训练轮数
    #     batch=32,                                                    # 批次大小，显存不足时减小
    #     close_mosaic=0,                                              # 最后N个epoch关闭mosaic增强，0=全程关闭
    #     workers=2,                                                   # 数据加载线程数，Windows下建议0
    #     device='0',                                                  # GPU设备号，'cpu'表示使用CPU
    #     optimizer='SGD',                                             # 优化器: SGD/Adam/AdamW
    #     # lr0=0.002,                                                 # 初始学习率（注释则使用默认值）
    #     # resume='runs/.../weights/last.pt',                         # 断点续训: 填入last.pt路径
    #     # amp=False,                                                 # 关闭自动混合精度（默认开启）
    #     # fraction=0.2,                                              # 仅使用部分数据（<1.0），调试用
    #     # pairs_rgb_ir=['visible','infrared'],                       # 双模态文件配对规则
    #     use_simotm="RGBT",                                           # 模态组合: RGBT=4通道[RGB+IR]
    #     channels=4,                                                  # 输入通道数
    #     project='runs/M3FD',                                         # 结果保存根目录
    #     name='M3FD-yolo11-RGBT-midfusion-CAS-',                      # 本次实验名称
    # )
