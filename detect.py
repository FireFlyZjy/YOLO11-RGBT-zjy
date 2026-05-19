import argparse
import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

"""
通用检测/预测脚本，支持多种模态配置。

Compared with the original YOLOv11 detect, this script adds two parameters:
    use_simotm: 模态组合方式
    channels:   输入通道数

使用示例:
    # RGB 3通道 (默认)
    python detect.py --weights best.pt --source path/to/images

    # Gray 1通道
    python detect.py --weights best.pt --source path/to/images --use_simotm Gray --channels 1

    # RGBT 4通道
    python detect.py --weights best.pt --source path/to/images --use_simotm RGBT --channels 4

    # RGBRGB6C 6通道
    python detect.py --weights best.pt --source path/to/images --use_simotm RGBRGB6C --channels 6

    # 多光谱
    python detect.py --weights best.pt --source path/to/images --use_simotm Multispectral_16bit --channels 7

    # OBB 任务
    python detect.py --weights best.pt --source path/to/images --use_simotm RGBT --channels 4 --task obb
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO 通用检测脚本 — 支持多模态、多通道")
    # 模型
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径 (.pt)")
    # 数据源
    parser.add_argument("--source", type=str, required=True, help="检测图像/视频/目录路径")
    # 模态
    parser.add_argument("--use_simotm", type=str, default="RGB",
                        choices=["RGB", "Gray", "RGBT", "RGBRGB6C", "SimOTMBBS", "RGBT_IR", "RGBRGB",
                                 "Multispectral", "Multispectral_16bit"],
                        help="模态组合方式 (default: RGB)")
    parser.add_argument("--channels", type=int, default=3, help="输入通道数 (default: 3)")
    # 任务
    parser.add_argument("--task", type=str, default="detect", help="任务类型: detect, obb, segment, pose 等")
    # 检测参数
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸 (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (default: 0.25)")
    parser.add_argument("--project", type=str, default="runs/detect", help="结果保存目录")
    parser.add_argument("--name", type=str, default="exp", help="实验名称")
    parser.add_argument("--save", type=lambda x: x.lower() == 'true', default=True, help="是否保存结果")
    parser.add_argument("--show", type=lambda x: x.lower() == 'true', default=False, help="是否实时显示")
    parser.add_argument("--save_frames", type=lambda x: x.lower() == 'true', default=False, help="是否保存帧")
    parser.add_argument("--visualize", type=lambda x: x.lower() == 'true', default=False,
                        help="是否可视化模型特征图")

    args = parser.parse_args()

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        show=args.show,
        save_frames=args.save_frames,
        use_simotm=args.use_simotm,
        channels=args.channels,
        save=args.save,
        conf=args.conf,
        visualize=args.visualize,
        task=args.task,
    )
