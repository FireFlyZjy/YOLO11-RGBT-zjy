"""
RT-SFOD DHF 推理脚本

论文: RT-SFOD (ECCV 2026)
策略: Dual-Head pseudo-label Fusion (DHF)

使用方式:
    D:\Anaconda\envs\torch310\python.exe ultralytics/cfg/models/26-RGBT/2026-07-05/train/inference_DHF.py \
        --model ultralytics/cfg/models/26-RGBT/2026-07-05/frequency/yolo26-RGBT-midfusion-uPCAD.yaml \
        --weights runs/detect/train/weights/best.pt \
        --source path/to/images \
        --device 0

推理策略说明:
    1. DHF 融合: 利用 YOLO 的 one2one 和 one2many 双头输出
    2. one2one 头: 高置信度锚点
    3. one2many 头: 补充检测
    4. 去重融合: 移除重叠框，保留互补检测
"""

import argparse
import sys
from pathlib import Path

import torch

# 添加项目根目录到路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]  # YOLOv11to26-RGBT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.ExtraModules.loss.RT_SFOD_Loss import DualHeadFusion


class DHFInference:
    """DHF 推理器: 使用双头融合进行推理"""

    def __init__(self, model_path: str, weights_path: str = None, device: str = '0',
                 tau_o2o: float = 0.5, tau_o2m: float = 0.5,
                 tau_no: float = 0.2, tau_dup: float = 0.7):
        """
        初始化 DHF 推理器

        参数:
            model_path: YAML 配置文件路径
            weights_path: 权重文件路径 (可选)
            device: 设备 ('0', '1', 'cpu')
            tau_o2o: one2one 头置信度阈值
            tau_o2m: one2many 头置信度阈值
            tau_no: 新框 IoU 阈值
            tau_dup: 去重 IoU 阈值
        """
        self.model_path = model_path
        self.weights_path = weights_path
        self.device = device

        # DHF 融合参数
        self.tau_o2o = tau_o2o
        self.tau_o2m = tau_o2m
        self.tau_no = tau_no
        self.tau_dup = tau_dup

        # 创建 DHF 融合器
        self.dhf = DualHeadFusion(
            tau_o2o=tau_o2o,
            tau_o2m=tau_o2m,
            tau_no=tau_no,
            tau_dup=tau_dup
        )

        # 加载模型
        self.model = YOLO(model_path)
        if weights_path:
            self.model.load(weights_path)

    def predict_standard(self, source: str, conf: float = 0.25, iou: float = 0.45, **kwargs):
        """
        标准预测 (不使用 DHF 融合)

        参数:
            source: 图像/视频/目录路径
            conf: 置信度阈值
            iou: NMS IoU 阈值
            **kwargs: 其他预测参数
        """
        print("=" * 60)
        print("RT-SFOD DHF 推理脚本")
        print("=" * 60)
        print(f"模型: {self.model_path}")
        print(f"权重: {self.weights_path or '默认'}")
        print(f"设备: {self.device}")
        print(f"推理模式: 标准预测")
        print("=" * 60)

        # 标准预测
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            **kwargs
        )

        print("=" * 60)
        print("预测完成！")
        print("=" * 60)

        return results

    def predict_with_dhf_demo(self, source: str, conf: float = 0.25, **kwargs):
        """
        带 DHF 融合的预测演示

        注意: 此函数展示如何使用 DHF 融合，但需要模型支持双头输出
        """
        print("=" * 60)
        print("RT-SFOD DHF 推理脚本 (带 DHF 融合)")
        print("=" * 60)
        print(f"模型: {self.model_path}")
        print(f"权重: {self.weights_path or '默认'}")
        print(f"设备: {self.device}")
        print(f"DHF 参数: tau_o2o={self.tau_o2o}, tau_o2m={self.tau_o2m}")
        print(f"          tau_no={self.tau_no}, tau_dup={self.tau_dup}")
        print("=" * 60)

        # 注意: 要真正使用 DHF 融合，需要:
        # 1. 模型支持 one2one 和 one2many 双头输出
        # 2. 修改推理逻辑，分别获取两个头的输出
        # 3. 使用 DHF 融合器合并结果

        print("\n[说明] DHF 融合集成方式:")
        print("1. 模型需要支持 one2one 和 one2many 双头输出")
        print("2. 获取 one2one 头输出: boxes_one2one = model.predict_one2one(source)")
        print("3. 获取 one2many 头输出: boxes_one2many = model.predict_one2many(source)")
        print("4. 融合: fused_boxes = dhf(boxes_one2one, boxes_one2many)")
        print("\n当前使用标准预测，DHF 融合未集成到推理循环中。")

        # 标准预测
        results = self.model.predict(
            source=source,
            conf=conf,
            device=self.device,
            **kwargs
        )

        print("=" * 60)
        print("预测完成！")
        print("=" * 60)

        return results


def main():
    parser = argparse.ArgumentParser(description="RT-SFOD DHF 推理脚本")
    parser.add_argument("--model", type=str, required=True,
                        help="YAML 配置文件路径")
    parser.add_argument("--weights", type=str, default=None,
                        help="权重文件路径")
    parser.add_argument("--source", type=str, required=True,
                        help="图像/视频/目录路径")
    parser.add_argument("--device", type=str, default="0",
                        help="设备 ('0', '1', 'cpu')")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值")
    parser.add_argument("--tau_o2o", type=float, default=0.5,
                        help="one2one 头置信度阈值")
    parser.add_argument("--tau_o2m", type=float, default=0.5,
                        help="one2many 头置信度阈值")
    parser.add_argument("--tau_no", type=float, default=0.2,
                        help="新框 IoU 阈值")
    parser.add_argument("--tau_dup", type=float, default=0.7,
                        help="去重 IoU 阈值")
    parser.add_argument("--use_dhf", action="store_true",
                        help="是否使用 DHF 融合 (需要模型支持双头输出)")
    args = parser.parse_args()

    # 创建推理器
    inferencer = DHFInference(
        model_path=args.model,
        weights_path=args.weights,
        device=args.device,
        tau_o2o=args.tau_o2o,
        tau_o2m=args.tau_o2m,
        tau_no=args.tau_no,
        tau_dup=args.tau_dup
    )

    # 开始推理
    if args.use_dhf:
        inferencer.predict_with_dhf_demo(
            source=args.source,
            conf=args.conf
        )
    else:
        inferencer.predict_standard(
            source=args.source,
            conf=args.conf,
            iou=args.iou
        )


if __name__ == "__main__":
    main()
