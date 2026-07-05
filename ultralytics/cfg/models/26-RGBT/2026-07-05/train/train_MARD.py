"""
RT-SFOD MARD 训练脚本

论文: RT-SFOD (ECCV 2026)
策略: Multi-scale Adaptive Representation Diversification (MARD)

使用方式:
    D:\Anaconda\envs\torch310\python.exe ultralytics/cfg/models/26-RGBT/2026-07-05/train/train_MARD.py \
        --model ultralytics/cfg/models/26-RGBT/2026-07-05/frequency/yolo26-RGBT-midfusion-uPCAD.yaml \
        --data ultralytics/cfg/datasets/flir.yaml \
        --epochs 100 --batch 4 --device 0

训练策略说明:
    1. MARD 损失: 在 P3/P4/P5 多尺度特征上应用方差+协方差正则化
    2. 方差损失: 鼓励特征多样性，避免特征坍缩
    3. 协方差损失: 减少特征冗余，提升特征效率
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 添加项目根目录到路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]  # YOLOv11to26-RGBT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.ExtraModules.loss.RT_SFOD_Loss import MARD_Loss


class MARDTrainer:
    """MARD 训练器: 在标准训练基础上添加 MARD 正则化损失"""

    def __init__(self, model_path: str, data_path: str, device: str = '0',
                 mard_alpha: float = 1.0, mard_beta: float = 0.1,
                 mard_gamma: float = 1.0, mard_weight: float = 0.05):
        """
        初始化 MARD 训练器

        参数:
            model_path: YAML 配置文件路径
            data_path: 数据集配置文件路径
            device: 设备 ('0', '1', 'cpu')
            mard_alpha: 方差损失权重
            mard_beta: 协方差损失权重
            mard_gamma: 方差阈值
            mard_weight: MARD 损失总权重
        """
        self.model_path = model_path
        self.data_path = data_path
        self.device = device

        # MARD 损失参数
        self.mard_alpha = mard_alpha
        self.mard_beta = mard_beta
        self.mard_gamma = mard_gamma
        self.mard_weight = mard_weight

        # 创建 MARD 损失函数
        self.mard_loss_fn = MARD_Loss(
            alpha=mard_alpha,
            beta=mard_beta,
            gamma=mard_gamma
        )

    def train_standard(self, epochs: int = 100, batch: int = 4, **kwargs):
        """
        标准训练 (不使用 MARD 损失)

        参数:
            epochs: 训练轮数
            batch: 批次大小
            **kwargs: 其他训练参数
        """
        print("=" * 60)
        print("RT-SFOD MARD 训练脚本")
        print("=" * 60)
        print(f"模型: {self.model_path}")
        print(f"数据: {self.data_path}")
        print(f"设备: {self.device}")
        print(f"训练模式: 标准训练 (MARD 损失需要自定义训练循环)")
        print("=" * 60)

        # 加载模型
        model = YOLO(self.model_path)

        # 标准训练
        model.train(
            data=self.data_path,
            epochs=epochs,
            batch=batch,
            device=self.device,
            workers=0,
            use_simotm='RGBT',
            channels=4,
            close_mosaic=0,
            **kwargs
        )

        print("=" * 60)
        print("训练完成！")
        print("=" * 60)

    def train_with_mard_demo(self, epochs: int = 100, batch: int = 4, **kwargs):
        """
        带 MARD 损失的训练演示 (需要自定义训练循环)

        注意: 此函数展示如何集成 MARD 损失，但需要修改 ultralytics 训练循环
        """
        print("=" * 60)
        print("RT-SFOD MARD 训练脚本 (带 MARD 损失)")
        print("=" * 60)
        print(f"模型: {self.model_path}")
        print(f"数据: {self.data_path}")
        print(f"设备: {self.device}")
        print(f"MARD 参数: alpha={self.mard_alpha}, beta={self.mard_beta}, gamma={self.mard_gamma}")
        print(f"MARD 权重: {self.mard_weight}")
        print("=" * 60)

        # 加载模型
        model = YOLO(self.model_path)

        # 注意: 要真正使用 MARD 损失，需要修改 ultralytics 的训练循环
        # 这里展示如何获取中间特征并计算 MARD 损失

        print("\n[说明] MARD 损失集成方式:")
        print("1. 在模型的 forward 中获取 P3/P4/P5 特征")
        print("2. 计算 MARD 损失: mard_result = mard_loss_fn(feats)")
        print("3. 添加到总损失: total_loss = det_loss + mard_weight * mard_result['mard_loss']")
        print("\n当前使用标准训练，MARD 损失未集成到训练循环中。")
        print("如需完整集成，请参考 RT-SFOD 论文的训练策略。")

        # 标准训练
        model.train(
            data=self.data_path,
            epochs=epochs,
            batch=batch,
            device=self.device,
            workers=0,
            use_simotm='RGBT',
            channels=4,
            close_mosaic=0,
            **kwargs
        )

        print("=" * 60)
        print("训练完成！")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RT-SFOD MARD 训练脚本")
    parser.add_argument("--model", type=str, required=True,
                        help="YAML 配置文件路径")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/flir.yaml",
                        help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--device", type=str, default="0",
                        help="设备 ('0', '1', 'cpu')")
    parser.add_argument("--mard_alpha", type=float, default=1.0,
                        help="方差损失权重")
    parser.add_argument("--mard_beta", type=float, default=0.1,
                        help="协方差损失权重")
    parser.add_argument("--mard_gamma", type=float, default=1.0,
                        help="方差阈值")
    parser.add_argument("--mard_weight", type=float, default=0.05,
                        help="MARD 损失总权重")
    parser.add_argument("--use_mard", action="store_true",
                        help="是否使用 MARD 损失 (需要自定义训练循环)")
    args = parser.parse_args()

    # 创建训练器
    trainer = MARDTrainer(
        model_path=args.model,
        data_path=args.data,
        device=args.device,
        mard_alpha=args.mard_alpha,
        mard_beta=args.mard_beta,
        mard_gamma=args.mard_gamma,
        mard_weight=args.mard_weight
    )

    # 开始训练
    if args.use_mard:
        trainer.train_with_mard_demo(
            epochs=args.epochs,
            batch=args.batch
        )
    else:
        trainer.train_standard(
            epochs=args.epochs,
            batch=args.batch
        )


if __name__ == "__main__":
    main()
