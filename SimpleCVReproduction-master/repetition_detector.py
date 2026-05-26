#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repetition_detector.py
检测模型输出中是否存在规律性重复内容。
支持扩展：只需继承 BaseDetector 并实现 `score()` 即可。
"""

import argparse
import json
import math
from collections import Counter
from typing import List, Tuple

class BaseDetector:
    """所有子检测器的基类"""
    def score(self, text: str) -> float:
        """
        返回 0~1 之间的异常分，越大越像重复。
        """
        raise NotImplementedError

class NgramRepeatDetector(BaseDetector):
    def __init__(self, n: int = 5, min_repeat: int = 3):
        self.n = n
        self.min_repeat = min_repeat  # 一个 n-gram 至少出现这么多次才计入

    def score(self, text: str) -> float:
        tokens = text.split()
        if len(tokens) < self.n:
            return 0.0

        ngrams = [tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)]
        counter = Counter(ngrams)
        if not counter:
            return 0.0

        # 计算重复 token 占比
        repeated_tok_cnt = sum(cnt for cnt in counter.values() if cnt >= self.min_repeat) * self.n
        ratio = repeated_tok_cnt / len(tokens)
        return min(1.0, ratio)


# ======================
# 具体检测器 2：字符级 Shannon 熵
# ======================
class EntropyDetector(BaseDetector):
    def __init__(self, char_level: bool = True):
        self.char_level = char_level

    def score(self, text: str) -> float:
        seq = list(text) if self.char_level else text.split()
        if not seq:
            return 0.0
        counter = Counter(seq)
        total = len(seq)
        entropy = -sum((c / total) * math.log2(c / total) for c in counter.values())
        # 经验阈值：英文文本熵<3 bits/char 时很可能重复
        max_ent = 4.5  # 经验值，可调
        ratio = max(0.0, max_ent - entropy) / max_ent
        return min(1.0, ratio)


# ======================
# 组合器
# ======================
class EnsembleDetector:
    def __init__(self, detectors: List[Tuple[BaseDetector, float]]):
        """
        detectors: [(detector_instance, weight), ...]
        """
        self.detectors = detectors

    def score(self, text: str) -> float:
        total = 0.0
        weight_sum = 0.0
        for det, w in self.detectors:
            total += det.score(text) * w
            weight_sum += w
        return total / weight_sum if weight_sum else 0.0


# ======================
# 主流程
# ======================
def main(input_path: str, threshold: float = 0.35, mode: str = 'eval'):
    # 初始化子检测器及权重
    detectors = [
        (NgramRepeatDetector(n=5, min_repeat=3), 0.6),
        (EntropyDetector(char_level=True), 0.4),
    ]
    ensemble = EnsembleDetector(detectors)

    if mode == 'eval':
        preds, gts = [], []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data["messages"][-1]["content"]  # 取 assistant 最后一条
                gt = data.get("gt", 0)

                score = ensemble.score(text)
                pred = 1 if score >= threshold else 0

                preds.append(pred)
                gts.append(gt)
                # 打印逐样本信息
                print(json.dumps({"score": round(score, 4),
                                "pred": pred,
                                "gt": gt},
                                ensure_ascii=False))
            
        # 计算准确率
        acc = sum(p == g for p, g in zip(preds, gts)) / len(gts) if gts else 0.0
        print(f"\nAccuracy: {acc:.4f}")

    elif mode == 'class':
        # 单纯分类，并统计出现的频率有多少 即 pred=1 的情况
        num_pred_1 = 0
        num_total = 0
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data["messages"][-1]["content"]
                score = ensemble.score(text)
                pred = 1 if score >= threshold else 0
                num_total += 1
                if pred == 1:
                    num_pred_1 += 1

        print(f"Invalid ratio: {num_pred_1 / num_total:.4f} ({num_pred_1}/{num_total})")
    elif mode == 'conflict':
        # 计算比例，pred=1, 但是 reward=1 的比例
        num_confict = 0
        num_total = 0
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data["messages"][-1]["content"]
                reward = data.get("reward", 1)  # 默认1
                score = ensemble.score(text)
                pred = 1 if score >= threshold else 0
                num_total += 1
                if reward == 1 and pred == 1:
                    num_confict += 1
        
        print(f"Conflict Ratio: {num_confict / num_total:.4f} ({num_confict}/{num_total})")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="异常分阈值，>=阈值判为重复")
    parser.add_argument("--mode", type=str, default='eval', choices=['eval', 'class', 'conflict'],help="eval 测评模式 (w/ gt) class 分类模式 w/o gt")
    args = parser.parse_args()
    main(args.input_jsonl, args.threshold, args.mode)
