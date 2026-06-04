# coding=utf-8
"""
SAM — Sharpness-Aware Minimization Optimizer

论文: Sharpness-Aware Minimization for Efficiently Improving Generalization (ICLR 2021)
      https://arxiv.org/abs/2010.01412

机制:
    SAM 是一种优化器封装, 通过在参数空间中寻找平坦区域 (flat minima) 来提高泛化能力.
    传统优化器 (SGD, AdamW) 倾向于收敛到尖锐最小值 (sharp minima), 对参数扰动敏感,
    泛化性能差. SAM 通过两步法寻找平坦最小值:

    1. first_step (上升):
       计算梯度 grad, 沿梯度方向扰动参数:
           w_perturbed = w + rho * grad / ||grad||
       这相当于寻找损失景观中"最陡"的点.

    2. second_step (下降):
       在扰动后的位置计算新梯度, 用该梯度更新原始参数.
       这个梯度的方向指引模型走向平坦区域.

    数学表达:
        min_w  max_{||epsilon|| <= rho}  L(w + epsilon)
        ≈ min_w  L(w + rho * grad/||grad||)

    其中 rho 控制邻域大小 (默认 0.05), 也是唯一需要调的超参数.

重要提示:
    SAM 是训练工具 (Training Utility), 不是网络模块. 它包装一个基础优化器
    (如 SGD, AdamW) 并在训练循环中通过 first_step / second_step 实现 SAM 算法.
    不参与模型前向传播, 仅用于训练阶段.

用法:

    # ----- 标准两步式 (推荐) -----
    from ultralytics.nn.ExtraModules.optim import SAM

    # 初始化: 包装基础优化器
    optimizer = SAM(
        model.parameters(),
        base_optimizer=torch.optim.SGD,  # 或 torch.optim.AdamW
        rho=0.05,                        # 邻域大小
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )

    # 训练循环
    for images, labels in dataloader:
        # 第一步: 正常前向 + 反向, 然后上升扰动
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # 第二步: 在扰动后参数上前向 + 反向, 然后真实更新
        loss2 = criterion(model(images), labels)
        loss2.backward()
        optimizer.second_step(zero_grad=True)

    # ----- 闭包式 (自动两步) -----
    def closure():
        loss = criterion(model(images), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    # ----- 集成到 YOLO 训练 -----
    # 替换默认优化器:
    #   from ultralytics import YOLO
    #   model = YOLO('yolo11n.pt')
    #   model.optimizer = SAM(model.model.parameters(),
    #                          base_optimizer=torch.optim.SGD,
    #                          rho=0.05, lr=0.01, momentum=0.937)
    #   # 在训练循环中手动调用 first_step / second_step

RGBT 价值:
    多模态 (RGB + Thermal) 检测任务中, 不同模态的数据分布差异可能导致
    损失景观崎岖不平. SAM 的平坦最小值搜索策略可以有效缓解模态间的
    优化冲突, 提升模型在跨模态场景下的泛化鲁棒性.
"""

import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization 优化器封装.

    SAM 不直接实现参数更新, 而是包装一个 base_optimizer (如 SGD, AdamW),
    通过 two-step 机制在参数空间中寻找平坦区域.

    Args:
        params (iterable): 模型参数或参数组
        base_optimizer (torch.optim.Optimizer class): 基础优化器类 (非实例)
        rho (float): 邻域大小 (扰动幅度), 默认 0.05. 建议范围 0.01-0.1
        **kwargs: 传递给基础优化器的参数 (如 lr, momentum, weight_decay)

    Attributes:
        base_optimizer (torch.optim.Optimizer): 基础优化器实例

    .. note::
        SAM 与学习率调度器一起使用时, 应调度 base_optimizer 的学习率.
        例如: scheduler.step() 会作用于 base_optimizer 的 param_groups.
    """

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        """
        初始化 SAM 优化器.

        Args:
            params: 模型参数 (model.parameters())
            base_optimizer: 基础优化器类, 如 torch.optim.SGD
            rho: 邻域大小, 控制扰动幅度
            **kwargs: 传递给基础优化器的参数

        Example:
            >>> optimizer = SAM(model.parameters(), torch.optim.SGD,
            ...                 rho=0.05, lr=0.01, momentum=0.9)
        """
        assert rho >= 0.0, f"rho 必须为非负数, 当前值: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)

        # 创建基础优化器实例 (使用当前 param_groups)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        SAM 的第一步: 梯度上升 (寻找锐利区域).

        根据当前梯度, 将参数沿梯度方向扰动 rho / ||grad|| 的距离:
            w_perturbed = w + rho * grad / ||grad||

        同时保存扰动向量 e_w, 供 second_step 恢复参数.

        Args:
            zero_grad: 是否在第一步后将梯度清零 (默认 False)
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p.device)  # 计算扰动
                p.add_(e_w)                         # 上升: w -> w + e(w)
                self.state[p]["e_w"] = e_w          # 保存扰动向量

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        SAM 的第二步: 参数恢复 + 基础优化器更新.

        将参数从 w + e(w) 恢复到 w, 然后使用基础优化器执行更新:
            w_{t+1} = base_optimizer(w_t, grad|_{w+e(w)})

        Args:
            zero_grad: 是否在第二步后将梯度清零 (默认 False)
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # 恢复: w + e(w) -> w

        self.base_optimizer.step()  # 执行 Sharpness-Aware 更新

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        一步 SAM 更新 (闭包模式).

        如果提供了 closure, 自动执行 first_step + second_step.
        如果未提供 closure, 使用标准优化器的 step (失效倒退).

        Args:
            closure (callable, optional): 计算损失并反向传播的闭包函数

        Example:
            >>> def closure():
            ...     loss = criterion(model(images), labels)
            ...     loss.backward()
            ...     return loss
            >>> optimizer.step(closure)
        """
        if closure is None:
            # 兜底: 使用基础优化器 step (但会失去 SAM 效果)
            self.base_optimizer.step()
            return

        closure = torch.enable_grad()(closure)  # 确保 closure 在梯度上下文中执行

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """
        计算所有参数梯度的全局 L2 范数.

        Returns:
            torch.Tensor: 梯度 L2 范数 (标量)
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def __repr__(self):
        return f"SAM(base_optimizer={self.base_optimizer.__class__.__name__}, rho={self.param_groups[0]['rho']})"
