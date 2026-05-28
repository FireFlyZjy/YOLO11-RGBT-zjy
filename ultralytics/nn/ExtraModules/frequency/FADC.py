import torch
import torch.nn as nn
import torch.nn.functional as F


class FADC(nn.Module):
    """FADC: Frequency Adaptive Dilation Convolution (频率自适应空洞卷积)

    论文: CVPR2024 - Frequency-Adaptive Dilated Convolution
    核心机制:
      1. 利用梯度幅值作为"频率内容"的代理: 高梯度 = 高频(细节),
         低梯度 = 低频(平滑区域)
      2. 根据频率内容自适应选择空洞率:
         - 高频区域 (细节/边缘): 小空洞率 (d=1), 保留精细信息
         - 低频区域 (平坦/背景): 大空洞率 (d=2,3), 扩大感受野
      3. 多个空洞卷积分支 (dilation=1,2,3) 以 softmax 加权融合
      4. 自适应核增强 (AKE): 利用频率特征对卷积输出做通道重标定

    为什么适用于 RGBT 检测:
      - 红外模态噪声多/细节少: 倾向于大空洞率, 捕捉更大范围上下文
      - RGB 模态细节多: 倾向于小空洞率, 保留精细纹理
      - 频率自适应让网络在每幅图的每个位置选择最优空洞率
      - 参数量仅略微增加, 适合 YOLO 轻量化框架

    YAML 使用示例:
      - [26, FADC, [128, 256, 3, 2]]           # 替代标准 Conv
      - [26, FADC, [256, 256, 3, 1, [1,2,3]]]  # 自定义空洞率
    """

    def __init__(self, c1, c2, k=3, s=1, dilations=None, act=True):
        """Initialize FADC.
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小 (default=3)
            s: 步长 (default=1)
            dilations: 空洞率列表 (default=[1,2,3])
            act: 激活函数 (default=True -> SiLU)
        """
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 3]
        self.dilations = dilations
        self.num_branches = len(dilations)

        # ===== 频率内容估计器 =====
        # 输入: 梯度幅值 (通过 Sobel 或 Laplace 计算)
        # 输出: 每个分支的权重图 (B, num_branches, H, W)

        # 简化的 Sobel 核 (检测高频边缘)
        self.register_buffer(
            "sobel_kernel",
            torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32),
        )

        # 频率特征编码器: 从梯度 + 原始特征预测空洞权重
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(c1 + 1, max(c1 // 8, 4), 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c1 // 8, 4), self.num_branches, 3, padding=1, bias=False),
        )

        # ===== 空洞卷积分支 =====
        self.branches = nn.ModuleList()
        for d in dilations:
            # padding 保持空间尺寸
            p = d * (k - 1) // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(c1, c2, k, s, p, dilation=d, bias=False),
                    nn.BatchNorm2d(c2),
                )
            )

        # ===== 自适应核增强 (AKE) =====
        # 类似 SENet, 但从"频率级"做通道注意力
        self.ake = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(c2 // 4, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c2 // 4, 4), c2, 1, bias=False),
            nn.Sigmoid(),
        )

        # 最终激活
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def _compute_gradient_magnitude(self, x):
        """计算空间梯度幅值, 作为频率内容的代理.

        高梯度 -> 高频 (细节/边缘)
        低梯度 -> 低频 (平滑区域)
        """
        # Sobel X
        sobel = self.sobel_kernel.to(x.device)  # (1, 1, 3, 3)
        # 对每个通道分别做边缘检测, 取平均值
        grad_x = F.conv2d(
            x.view(-1, 1, x.shape[2], x.shape[3]),
            sobel,
            padding=1,
        ).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        # Sobel Y (旋转 90 度)
        sobel_y = sobel.transpose(-2, -1)
        grad_y = F.conv2d(
            x.view(-1, 1, x.shape[2], x.shape[3]),
            sobel_y,
            padding=1,
        ).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        # 梯度幅值, 通道间平均
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        grad_magnitude = grad_magnitude.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        return grad_magnitude

    def forward(self, x):
        """前向传播:
        梯度估计 -> 频率编码 -> softmax 权重 -> 多分支加权融合 -> AKE -> 激活
        """
        B, C, H, W = x.shape

        # ===== 1. 频率内容估计 =====
        grad = self._compute_gradient_magnitude(x)  # (B, 1, H, W)
        freq_feat = torch.cat([x, grad], dim=1)     # (B, C+1, H, W)
        weights = self.freq_encoder(freq_feat)       # (B, num_branches, H, W)

        # 在分支维度上做 Softmax: 每像素各分支权重和为 1
        weights = torch.softmax(weights, dim=1)  # (B, num_branches, H, W)

        # ===== 2. 多分支空洞卷积 (w/ stride 处理) =====
        # 注意: 当 s>1 时, 输出分辨率降低, 需要插值对齐权重
        outs = []
        for i, branch in enumerate(self.branches):
            out_i = branch(x)            # (B, c2, H_out, W_out)
            w_i = weights[:, i:i+1]      # (B, 1, H, W)
            if out_i.shape[-2:] != w_i.shape[-2:]:
                # stride 导致分辨率变化, 对权重做下采样
                w_i = F.interpolate(
                    w_i, size=out_i.shape[-2:], mode="nearest"
                )
            outs.append(out_i * w_i)     # 加权

        # 融合所有分支
        out = sum(outs)  # (B, c2, H_out, W_out)

        # ===== 3. 自适应核增强 =====
        # 根据频率特征对输出做通道重标定
        scale = self.ake(out)  # (B, c2, 1, 1)
        out = out * scale

        # 激活
        out = self.act(out)
        return out

    def __repr__(self):
        return (f"FADC(c1={self.branches[0][0].in_channels}, "
                f"c2={self.branches[0][0].out_channels}, "
                f"dilations={self.dilations})")
