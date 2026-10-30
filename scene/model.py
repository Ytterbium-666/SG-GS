import torch
import torch.nn as nn
import torch.nn.functional as F


class SHEncoding(nn.Module):
    """
    Spherical Harmonics Encoding.

    Args:
        levels: SH levels. Output dimension = levels ** 2.
                levels=1 -> 1 dim
                levels=2 -> 4 dims
                levels=3 -> 9 dims
                levels=4 -> 16 dims
        implementation: "torch" or "tcnn".
                        If "tcnn" is selected but tiny-cuda-nn is unavailable,
                        it falls back to torch implementation.
    """

    def __init__(self, levels: int = 4, implementation: str = "torch"):
        super().__init__()

        assert levels >= 1 and levels <= 5, "This implementation supports levels from 1 to 5."
        assert implementation in ["torch", "tcnn"], "implementation must be 'torch' or 'tcnn'."

        self.levels = levels
        self.out_dim = levels * levels
        self.implementation = implementation

        self.tcnn_encoding = None

    def get_out_dim(self) -> int:
        return self.out_dim

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            directions: Tensor with shape [..., 3].
                        Usually normalized directions.

        Returns:
            SH encoded tensor with shape [..., levels ** 2].
        """

        if self.implementation == "tcnn" and self.tcnn_encoding is not None:
            original_shape = directions.shape[:-1]

            x = directions.reshape(-1, 3)

            # tiny-cuda-nn generally expects float32 CUDA tensor
            x = x.contiguous()

            encoded = self.tcnn_encoding(x)

            encoded = encoded.reshape(*original_shape, self.out_dim)
            return encoded

        return self._torch_forward(directions)

    def _torch_forward(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Pure PyTorch spherical harmonics encoding.

        The output ordering follows common real SH basis order:
        level 0: 1
        level 1: y, z, x
        level 2: xy, yz, 2z^2 - x^2 - y^2, xz, x^2 - y^2
        level 3: ...
        """

        assert directions.shape[-1] == 3, "Input directions must have shape [..., 3]."

        x = directions[..., 0]
        y = directions[..., 1]
        z = directions[..., 2]

        result = []

        # l = 0
        result.append(torch.full_like(x, 0.28209479177387814))

        if self.levels >= 2:
            # l = 1
            result.append(-0.4886025119029199 * y)
            result.append(0.4886025119029199 * z)
            result.append(-0.4886025119029199 * x)

        if self.levels >= 3:
            # l = 2
            result.append(1.0925484305920792 * x * y)
            result.append(-1.0925484305920792 * y * z)
            result.append(0.31539156525252005 * (3.0 * z * z - 1.0))
            result.append(-1.0925484305920792 * x * z)
            result.append(0.5462742152960396 * (x * x - y * y))

        if self.levels >= 4:
            # l = 3
            result.append(-0.5900435899266435 * y * (3.0 * x * x - y * y))
            result.append(2.890611442640554 * x * y * z)
            result.append(-0.4570457994644658 * y * (5.0 * z * z - 1.0))
            result.append(0.3731763325901154 * z * (5.0 * z * z - 3.0))
            result.append(-0.4570457994644658 * x * (5.0 * z * z - 1.0))
            result.append(1.445305721320277 * z * (x * x - y * y))
            result.append(-0.5900435899266435 * x * (x * x - 3.0 * y * y))

        if self.levels >= 5:
            # l = 4
            result.append(2.5033429417967046 * x * y * (x * x - y * y))
            result.append(-1.7701307697799304 * y * z * (3.0 * x * x - y * y))
            result.append(0.9461746957575601 * x * y * (7.0 * z * z - 1.0))
            result.append(-0.6690465435572892 * y * z * (7.0 * z * z - 3.0))
            result.append(0.10578554691520431 * (35.0 * z ** 4 - 30.0 * z * z + 3.0))
            result.append(-0.6690465435572892 * x * z * (7.0 * z * z - 3.0))
            result.append(0.47308734787878004 * (x * x - y * y) * (7.0 * z * z - 1.0))
            result.append(-1.7701307697799304 * x * z * (x * x - 3.0 * y * y))
            result.append(0.6258357354491761 * (
                x ** 4 - 6.0 * x * x * y * y + y ** 4
            ))

        return torch.stack(result, dim=-1)

class MLP(nn.Module):
    """
    A lightweight PyTorch MLP replacement for nerfstudio.field_components.mlp.MLP.

    Args:
        in_dim: Input feature dimension.
        num_layers: Total number of linear layers.
                    If num_layers=1, it directly maps in_dim -> out_dim.
                    If num_layers>1, it uses hidden layers.
        layer_width: Hidden layer width.
        out_dim: Output dimension.
        activation: Activation function used after hidden layers.
        out_activation: Activation function used after output layer.
        implementation: Kept only for API compatibility. Ignored here.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: int,
        activation: nn.Module = nn.ReLU(),
        out_activation: nn.Module = None,
        implementation: str = "torch",
    ):
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"

        self.in_dim = in_dim
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.out_dim = out_dim
        self.activation = activation
        self.out_activation = out_activation
        self.implementation = implementation

        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            # First layer
            layers.append(nn.Linear(in_dim, layer_width))
            layers.append(self._clone_activation(activation))

            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(layer_width, layer_width))
                layers.append(self._clone_activation(activation))

            # Output layer
            layers.append(nn.Linear(layer_width, out_dim))

        if out_activation is not None:
            layers.append(self._clone_activation(out_activation))

        self.net = nn.Sequential(*layers)

    def _clone_activation(self, activation):
        """
        Avoid reusing the same activation module instance when possible.
        """
        if activation is None:
            return nn.Identity()

        # Most PyTorch activations can be recreated from their class.
        try:
            return activation.__class__()
        except Exception:
            return activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiExponentialAttenuation(nn.Module):
    def __init__(self, num_channels=3):
        super(BiExponentialAttenuation, self).__init__()
        
        # 将 a, b, c, d 定义为可学习的参数 (nn.Parameter)
        # 初始化时给予一些合理的随机值。
        # 注意：这里 c 变量名与公式中的 c (通道) 冲突，代码里用 c_param 代替
        self.raw_a = nn.Parameter(torch.randn(1, num_channels))
        self.raw_b = nn.Parameter(torch.randn(1, num_channels))
        self.raw_c = nn.Parameter(torch.randn(1, num_channels))
        self.raw_d = nn.Parameter(torch.randn(1, num_channels))
    
    def get_physical_params(self):
        """
        辅助函数：将底层参数转换为真实具有物理意义的 a, b, c, d
        """
        # a, c 必须为正数
        a = F.softplus(self.raw_a)
        c = F.softplus(self.raw_c)
        
        # b, d 必须为负数 (注意前面的负号)
        # 加上一个小常数 (如 1e-6) 防止衰减系数绝对值为 0
        b = - (F.softplus(self.raw_b) + 1e-6)
        d = - (F.softplus(self.raw_d) + 1e-6)
        
        return a, b, c, d

    def forward(self, z):
        """
        输入:
            z: 深度张量, 形状可以是 (N,) 或 (N, 1)，N 是像素或样本数量
        输出:
            beta: 对应深度的衰减系数，形状为 (N, 3)
        """
        a, b, c, d = self.get_physical_params()
        
        # 严格按照公式计算：beta(z) = a * exp(b * z) + c * exp(d * z)
        term1 = a * torch.exp(b * z)
        term2 = c * torch.exp(d * z)
        
        # 最终输出形状为 (B, C, H, W)
        return term1 + term2