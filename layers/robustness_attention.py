import torch
import torch.nn as nn
from utils import init_weights  

class RobustnessAttention(nn.Module):
    def __init__(self,
                input_dim: int,
                hidden_dim: int) -> None:
        super().__init__()
        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4"
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim + 1, hidden_dim)  # +1 for violation score
        self.value = nn.Linear(input_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(hidden_dim, 8)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.apply(init_weights)  

    def forward(self, traj, robustness):
        """
        traj: [N, K, F, 2] 轨迹坐标
        robustness: [N, K] 违规分数
        输出: [N, K, F, 2] 轨迹调整量
        """
        N, K, F, _ = traj.shape
        traj_flat = traj.view(N*K, F, 2)
        robustness_flat = robustness.view(N*K, 1)
        
        # 拼接特征
        k_input = torch.cat([
            traj_flat, 
            robustness_flat.unsqueeze(1).expand(-1, F, -1)
        ], dim=-1)  # [N*K, F, 3]
        
        # 注意力计算
        q = self.query(traj_flat).transpose(0,1)  # [F, N*K, D]
        k = self.key(k_input).transpose(0,1)
        v = self.value(traj_flat).transpose(0,1)
        attn_out, _ = self.mha(q, k, v)  # [F, N*K, D]
        
        # 输出调整量
        delta = self.out_proj(attn_out.transpose(0,1))  # [N*K, F, 2]
        return delta.view(N, K, F, 2)