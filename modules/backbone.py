from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse

from layers import GraphAttention
from layers import TwoLayerMLP
from layers import RobustnessAttention
from utils import compute_angles_lengths_2D
from utils import init_weights
from utils import wrap_angle
from utils import drop_edge_between_samples
from utils import transform_point_to_local_coordinate
from utils import transform_point_to_global_coordinate
from utils import transform_traj_to_global_coordinate
from utils import transform_traj_to_local_coordinate

class Backbone(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_attn_layers: int, 
                 num_modes: int,
                 num_heads: int,
                 dropout: float,
                 safety_params: dict) -> None:
        super(Backbone, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.duration = duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_attn_layers = num_attn_layers
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.dropout = dropout
        # 鲁棒性参数
        self.lon_time = safety_params.get('lon_time', 2.0)  # 纵向时间阈值
        self.lateral_threshold = safety_params.get('lateral_threshold', 1.5)  # 横向距离阈值
        self.ttc_vehicle = safety_params.get('ttc_vehicle', 1.5)  # 车-车TTC阈值
        self.ttc_vru = safety_params.get('ttc_vru', 1.6)  # 车-VRU TTC阈值
        self.distance_weight = safety_params.get('distance_weight', 1.0)  # 距离违规权重
        self.ttc_weight = safety_params.get('ttc_weight', 1.0)  # TTC违规权重
        self.lateral_weight = safety_params.get('lateral_weight', 1.0)  # 横向违规权重

        self.mode_tokens = nn.Embedding(num_modes, hidden_dim)     #[K,D]

        self.a_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        self.l2m_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.t2m_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.m2m_h_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_a_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_s_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.t2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.m2m_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=False, if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_propose = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps*2)

        self.proposal_to_anchor = TwoLayerMLP(input_dim=self.num_future_steps*2, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)
        self.t2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False)

        self.n2n_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_refine = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps*2)
        self.robustness_attn = RobustnessAttention(input_dim=2, hidden_dim=hidden_dim)
        self.traj_delta_mlp = TwoLayerMLP(
            input_dim=hidden_dim,  # 匹配robustness_attn的输出维度
            hidden_dim=hidden_dim,
            output_dim=self.num_future_steps*2  # 输出F步的调整量 (dx, dy)
        )
        
        self.apply(init_weights)

    def calculate_robustness(self, pred_traj, agent_data):
        """
        输入:
        pred_traj: [num_agents, H, K, F, 2] 多模态预测轨迹
        agent_data: 包含'category'字段的字典，[num_agents]表示参与者类型
        输出:
        violation_scores: [num_agents, K] 每个模态的违规程度评分
        """
        device = pred_traj.device
        num_agents, H, K, M, _ = pred_traj.shape

        # ================== 维度预处理 ==================
        # 将模态维度合并到代理维度 [num_agents*K, H, F, 2]
        traj_flat = pred_traj.permute(0, 2, 1, 3, 4).contiguous()
        traj_flat = traj_flat.view(-1, H, M, 2)  # [num_agents*K, H, F, 2]

        # ================== 速度计算 ==================
        dt = 0.1  # 时间间隔
        vel = (traj_flat[:, :, 1:] - traj_flat[:, :, :-1]) / dt
        vel = F.pad(vel, (0, 0, 0, 1, 0, 0), mode='replicate')  # [num_agents*K, H, F, 2]

        # ===从=============== 交互计算 ==================
        # 生成扩展维度 [num_agents*K, num_agents*K, H, F, 2]
        delta_pos = traj_flat.unsqueeze(1) - traj_flat.unsqueeze(0)
        distance_matrix = torch.norm(delta_pos, dim=-1)  # [num_agents*K, num_agents*K, H, F]

        # ================== 参与者类型处理 ==================
        # 原始代理类型 [num_agents] -> 扩展为 [num_agents*K]
        agent_types = agent_data['category']  # 假设 'category' 是一个张量
        agent_types = agent_types.repeat_interleave(K).to(device)
        is_vehicle = (agent_types == 1)
        is_vru = ~is_vehicle

        # ================== 交互掩码 ==================
        # 基本交互掩码 [num_agents*K, num_agents*K]
        vv_mask = is_vehicle.unsqueeze(1) & is_vehicle.unsqueeze(0)
        vn_mask = (is_vehicle.unsqueeze(1) & is_vru.unsqueeze(0)) | \
                (is_vru.unsqueeze(1) & is_vehicle.unsqueeze(0))
        nn_mask = is_vru.unsqueeze(1) & is_vru.unsqueeze(0)

        # 排除同一代理不同模态的交互
        agent_idx = torch.arange(num_agents).repeat_interleave(K, dim=0).to(device)
        same_agent = agent_idx.unsqueeze(1) == agent_idx.unsqueeze(0)
        vv_mask &= ~same_agent
        vn_mask &= ~same_agent
        nn_mask &= ~same_agent

        # ================== 动态阈值计算 ==================
        # 纵向速度分量 (假设y方向为横向)
        speed_y = torch.abs(vel[..., 1])  # [num_agents*K, H, F]

        # 车-车阈值 [num_agents*K, H, F]
        vv_threshold = speed_y * self.lon_time

        # 车-VRU阈值 [num_agents*K, H, F]
        vehicle_speed = speed_y * is_vehicle.float().view(-1,1,1)
        vru_threshold = vehicle_speed * self.lon_time

        # 组合阈值 [num_agents*K, num_agents*K, H, F]
        lon_threshold = torch.zeros_like(distance_matrix)
        lon_threshold[vv_mask] = vv_threshold.unsqueeze(1).expand_as(lon_threshold)[vv_mask]
        lon_threshold[vn_mask] = vru_threshold.unsqueeze(1).expand_as(lon_threshold)[vn_mask]

        # ================== TTC计算优化 ==================
        # 速度差 [num_agents*K, num_agents*K, H, F, 2]
        delta_vel = vel.unsqueeze(1) - vel.unsqueeze(0)
        
        # 投影计算（仅计算前F-1帧）
        dot_product = torch.sum(delta_pos[..., :-1, :] * delta_vel[..., :-1, :], dim=-1)
        vel_norm_sq = torch.sum(delta_vel[..., :-1, :]**2, dim=-1) + 1e-7
        ttc = torch.where(
            (vel_norm_sq > 1e-3) & (dot_product < 0),
            -dot_product / vel_norm_sq,
            torch.finfo(torch.float32).max
        )

        # ================== 违规计算 ==================
        lateral_distance = torch.abs(delta_pos[..., 1])  # [num_agents*K, num_agents*K, H, F]
        lateral_threshold = torch.zeros_like(lateral_distance)
        lateral_threshold[vn_mask] = self.lateral_threshold
        # 横向违规 [num_agents*K, num_agents*K, H, F]
        lateral_deficit = torch.zeros_like(lateral_distance)
        if vn_mask.any():
            raw_deficit = self.lateral_threshold - lateral_distance[vn_mask]
            lateral_deficit[vn_mask] = torch.relu(raw_deficit) / (self.lateral_threshold + 1e-9)

        # 距离违规 [num_agents*K, num_agents*K, H, F]
        safe_lon_threshold = lon_threshold + 1e-9
        distance_deficit = torch.relu(lon_threshold - distance_matrix) / safe_lon_threshold

        
        # TTC违规 [num_agents*K, num_agents*K, H, F]
        ttc_threshold = torch.zeros_like(ttc)  # 初始为全0
        vv_mask = vv_mask.unsqueeze(-1).unsqueeze(-1)
        vn_mask = vn_mask.unsqueeze(-1).unsqueeze(-1)
        vv_mask = vv_mask.expand_as(ttc_threshold) 
        vn_mask = vn_mask.expand_as(ttc_threshold) 
        ttc_threshold[vv_mask] = self.ttc_vehicle  # 广播维度
        ttc_threshold[vn_mask] = self.ttc_vru
        ttc_deficit = torch.relu(ttc_threshold - ttc)   
        safe_ttc_threshold = ttc_threshold + (ttc_threshold == 0) * 1e-9  # 保持分母非零
        ttc_deficit = ttc_deficit / safe_ttc_threshold # [num_agents*K, num_agents*K, H, F-1]
        ttc_deficit = F.pad(ttc_deficit, (0, 1), mode='constant', value=0) 

        # 综合违规 [num_agents*K, num_agents*K, H, F]
        combined_deficit = torch.max(
            distance_deficit,
            torch.max(ttc_deficit, lateral_deficit)
        )

        # 清理可能的数值异常（根据实际情况可选）
        combined_deficit = torch.nan_to_num(combined_deficit, nan=0.0, posinf=0.0, neginf=0.0) 

        # 过滤无效交互
        combined_deficit[nn_mask] = 0
        combined_deficit[same_agent] = 0

        # 聚合违规分数 [num_agents*K]
        violation_scores = combined_deficit.mean(dim=(1,2,3))  # 聚合所有交互和时间维度
        
        # 恢复原始维度 [num_agents, K]
        return violation_scores.view(num_agents, K)

    def forward(self, data: Batch, l_embs: torch.Tensor) -> torch.Tensor:
        # initialization
        a_velocity_length = data['agent']['velocity_length']                            #[(N1,...,Nb),H]
        a_velocity_theta = data['agent']['velocity_theta']                              #[(N1,...,Nb),H]
        a_length = data['agent']['length'].unsqueeze(-1).repeat_interleave(self.num_historical_steps,-1)      #[(N1,...,Nb),H]
        a_width = data['agent']['width'].unsqueeze(-1).repeat_interleave(self.num_historical_steps,-1)      #[(N1,...,Nb),H]
        a_type = data['agent']['type'].unsqueeze(-1).repeat_interleave(self.num_historical_steps,-1)      #[(N1,...,Nb),H]
        a_input = torch.stack([a_velocity_length, a_velocity_theta, a_length, a_width, a_type], dim=-1)
        a_embs = self.a_emb_layer(input=a_input)    #[(N1,...,Nb),H,D]
        
        num_all_agent = a_length.size(0)                # N1+...+Nb
        m_embs = self.mode_tokens.weight.unsqueeze(0).repeat_interleave(self.num_historical_steps,0)            #[H,K,D]
        m_embs = m_embs.unsqueeze(1).repeat_interleave(num_all_agent,1).reshape(-1, self.hidden_dim)            #[H*(N1,...,Nb)*K,D]

        m_batch = data['agent']['batch'].unsqueeze(1).repeat_interleave(self.num_modes,1)                       # [(N1,...,Nb),K]
        m_position = data['agent']['position'][:,:self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes,2)  #[(N1,...,Nb),H,K,2]
        m_heading = data['agent']['heading'][:,:self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes,2)    #[(N1,...,Nb),H,K]
        m_valid_mask = data['agent']['visible_mask'][:,:self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes,2)  #[(N1,...,Nb),H,K]
        m_velocity = data['agent']['velocity'][:,:self.num_historical_steps].unsqueeze(2).repeat_interleave(self.num_modes,2)  #[(N1,...,Nb),H,,K]

        #ALL EDGE
        #t2m edge
        t2m_position_t = data['agent']['position'][:,:self.num_historical_steps].reshape(-1,2)      #[(N1,...,Nb)*H,2]
        t2m_position_m = m_position.reshape(-1,2)                                                   #[(N1,...,Nb)*H*K,2]
        t2m_heading_t = data['agent']['heading'].reshape(-1)                                        #[(N1,...,Nb)]
        t2m_heading_m = m_heading.reshape(-1)                                                       #[(N1,...,Nb)*H*K]
        t2m_valid_mask_t = data['agent']['visible_mask'][:,:self.num_historical_steps]              #[(N1,...,Nb),H]
        t2m_valid_mask_m = m_valid_mask.reshape(num_all_agent,-1)                                   #[(N1,...,Nb),H*K]
        t2m_valid_mask = t2m_valid_mask_t.unsqueeze(2) & t2m_valid_mask_m.unsqueeze(1)              #[(N1,...,Nb),H,H*K]
        t2m_edge_index = dense_to_sparse(t2m_valid_mask)[0]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1]/self.num_modes) >= t2m_edge_index[0]]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1]/self.num_modes) - t2m_edge_index[0] <= self.duration]
        t2m_edge_vector = transform_point_to_local_coordinate(t2m_position_t[t2m_edge_index[0]], t2m_position_m[t2m_edge_index[1]], t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_length, t2m_edge_attr_theta = compute_angles_lengths_2D(t2m_edge_vector)
        t2m_edge_attr_heading = wrap_angle(t2m_heading_t[t2m_edge_index[0]] - t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_interval = t2m_edge_index[0] - torch.floor(t2m_edge_index[1]/self.num_modes)
        # 计算相对速度（需要从速度数据中提取相对速度）
        t2m_velocity_t = data['agent']['velocity'][:, :self.num_historical_steps].reshape(-1, 2)  # agent 的速度
        t2m_velocity_m = m_velocity.reshape(-1, 2)  # 模式的速度
        t2m_velocity_rel = t2m_velocity_m[t2m_edge_index[1]] - t2m_velocity_t[t2m_edge_index[0]]  # 相对速度
        # 计算相对位置与相对速度的点积
        dot_product = torch.sum(t2m_edge_vector * t2m_velocity_rel, dim=-1)
        # 计算相对速度的平方
        vel_norm_sq = torch.sum(t2m_velocity_rel ** 2, dim=-1) + 1e-7
        # 计算 TTC：只有当相对速度足够大并且两者相对位置是负值（表示接近）时，才计算 TTC
        t2m_ttc = torch.where(
            (vel_norm_sq > 1e-3) & (dot_product < 0),
            -dot_product / vel_norm_sq,
            torch.finfo(torch.float32).max  # 如果不满足条件，则使用最大值
        )
        # 对TTC进行对数缩放
        t2m_ttc = torch.clamp(t2m_ttc, min=1e-3, max=10)  # 限制有效范围
        # 将所有边缘特征合并
        t2m_edge_attr_input = torch.stack([
            t2m_edge_attr_length,
            t2m_edge_attr_theta,
            t2m_edge_attr_heading,
            t2m_edge_attr_interval,
            t2m_ttc  # 添加 TTC 作为新的边缘特征 [num_edges]
        ], dim=-1)
        t2m_edge_attr_embs = self.t2m_emb_layer(input=t2m_edge_attr_input)

        #l2m edge
        l2m_position_l = data['lane']['position']                       #[(M1,...,Mb),2]
        l2m_position_m = m_position.reshape(-1,2)                       #[(N1,...,Nb)*H*K,2]
        l2m_heading_l = data['lane']['heading']                         #[(M1,...,Mb)]
        l2m_heading_m = m_heading.reshape(-1)                           #[(N1,...,Nb)]
        l2m_batch_l = data['lane']['batch']                             #[(M1,...,Mb)]
        l2m_batch_m = m_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps,1).reshape(-1)       #[(N1,...,Nb)*H*K]
        l2m_valid_mask_l = data['lane']['visible_mask']                                                     #[(M1,...,Mb)]
        l2m_valid_mask_m = m_valid_mask.reshape(-1)                                                         #[(N1,...,Nb)*H*K]
        l2m_valid_mask = l2m_valid_mask_l.unsqueeze(1)&l2m_valid_mask_m.unsqueeze(0)                        #[(M1,...,Mb),(N1,...,Nb)*H*K]
        l2m_valid_mask = drop_edge_between_samples(l2m_valid_mask, batch=(l2m_batch_l, l2m_batch_m))
        l2m_edge_index = dense_to_sparse(l2m_valid_mask)[0]
        l2m_edge_index = l2m_edge_index[:, torch.norm(l2m_position_l[l2m_edge_index[0]] - l2m_position_m[l2m_edge_index[1]], p=2, dim=-1) < self.l2a_radius]
        l2m_edge_vector = transform_point_to_local_coordinate(l2m_position_l[l2m_edge_index[0]], l2m_position_m[l2m_edge_index[1]], l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_length, l2m_edge_attr_theta = compute_angles_lengths_2D(l2m_edge_vector)
        l2m_edge_attr_heading = wrap_angle(l2m_heading_l[l2m_edge_index[0]] - l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_input = torch.stack([l2m_edge_attr_length, l2m_edge_attr_theta, l2m_edge_attr_heading], dim=-1)
        l2m_edge_attr_embs = self.l2m_emb_layer(input=l2m_edge_attr_input)

        #mode edge
        #m2m_a_edge
        m2m_a_position = m_position.permute(1,2,0,3).reshape(-1, 2)    #[H*K*(N1,...,Nb),2]
        m2m_a_heading = m_heading.permute(1,2,0).reshape(-1)           #[H*K*(N1,...,Nb)]
        m2m_a_batch = data['agent']['batch']                           #[(N1,...,Nb)]
        m2m_a_valid_mask = m_valid_mask.permute(1,2,0).reshape(self.num_historical_steps * self.num_modes, -1)  #[H*K,(N1,...,Nb)]
        m2m_a_valid_mask = m2m_a_valid_mask.unsqueeze(2) & m2m_a_valid_mask.unsqueeze(1)                        #[H*K,(N1,...,Nb),(N1,...,Nb)]
        m2m_a_valid_mask = drop_edge_between_samples(m2m_a_valid_mask, m2m_a_batch)
        m2m_a_edge_index = dense_to_sparse(m2m_a_valid_mask)[0]
        m2m_a_edge_index = m2m_a_edge_index[:, m2m_a_edge_index[1] != m2m_a_edge_index[0]]
        m2m_a_edge_index = m2m_a_edge_index[:, torch.norm(m2m_a_position[m2m_a_edge_index[1]] - m2m_a_position[m2m_a_edge_index[0]],p=2,dim=-1) < self.a2a_radius]
        m2m_a_edge_vector = transform_point_to_local_coordinate(m2m_a_position[m2m_a_edge_index[0]], m2m_a_position[m2m_a_edge_index[1]], m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_edge_attr_length, m2m_a_edge_attr_theta = compute_angles_lengths_2D(m2m_a_edge_vector)
        m2m_a_edge_attr_heading = wrap_angle(m2m_a_heading[m2m_a_edge_index[0]] - m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_velocity = m_velocity.reshape(-1, 2)
        m2m_a_vel_rel = m2m_a_velocity[m2m_a_edge_index[1]] - m2m_a_velocity[m2m_a_edge_index[0]]
        m2m_a_dot = torch.sum(m2m_a_edge_vector * m2m_a_vel_rel, dim=-1)
        m2m_a_vel_norm_sq = torch.sum(m2m_a_vel_rel ** 2, dim=-1) + 1e-7
        m2m_a_ttc = torch.where(
            (m2m_a_vel_norm_sq > 1e-3) & (m2m_a_dot < 0),
            -m2m_a_dot / m2m_a_vel_norm_sq,
            torch.finfo(torch.float32).max
        )
        m2m_a_ttc = torch.clamp(m2m_a_ttc, min=1e-3, max=10)  # 限制有效范围
        m2m_a_edge_attr_input = torch.stack([
            m2m_a_edge_attr_length,
            m2m_a_edge_attr_theta,
            m2m_a_edge_attr_heading,
            m2m_a_ttc  # [num_edges]
        ], dim=-1)
        m2m_a_edge_attr_embs = self.m2m_a_emb_layer(input=m2m_a_edge_attr_input)

        #m2m_h                        
        m2m_h_position = m_position.permute(2,0,1,3).reshape(-1, 2)    #[K*(N1,...,Nb)*H,2]
        m2m_h_heading = m_heading.permute(2,0,1).reshape(-1)           #[K*(N1,...,Nb)*H]
        m2m_h_valid_mask = m_valid_mask.permute(2,0,1).reshape(-1, self.num_historical_steps)   #[K*(N1,...,Nb),H]
        m2m_h_valid_mask = m2m_h_valid_mask.unsqueeze(2) & m2m_h_valid_mask.unsqueeze(1)        #[K*(N1,...,Nb),H,H]     
        m2m_h_edge_index = dense_to_sparse(m2m_h_valid_mask)[0]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] > m2m_h_edge_index[0]]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] - m2m_h_edge_index[0] <= self.duration]
        m2m_h_edge_vector = transform_point_to_local_coordinate(m2m_h_position[m2m_h_edge_index[0]], m2m_h_position[m2m_h_edge_index[1]], m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_length, m2m_h_edge_attr_theta = compute_angles_lengths_2D(m2m_h_edge_vector)
        m2m_h_edge_attr_heading = wrap_angle(m2m_h_heading[m2m_h_edge_index[0]] - m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_interval = m2m_h_edge_index[0] - m2m_h_edge_index[1]
        # 计算相对速度
        m2m_h_velocity = m_velocity.reshape(-1, 2)
        m2m_h_velocity_rel =m2m_h_velocity[m2m_h_edge_index[1]] - m2m_h_velocity[m2m_h_edge_index[0]]  # 相对速度
        # 计算点积和相对速度的平方
        dot_product = torch.sum(m2m_h_edge_vector * m2m_h_velocity_rel, dim=-1)
        vel_norm_sq = torch.sum(m2m_h_velocity_rel ** 2, dim=-1) + 1e-7
        # 计算 TTC：只有当相对速度足够大并且两者是接近的（dot_product < 0），才计算 TTC
        m2m_h_ttc = torch.where(
            (vel_norm_sq > 1e-3) & (dot_product < 0),
            -dot_product / vel_norm_sq,
            torch.finfo(torch.float32).max  # 如果不满足条件，则设为最大值
        )
        m2m_h_ttc = torch.clamp(m2m_h_ttc, min=1e-3, max=10)  # 限制有效范围

        m2m_h_edge_attr_input = torch.stack([m2m_h_edge_attr_length, m2m_h_edge_attr_theta, m2m_h_edge_attr_heading, m2m_h_edge_attr_interval,m2m_h_ttc], dim=-1)
        m2m_h_edge_attr_embs = self.m2m_h_emb_layer(input=m2m_h_edge_attr_input)

        #m2m_s edge
        m2m_s_valid_mask = m_valid_mask.transpose(0,1).reshape(-1, self.num_modes)              #[H*(N1,...,Nb),K]
        m2m_s_valid_mask = m2m_s_valid_mask.unsqueeze(2) & m2m_s_valid_mask.unsqueeze(1)        #[H*(N1,...,Nb),K,K]
        m2m_s_edge_index = dense_to_sparse(m2m_s_valid_mask)[0]
        m2m_s_edge_index = m2m_s_edge_index[:, m2m_s_edge_index[0] != m2m_s_edge_index[1]]

        #ALL ATTENTION
        #t2m attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)  #[(N1,...,Nb)*H,D]
        m_embs_t = self.t2m_attn_layer(x = [t_embs, m_embs], edge_index = t2m_edge_index, edge_attr = t2m_edge_attr_embs)         #[(N1,...,Nb)*H*K,D]

        #l2m attention
        m_embs_l = self.l2m_attn_layer(x = [l_embs, m_embs], edge_index = l2m_edge_index, edge_attr = l2m_edge_attr_embs)         #[(N1,...,Nb)*H*K,D]
        
        m_embs = m_embs_t + m_embs_l
        m_embs = m_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1,self.hidden_dim)       #[H*(N1,...,Nb)*K,D]
        #moda attention  
        for i in range(self.num_attn_layers):
            #m2m_a
            m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(1,2).reshape(-1, self.hidden_dim)  #[H*K*(N1,...,Nb),D]
            m_embs = self.m2m_a_attn_layers[i](x = m_embs, edge_index = m2m_a_edge_index, edge_attr = m2m_a_edge_attr_embs)
            #m2m_h
            m_embs = m_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(1,2,0,3).reshape(-1, self.hidden_dim)  #[K*(N1,...,Nb)*H,D]
            m_embs = self.m2m_h_attn_layers[i](x = m_embs, edge_index = m2m_h_edge_index, edge_attr = m2m_h_edge_attr_embs)
            #m2m_s
            m_embs = m_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim).transpose(0,2).reshape(-1, self.hidden_dim)  #[H*(N1,...,Nb)*K,D]
            m_embs = self.m2m_s_attn_layers[i](x = m_embs, edge_index = m2m_s_edge_index)
        m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1, self.hidden_dim)      #[(N1,...,Nb)*H*K,D]

        #generate traj
        traj_propose = self.traj_propose(m_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 2)         #[(N1,...,Nb),H,K,F,2]
        traj_propose = transform_traj_to_global_coordinate(traj_propose, m_position, m_heading)        #[(N1,...,Nb),H,K,F,2]

        #generate anchor
        proposal = traj_propose.detach()        #[(N1,...,Nb),H,K,F,2]
        
        n_batch = m_batch                                                                                                   #[(N1,...,Nb),K]
        n_position = proposal[:,:,:, self.num_future_steps // 2,:]                                                     #[(N1,...,Nb),H,K,2]
        # 提取相邻位置差值（只考虑历史轨迹的差值，未来轨迹不考虑）
        position_diff = proposal[:, :, :, 1:, :] - proposal[:, :, :, :-1, :]  # 计算相邻位置的差值 [(N1,...,Nb), H, K, F-1, 2]
        # 假设时间间隔为 0.1 秒
        time_interval = 0.1
        # 计算速度：位置差值除以时间间隔
        velocity = position_diff / time_interval  # [(N1,...,Nb), H, K, F-1, 2]
        # 处理最后一个时间步的速度（确保最后一个时间步有速度，非零）
        # 通过计算最后两个位置之间的差值来推算最后一个时间步的速度
        last_position_diff = proposal[:, :, :, -1, :] - proposal[:, :, :, -2, :]  # 计算最后两个时间步的差值
        last_velocity = last_position_diff / time_interval  # 用相同的时间间隔计算速度
        # 将最后一个时间步的速度添加到速度张量中，并将维度对齐为 [N1, ..., Nb, H, K, 2]
        velocity = torch.cat([velocity, last_velocity.unsqueeze(-2)], dim=-2)  # [(N1,...,Nb), H, K, F, 2]
        n_velocity = velocity[:, :, :, :-1, :]  # [(N1,...,Nb), H, K, 2]
        _, n_heading = compute_angles_lengths_2D(proposal[:,:,:, self.num_future_steps // 2,:] - proposal[:,:,:, self.num_future_steps // 2 - 1,:])  #[(N1,...,Nb),H,K]
        n_valid_mask = m_valid_mask                                                                                         #[(N1,...,Nb),H,K]
        
        proposal = transform_traj_to_local_coordinate(proposal, n_position, n_heading)                                      #[(N1,...,Nb),H,K,F,2]
        anchor = self.proposal_to_anchor(proposal.reshape(-1, self.num_future_steps*2))                                     #[(N1,...,Nb)*H*K,D]
        n_embs = anchor                                                                                                     #[(N1,...,Nb)*H*K,D]

        #t2n edge
        t2n_position_t = data['agent']['position'][:,:self.num_historical_steps].reshape(-1,2)      #[(N1,...,Nb)*H,2]
        t2n_position_n = n_position.reshape(-1,2)                                                   #[(N1,...,Nb)*H*K,2]
        t2n_heading_t = data['agent']['heading'].reshape(-1)                                        #[(N1,...,Nb)]
        t2n_heading_n = n_heading.reshape(-1)                                                       #[(N1,...,Nb)*H*K]
        t2n_valid_mask_t = data['agent']['visible_mask'][:,:self.num_historical_steps]              #[(N1,...,Nb),H]
        t2n_valid_mask_n = n_valid_mask.reshape(num_all_agent,-1)                                   #[(N1,...,Nb),H*K]
        t2n_valid_mask = t2n_valid_mask_t.unsqueeze(2) & t2n_valid_mask_n.unsqueeze(1)              #[(N1,...,Nb),H,H*K]
        t2n_edge_index = dense_to_sparse(t2n_valid_mask)[0]
        t2n_edge_index = t2n_edge_index[:, torch.floor(t2n_edge_index[1]/self.num_modes) >= t2n_edge_index[0]]
        t2n_edge_index = t2n_edge_index[:, torch.floor(t2n_edge_index[1]/self.num_modes) - t2n_edge_index[0] <= self.duration]
        t2n_edge_vector = transform_point_to_local_coordinate(t2n_position_t[t2n_edge_index[0]], t2n_position_n[t2n_edge_index[1]], t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_length, t2n_edge_attr_theta = compute_angles_lengths_2D(t2n_edge_vector)
        t2n_edge_attr_heading = wrap_angle(t2n_heading_t[t2n_edge_index[0]] - t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_interval = t2n_edge_index[0] - torch.floor(t2n_edge_index[1]/self.num_modes) - self.num_future_steps//2
        # 计算相对速度（需要从速度数据中提取相对速度）
        t2n_velocity_t = data['agent']['velocity'][:, :self.num_historical_steps].reshape(-1, 2)  # agent 的速度
        t2n_velocity_n = n_velocity.reshape(-1, 2)  # 模式的速度
        t2n_velocity_rel = t2n_velocity_n[t2n_edge_index[1]] - t2n_velocity_t[t2n_edge_index[0]]  # 相对速度
        # 计算相对位置与相对速度的点积
        dot_product = torch.sum(t2n_edge_vector * t2n_velocity_rel, dim=-1)
        # 计算相对速度的平方
        vel_norm_sq = torch.sum(t2n_velocity_rel ** 2, dim=-1) + 1e-7
        # 计算 TTC：只有当相对速度足够大并且两者相对位置是负值（表示接近）时，才计算 TTC
        t2n_ttc = torch.where(
            (vel_norm_sq > 1e-3) & (dot_product < 0),
            -dot_product / vel_norm_sq,
            torch.finfo(torch.float32).max  # 如果不满足条件，则使用最大值
        )
        # 对TTC进行对数缩放
        t2n_ttc = torch.clamp(t2n_ttc, min=1e-3, max=10)  # 限制有效范围

        t2n_edge_attr_input = torch.stack([t2n_edge_attr_length, t2n_edge_attr_theta, t2n_edge_attr_heading, t2n_edge_attr_interval,t2n_ttc], dim=-1)
        t2n_edge_attr_embs = self.t2m_emb_layer(input=t2n_edge_attr_input)

        #l2n edge
        l2n_position_l = data['lane']['position']                       #[(M1,...,Mb),2]
        l2n_position_n = n_position.reshape(-1,2)                       #[(N1,...,Nb)*H*K,2]
        l2n_heading_l = data['lane']['heading']                         #[(M1,...,Mb)]
        l2n_heading_n = n_heading.reshape(-1)                           #[(N1,...,Nb)*H*K]
        l2n_batch_l = data['lane']['batch']                             #[(M1,...,Mb)]
        l2n_batch_n = n_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps,1).reshape(-1)       #[(N1,...,Nb)*H*K]
        l2n_valid_mask_l = data['lane']['visible_mask']                                                     #[(M1,...,Mb)]
        l2n_valid_mask_n = n_valid_mask.reshape(-1)                                                         #[(N1,...,Nb)*H*K]
        l2n_valid_mask = l2n_valid_mask_l.unsqueeze(1) & l2n_valid_mask_n.unsqueeze(0)                      #[(M1,...,Mb),(N1,...,Nb)*H*K]
        l2n_valid_mask = drop_edge_between_samples(l2n_valid_mask, batch=(l2n_batch_l, l2n_batch_n))
        l2n_edge_index = dense_to_sparse(l2n_valid_mask)[0]
        l2n_edge_index = l2n_edge_index[:, torch.norm(l2n_position_l[l2n_edge_index[0]] - l2n_position_n[l2n_edge_index[1]], p=2, dim=-1) < self.l2a_radius]
        l2n_edge_vector = transform_point_to_local_coordinate(l2n_position_l[l2n_edge_index[0]], l2n_position_n[l2n_edge_index[1]], l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_length, l2n_edge_attr_theta = compute_angles_lengths_2D(l2n_edge_vector)
        l2n_edge_attr_heading = wrap_angle(l2n_heading_l[l2n_edge_index[0]] - l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_input = torch.stack([l2n_edge_attr_length, l2n_edge_attr_theta, l2n_edge_attr_heading], dim=-1)
        l2n_edge_attr_embs = self.l2m_emb_layer(input = l2n_edge_attr_input)

        #mode edge
        #n2n_a_edge
        n2n_a_position = n_position.permute(1,2,0,3).reshape(-1, 2)    #[H*K*(N1,...,Nb),2]
        n2n_a_heading = n_heading.permute(1,2,0).reshape(-1)           #[H*K*(N1,...,Nb)]
        n2n_a_batch = data['agent']['batch']                           #[(N1,...,Nb)]
        n2n_a_valid_mask = n_valid_mask.permute(1,2,0).reshape(self.num_historical_steps * self.num_modes, -1)   #[H*K,(N1,...,Nb)]
        n2n_a_valid_mask = n2n_a_valid_mask.unsqueeze(2) & n2n_a_valid_mask.unsqueeze(1)        #[H*K,(N1,...,Nb),(N1,...,Nb)]
        n2n_a_valid_mask = drop_edge_between_samples(n2n_a_valid_mask, n2n_a_batch)
        n2n_a_edge_index = dense_to_sparse(n2n_a_valid_mask)[0]
        n2n_a_edge_index = n2n_a_edge_index[:, n2n_a_edge_index[1] != n2n_a_edge_index[0]]
        n2n_a_edge_index = n2n_a_edge_index[:, torch.norm(n2n_a_position[n2n_a_edge_index[1]] - n2n_a_position[n2n_a_edge_index[0]],p=2,dim=-1) < self.a2a_radius]
        n2n_a_edge_vector = transform_point_to_local_coordinate(n2n_a_position[n2n_a_edge_index[0]], n2n_a_position[n2n_a_edge_index[1]], n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_edge_attr_length, n2n_a_edge_attr_theta = compute_angles_lengths_2D(n2n_a_edge_vector)
        n2n_a_edge_attr_heading = wrap_angle(n2n_a_heading[n2n_a_edge_index[0]] - n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_velocity = n_velocity.reshape(-1, 2)
        n2n_a_vel_rel = n2n_a_velocity[n2n_a_edge_index[1]] - n2n_a_velocity[n2n_a_edge_index[0]]
        n2n_a_dot = torch.sum(n2n_a_edge_vector * n2n_a_vel_rel, dim=-1)
        n2n_a_vel_norm_sq = torch.sum(n2n_a_vel_rel ** 2, dim=-1) + 1e-7
        n2n_a_ttc = torch.where(
            (n2n_a_vel_norm_sq > 1e-3) & (n2n_a_dot < 0),
            -n2n_a_dot / n2n_a_vel_norm_sq,
            torch.finfo(torch.float32).max
        )
        n2n_a_ttc = torch.clamp(n2n_a_ttc, min=1e-3, max=10)  # 限制有效范围

        n2n_a_edge_attr_input = torch.stack([n2n_a_edge_attr_length, n2n_a_edge_attr_theta, n2n_a_edge_attr_heading, n2n_a_ttc], dim=-1)
        n2n_a_edge_attr_embs = self.m2m_a_emb_layer(input=n2n_a_edge_attr_input)

        #n2n_h edge                        
        n2n_h_position = n_position.permute(2,0,1,3).reshape(-1, 2)    #[K*(N1,...,Nb)*H,2]
        n2n_h_heading = n_heading.permute(2,0,1).reshape(-1)           #[K*(N1,...,Nb)*H]
        n2n_h_valid_mask = n_valid_mask.permute(2,0,1).reshape(-1, self.num_historical_steps)   #[K*(N1,...,Nb),H]
        n2n_h_valid_mask = n2n_h_valid_mask.unsqueeze(2) & n2n_h_valid_mask.unsqueeze(1)        #[K*(N1,...,Nb),H,H]        
        n2n_h_edge_index = dense_to_sparse(n2n_h_valid_mask)[0]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] > n2n_h_edge_index[0]]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] - n2n_h_edge_index[0] <= self.duration]   
        n2n_h_edge_vector = transform_point_to_local_coordinate(n2n_h_position[n2n_h_edge_index[0]], n2n_h_position[n2n_h_edge_index[1]], n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_length, n2n_h_edge_attr_theta = compute_angles_lengths_2D(n2n_h_edge_vector)
        n2n_h_edge_attr_heading = wrap_angle(n2n_h_heading[n2n_h_edge_index[0]] - n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_interval = n2n_h_edge_index[0] - n2n_h_edge_index[1]
        # 计算相对速度
        n2n_h_velocity = n_velocity.reshape(-1, 2)
        n2n_h_velocity_rel = n2n_h_velocity[n2n_h_edge_index[1]] - n2n_h_velocity[n2n_h_edge_index[0]]  # 相对速度
        # 计算点积和相对速度的平方
        dot_product = torch.sum(n2n_h_edge_vector * n2n_h_velocity_rel, dim=-1)
        vel_norm_sq = torch.sum(n2n_h_velocity_rel ** 2, dim=-1) + 1e-7
        # 计算 TTC：只有当相对速度足够大并且两者是接近的（dot_product < 0），才计算 TTC
        n2n_h_ttc = torch.where(
            (vel_norm_sq > 1e-3) & (dot_product < 0),
            -dot_product / vel_norm_sq,
            torch.finfo(torch.float32).max  # 如果不满足条件，则设为最大值
        )
        n2n_h_ttc = torch.clamp(n2n_h_ttc, min=1e-3, max=10)  # 限制有效范围

        n2n_h_edge_attr_input = torch.stack([n2n_h_edge_attr_length, n2n_h_edge_attr_theta, n2n_h_edge_attr_heading, n2n_h_edge_attr_interval,n2n_h_ttc], dim=-1)
        n2n_h_edge_attr_embs = self.m2m_h_emb_layer(input=n2n_h_edge_attr_input)

        #n2n_s edge
        n2n_s_position = n_position.transpose(0,1).reshape(-1,2)                                #[H*(N1,...,Nb)*K,2]
        n2n_s_heading = n_heading.transpose(0,1).reshape(-1)                                    #[H*(N1,...,Nb)*K]
        n2n_s_valid_mask = n_valid_mask.transpose(0,1).reshape(-1, self.num_modes)              #[H*(N1,...,Nb),K]
        n2n_s_valid_mask = n2n_s_valid_mask.unsqueeze(2) & n2n_s_valid_mask.unsqueeze(1)        #[H*(N1,...,Nb),K,K]
        n2n_s_edge_index = dense_to_sparse(n2n_s_valid_mask)[0]
        n2n_s_edge_index = n2n_s_edge_index[:, n2n_s_edge_index[0] != n2n_s_edge_index[1]]
        n2n_s_edge_vector = transform_point_to_local_coordinate(n2n_s_position[n2n_s_edge_index[0]], n2n_s_position[n2n_s_edge_index[1]], n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_length, n2n_s_edge_attr_theta = compute_angles_lengths_2D(n2n_s_edge_vector)
        n2n_s_edge_attr_heading = wrap_angle(n2n_s_heading[n2n_s_edge_index[0]] - n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_input = torch.stack([n2n_s_edge_attr_length, n2n_s_edge_attr_theta, n2n_s_edge_attr_heading], dim=-1)
        n2n_s_edge_attr_embs = self.m2m_s_emb_layer(input=n2n_s_edge_attr_input)

        #t2n attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)  #[(N1,...,Nb)*H,D]
        n_embs_t = self.t2n_attn_layer(x = [t_embs, n_embs], edge_index = t2n_edge_index, edge_attr = t2n_edge_attr_embs)         #[(N1,...,Nb)*H*K,D]

        #l2m attention
        n_embs_l = self.l2n_attn_layer(x = [l_embs, n_embs], edge_index = l2n_edge_index, edge_attr = l2n_edge_attr_embs)         #[(N1,...,Nb)*H*K,D]

        n_embs = n_embs_t + n_embs_l
        n_embs = n_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1,self.hidden_dim)       #[H*(N1,...,Nb)*K,D]
        #moda attention  
        for i in range(self.num_attn_layers):
            #m2m_a
            n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(1,2).reshape(-1, self.hidden_dim)  #[H*K*(N1,...,Nb),D]
            n_embs = self.n2n_a_attn_layers[i](x = n_embs, edge_index = n2n_a_edge_index, edge_attr = n2n_a_edge_attr_embs)
            #m2m_h
            n_embs = n_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(1,2,0,3).reshape(-1, self.hidden_dim)  #[K*(N1,...,Nb)*H,D]
            n_embs = self.n2n_h_attn_layers[i](x = n_embs, edge_index = n2n_h_edge_index, edge_attr = n2n_h_edge_attr_embs)
            #m2m_s
            n_embs = n_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim).transpose(0,2).reshape(-1, self.hidden_dim)  #[H*(N1,...,Nb)*K,D]
            n_embs = self.n2n_s_attn_layers[i](x = n_embs, edge_index = n2n_s_edge_index, edge_attr = n2n_s_edge_attr_embs)
        n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0,1).reshape(-1, self.hidden_dim)      #[(N1,...,Nb)*H*K,D

        #generate refinement
        traj_refine = self.traj_refine(n_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 2)                  #[(N1,...,Nb),H,K,F,2]         
        traj_output = transform_traj_to_global_coordinate(proposal + traj_refine, n_position, n_heading)                                               #[(N1,...,Nb),H,K,F,2] 
        agent_types = data['agent']
        # 计算违规分数
        violation_scores = self.calculate_robustness(traj_output, agent_types)  # [num_agents, K]
                
        # 注意力机制生成调整量
        # 取时间维度均值 [N, H, K, F, 2] -> [N, K, F, 2]
        proposal_mean = proposal.mean(dim=1)
        attn_features = self.robustness_attn(proposal_mean, violation_scores)  # [N, K, F, 2]
    
        # 生成轨迹修正量
        traj_delta = attn_features.view(num_all_agent * self.num_modes, -1)  # [N*K, F*2]
        # 将 traj_delta 的形状调整为 [N, K, F, 2]
        traj_delta = traj_delta.view(num_all_agent, self.num_modes, -1, 2)  # [N, K, F, 2]

        # 扩展 traj_delta 的形状为 [N, H, K, F, 2]
        traj_delta = traj_delta.unsqueeze(1).expand(-1, self.num_historical_steps, -1, -1, -1)  # [N, H, K, F, 2]
    
        # 综合输出（全局坐标系）
        final_traj = traj_output + traj_delta  # [N, H, K, F, 2]
        return traj_propose, traj_output , final_traj, violation_scores       #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2]
