import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
import math
import pandas as pd

from losses import JointLoss
from metrics import minJointADE
from metrics import minJointFDE
from modules import Backbone
from modules import MapEncoder

from utils import generate_target
from utils import generate_predict_mask
from utils import compute_angles_lengths_2D

#torch.set_float32_matmul_precision('high')

class RuleNet(pl.LightningModule):
    

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_visible_steps: int,
                 num_modes: int,
                 num_attn_layers: int,
                 num_hops: int,
                 num_heads: int,
                 dropout: float,
                 lr: float,
                 weight_decay: float,
                 warmup_epochs: int,
                 T_max: int,
                 distance_weight: float = 1.0,
                 ttc_weight: float = 1.0,
                 lateral_weight: float = 1.0,
                 ttc_vehicle: float = 1.5,        
                 ttc_vru: float = 1.6,
                 lon_time: float = 2.0,
                 lateral_threshold: float = 1.5,
                 **kwargs) -> None:
        super(RuleNet, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.duration = duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_visible_steps = num_visible_steps
        self.num_modes = num_modes
        self.num_attn_layers = num_attn_layers
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.ttc_vehicle = ttc_vehicle         # TTC时间阈值（秒）
        self.ttc_vru = ttc_vru         # TTC时间阈值（秒）
        self.distance_weight = distance_weight       # 纵向距离违规权重
        self.ttc_weight = ttc_weight            # TTC违规权重       
        self.lateral_threshold = lateral_threshold     # 横向安全距离阈值（米）
        self.lateral_weight = lateral_weight        # 横向违规权重
        self.lon_time = lon_time

        self.Backbone = Backbone(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            duration=duration,
            a2a_radius=a2a_radius,
            l2a_radius=l2a_radius,
            num_attn_layers=num_attn_layers,
            num_modes=num_modes,
            num_heads=num_heads,
            dropout=dropout,
            safety_params={
		    'lon_time': 2.0,
		    'lateral_threshold': 1.5,
		    'ttc_vehicle': 1.5,
		    'ttc_vru': 1.6,
		    'distance_weight': 1.0,
		    'ttc_weight': 1.0,
		    'lateral_weight': 1.0
		}
            
        )
        self.MapEncoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )

        self.joint_loss = JointLoss()

        self.min_joint_ade = minJointADE()
        self.min_joint_fde = minJointFDE()

        self._columns = ['case_id', 'track_id', 'frame_id', 'timestamp_ms', 'interesting_agent',
                         'x1', 'y1', 'psi_rad1',
                         'x2', 'y2', 'psi_rad2',
                         'x3', 'y3', 'psi_rad3',
                         'x4', 'y4', 'psi_rad4',
                         'x5', 'y5', 'psi_rad5',
                         'x6', 'y6', 'psi_rad6']
        self.test_output = dict()

    def forward(self, 
                data: Batch):
        lane_embs = self.MapEncoder(data=data)
        pred = self.Backbone(data=data, l_embs=lane_embs)
        return pred
    
    def calculate_safety_violation(self, pred_traj, agent_data):
        device = pred_traj.device

        # ================== 速度计算 ==================
        dt = 0.1  # 时间间隔（仅用于速度计算）
        vel = (pred_traj[:, 1:] - pred_traj[:, :-1]) / dt
        vel = F.pad(vel, (0, 0, 0, 1), value=0)  # [na, F, 2]

        # ================== 相对位置计算 ==================
        delta_pos = pred_traj.unsqueeze(1) - pred_traj.unsqueeze(0)  # [na, na, F, 2]
        distance_matrix = torch.norm(delta_pos, dim=-1)  # [na, na, F]

        # ================== 参与者类型掩码 ==================
        is_vehicle = (agent_data == 1).to(device)
        is_vru = ~is_vehicle
        
        # 三维交互掩码 [na, na, F]
        vv_mask = is_vehicle.unsqueeze(1) & is_vehicle.unsqueeze(0)
        vn_mask = (is_vehicle.unsqueeze(1) & is_vru.unsqueeze(0)) | \
                (is_vru.unsqueeze(1) & is_vehicle.unsqueeze(0))
        nn_mask = is_vru.unsqueeze(1) & is_vru.unsqueeze(0) #[na,na]

        # ================== 动态阈值计算 ==================
        #speed_norm = torch.norm(vel, dim=2)  # [na, F]
        speed_y = torch.abs(vel[..., 1])

        # 车-车纵向阈值
        vv_threshold = speed_y.unsqueeze(1) * self.lon_time # [na,1, F]
        num_agents = vv_threshold.shape[0]  # 获取第一维大小na
        vv_threshold = vv_threshold.expand(num_agents, num_agents, -1)  # 扩展为 [num_agents, num_agents, F]


        # 车-VRU纵向阈值
        vehicle_speed = speed_y * is_vehicle.float().unsqueeze(1)
        vru_threshold = vehicle_speed.unsqueeze(1) * self.lon_time
        num_agents = vru_threshold.shape[0] 
        vru_threshold = vru_threshold.expand(num_agents, num_agents, -1)  # 扩展为 [num_agents, num_agents, F]

        # 组合阈值矩阵
        lon_threshold = torch.zeros_like(vv_threshold)
        lon_threshold[vv_mask] = vv_threshold[vv_mask]
        lon_threshold[vn_mask] = vru_threshold[vn_mask]


        # ================== TTC计算 ==================
        delta_vel = vel.unsqueeze(1) - vel.unsqueeze(0)  # [na, na, F, 2]
        delta_pos_t = delta_pos[:, :, :-1, :]  # [na, na, F-1, 2]
        delta_vel_t = delta_vel[:, :, :-1, :]  # [na, na, F-1, 2]
        
        # 投影计算
        dot_product = torch.sum(delta_pos_t * delta_vel_t, dim=-1)  # [na, na, F-1]
        vel_norm_sq = torch.sum(delta_vel_t**2, dim=-1) + 1e-7
        ttc = torch.where(
            (vel_norm_sq > 1e-3) & (dot_product < 0),
            -dot_product / vel_norm_sq,
            torch.finfo(torch.float32).max
        )
        # 计算横向距离（假设x为纵向，y为横向）
        lateral_distance = torch.abs(delta_pos[..., 1])  # [na, na, F]

        lateral_deficit = torch.zeros_like(lateral_distance)
        if vn_mask.any():
            # 仅在vn_mask位置计算横向违规
            raw_deficit = self.lateral_threshold - lateral_distance[vn_mask]
            # 使用安全除法，防止threshold为零（尽管通常应大于零）
            lateral_deficit[vn_mask] = torch.relu(raw_deficit) / (self.lateral_threshold + 1e-9)


        # ================== 违规程度计算 ==================
        # 距离违规差值（正值表示违规程度）
        safe_lon_threshold = lon_threshold + 1e-9
        distance_deficit = torch.relu(lon_threshold - distance_matrix) / safe_lon_threshold
        

        # TTC阈值矩阵（根据交互类型分配）
        ttc_threshold = torch.zeros_like(ttc)  # 初始为全0
        ttc_threshold[vv_mask] = self.ttc_vehicle  # 车-车
        ttc_threshold[vn_mask] = self.ttc_vru      # 车-VRU
        
        # TTC违规差值计算（阈值 - 实际TTC）
        # 计算TTC违规差值（非掩码位置阈值0，此时只有ttc<0会产生违规，而ttc通常>=0）
        ttc_deficit = torch.relu(ttc_threshold - ttc)
        # 安全除法：在掩码位置使用实际阈值，非掩码位置分母为1e-9（此时分子已为0）
        safe_ttc_threshold = ttc_threshold + (ttc_threshold == 0) * 1e-9  # 保持分母非零
        ttc_deficit = ttc_deficit / safe_ttc_threshold #[na, na, F-1]
        ttc_deficit = F.pad(ttc_deficit, (0, 1), mode='constant', value=0)#[na, na, F]

        # ================== 综合违规聚合 ==================
        # 各违规矩阵已确保非掩码位置为0，直接取最大值
        combined_deficit = torch.max(
            distance_deficit,
            torch.max(ttc_deficit, lateral_deficit)
        )

        # 清理可能的数值异常（根据实际情况可选）
        combined_deficit = torch.nan_to_num(combined_deficit, nan=0.0, posinf=0.0, neginf=0.0) #[N, N, F]
        
        # 过滤无效交互
        combined_deficit[nn_mask] = 0          # 忽略VRU-VRU
        diag_indices = torch.arange(combined_deficit.shape[0])  # 生成 [0, 1, ..., N-1]
        combined_deficit[diag_indices, diag_indices, :] = 0  # 把所有 F 维度的对角线置零

        # 计算所有帧的平均违规分数（包括非违规帧）
        violation_scores = combined_deficit.mean(dim=(1,2))
        
        return violation_scores 


    def training_step(self,data,batch_idx):
        #traj_propose, traj_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F],[(N1,...,Nb),H,K]
        traj_propose, _ , traj_output, violation_scores = self(data)
        agent_mask = data['agent']['category'] == 1
        traj_propose = traj_propose[agent_mask]
        traj_output = traj_output[agent_mask]
        violation_scores = violation_scores[agent_mask]

        target_traj, target_mask = generate_target(position=data['agent']['position'], 
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  #[(N1,...Nb),H,F,2],[(N1,...Nb),H,F]

        target_traj = target_traj[agent_mask]
        target_mask = target_mask[agent_mask]
        
        agent_batch = data['agent']['batch'][agent_mask]
        batch_size = len(data['case_id'])
        errors = (torch.norm(traj_propose - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)  #[(n1,...nb),H,K]
        joint_errors = [error.sum(dim=0, keepdim=True) for error in unbatch(errors, agent_batch)]
        joint_errors = torch.cat(joint_errors, dim=0)    #[b,H,K]

        num_agent_pre_batch = torch.bincount(agent_batch)
        best_mode_index = joint_errors.argmin(dim=-1)     #[b,H]
        best_mode_index = best_mode_index.repeat_interleave(num_agent_pre_batch, 0)     #[(N1,...Nb),H]
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
        traj_best_output = traj_output[torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
        
        predict_mask = generate_predict_mask(data['agent']['visible_mask'][agent_mask,:self.num_historical_steps], self.num_visible_steps)   #[(n1,...nb),H]
        targ_mask = target_mask[predict_mask]                               #[Na,F]
        traj_pro = traj_best_propose[predict_mask]                          #[Na,F,2]
        traj_ref = traj_best_output[predict_mask]                           #[Na,F,2]
        targ_traj = target_traj[predict_mask]                               #[Na,F,2]
        category = data['agent']['category'][agent_mask]
        category_expanded = category.unsqueeze(1).expand(-1, self.num_historical_steps)
        valid_category = category_expanded[predict_mask]  # [Na]

        gt_violation_scores = self.calculate_safety_violation(
            pred_traj=targ_traj,
            agent_data = valid_category )
        # 提取最优模态的违规值
        best_violation_scores = violation_scores[torch.arange(violation_scores.size(0))[:, None], best_mode_index]  # [N, H]
        # 过滤预测时间步的违规值
        best_violation_scores = best_violation_scores[predict_mask]  # [Na]

        total_loss, pro_loss, refine_loss, violation_loss = self.joint_loss(traj_pro[targ_mask], traj_ref[targ_mask],targ_traj[targ_mask], best_violation_scores, gt_violation_scores)
        self.log('train_reg_loss_traj_propose', pro_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_reg_loss_traj_refine', refine_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_reg_loss_violation', violation_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return total_loss

    def validation_step(self,data,batch_idx):
        traj_propose, _ , traj_output, violation_scores = self(data)
        agent_mask = data['agent']['category'] == 1
        traj_propose = traj_propose[agent_mask]
        traj_output = traj_output[agent_mask]
        violation_scores = violation_scores[agent_mask]

        target_traj, target_mask = generate_target(position=data['agent']['position'], 
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  #[(N1,...Nb),H,F,2],[(N1,...Nb),H,F]

        target_traj = target_traj[agent_mask]
        target_mask = target_mask[agent_mask]
        
        agent_batch = data['agent']['batch'][agent_mask]
        batch_size = len(data['case_id'])
        errors = (torch.norm(traj_propose - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)  #[(n1,...nb),H,K]
        joint_errors = [error.sum(dim=0, keepdim=True) for error in unbatch(errors, agent_batch)]
        joint_errors = torch.cat(joint_errors, dim=0)    #[b,H,K]

        num_agent_pre_batch = torch.bincount(agent_batch)
        best_mode_index = joint_errors.argmin(dim=-1)     #[b,H]
        best_mode_index = best_mode_index.repeat_interleave(num_agent_pre_batch, 0)     #[(N1,...Nb),H]
        traj_best_propose = traj_propose[torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
        traj_best_output = traj_output[torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None, :], best_mode_index]   #[(n1,...nb),H,F,2]
        
        predict_mask = generate_predict_mask(data['agent']['visible_mask'][agent_mask,:self.num_historical_steps], self.num_visible_steps)   #[(n1,...nb),H]
        targ_mask = target_mask[predict_mask]                               #[Na,F]
        traj_pro = traj_best_propose[predict_mask]                          #[Na,F,2]
        traj_ref = traj_best_output[predict_mask]                           #[Na,F,2]
        targ_traj = target_traj[predict_mask]                               #[Na,F,2]
        category = data['agent']['category'][agent_mask]
        category_expanded = category.unsqueeze(1).expand(-1, self.num_historical_steps)
        valid_category = category_expanded[predict_mask]  # [Na]

        gt_violation_scores = self.calculate_safety_violation(
            pred_traj=targ_traj,
            agent_data = valid_category )
        # 提取最优模态的违规值
        best_violation_scores = violation_scores[torch.arange(violation_scores.size(0))[:, None], best_mode_index]  # [N, H]
        # 过滤预测时间步的违规值
        best_violation_scores = best_violation_scores[predict_mask]  # [Na]

        total_loss, pro_loss, refine_loss, violation_loss = self.joint_loss(traj_pro[targ_mask], traj_ref[targ_mask],targ_traj[targ_mask], best_violation_scores, gt_violation_scores)
        self.log('val_reg_loss_traj_propose', pro_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_reg_loss_traj_refine', refine_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_reg_loss_violation', violation_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_gt_violation', gt_violation_scores.mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_violation', best_violation_scores.mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        
        visible_mask = data['agent']['visible_mask'][agent_mask]                      #[(n1,...nb),H+F]
        visible_num = visible_mask.sum(dim=-1)                                        #[(n1,...nb)]
        scored_mask = visible_num == self.num_historical_steps + self.num_future_steps
        scored_predict_traj = unbatch(traj_output[scored_mask,-1], agent_batch[scored_mask])                   #[(n1,K,F,2),...,(nb,K,F,2)]
        scored_target_traj = unbatch(target_traj[scored_mask,-1], agent_batch[scored_mask])                    #[(n1,F,2),...,(nb,F,2)]
        scored_target_mask = unbatch(target_mask[scored_mask,-1], agent_batch[scored_mask])                    #[(n1,F),...,(nb,F)]

        self.min_joint_ade.update(scored_predict_traj, scored_target_traj, scored_target_mask)
        self.min_joint_fde.update(scored_predict_traj, scored_target_traj, scored_target_mask)
        self.log('val_minJointADE', self.min_joint_ade, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val_minJointFDE', self.min_joint_fde, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)


    def test_step(self,data,batch_idx):
        traj_propose, _ , traj_output, violation_scores = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F]

        agent_mask = data['agent']['category'] == 1
        agent_batch = data['agent']['batch'][agent_mask]
        batch_size = len(data['case_id'])
        num_agent_pre_batch = torch.bincount(agent_batch)

        scenario_name = data['scenario_name']   #[b]
        case_id = data['case_id']               #[b]
        agent_id = data['agent']['id'][agent_mask]
        agent_interset = data['agent']['interest'][agent_mask]
        traj_output = traj_output[agent_mask, -1]
        tep = torch.cat([data['agent']['position'][agent_mask, -1:].unsqueeze(1).repeat_interleave(self.num_modes,1), traj_output], dim=-2) #[(n1+...+nb),K,F+1,2]
        _, yaw_output = compute_angles_lengths_2D(tep[:,:,1:] - tep[:,:,:-1])   #[(n1+...+nb),K,F]

        scored_agent_id = unbatch(agent_id, agent_batch)                        #[n1,...nb]
        scored_agent_interset = unbatch(agent_interset, agent_batch)            #[n1,...nb]
        scored_predict_traj = unbatch(traj_output, agent_batch)           #[(n1,K,F,2),...,(nb,K,F,2)]
        scored_predict_yaw = unbatch(yaw_output, agent_batch)             #[(n1,K,F),...,(nb,K,F)]
        
        case_id = case_id.cpu().numpy()
        scored_agent_id = [agent_id.cpu().numpy() for agent_id in scored_agent_id]
        scored_agent_interset = [agent_interset.cpu().numpy() for agent_interset in scored_agent_interset]
        scored_predict_traj = [predict_traj.cpu().numpy() for predict_traj in scored_predict_traj]
        scored_predict_yaw = [predict_yaw.cpu().numpy() for predict_yaw in scored_predict_yaw]
        
        scored_frame_id = list(range(30))
        scored_frame_id = [id + 11 for id in scored_frame_id]
        scored_timestamp_ms = [frame_id * 100 for frame_id in scored_frame_id]

        for i in range(batch_size):
            rows = []
            for j in range(num_agent_pre_batch[i]):
                for k in range(self.num_future_steps):
                    row = [case_id[i], scored_agent_id[i][j], scored_frame_id[k], scored_timestamp_ms[k], scored_agent_interset[i][j],
                        scored_predict_traj[i][j,0,k,0], scored_predict_traj[i][j,0,k,1], scored_predict_yaw[i][j,0,k],
                        scored_predict_traj[i][j,1,k,0], scored_predict_traj[i][j,1,k,1], scored_predict_yaw[i][j,1,k],
                        scored_predict_traj[i][j,2,k,0], scored_predict_traj[i][j,2,k,1], scored_predict_yaw[i][j,2,k],
                        scored_predict_traj[i][j,3,k,0], scored_predict_traj[i][j,3,k,1], scored_predict_yaw[i][j,3,k],
                        scored_predict_traj[i][j,4,k,0], scored_predict_traj[i][j,4,k,1], scored_predict_yaw[i][j,4,k],
                        scored_predict_traj[i][j,5,k,0], scored_predict_traj[i][j,5,k,1], scored_predict_yaw[i][j,5,k]]
                    rows.append(row)

            if scenario_name[i] in self.test_output:
                self.test_output[scenario_name[i]] = self.test_output[scenario_name[i]] + rows
            else:
                self.test_output[scenario_name[i]] = rows

    def on_test_end(self):
        for key, value in self.test_output.items():
            df = pd.DataFrame(value, columns=self._columns)
            df['track_to_predict'] = 1
            df.to_csv('./test_output/' + key + '_sub.csv', index=False)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        
        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('RuleNet')
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_historical_steps', type=int, default=10)
        parser.add_argument('--num_future_steps', type=int, default=30)
        parser.add_argument('--duration', type=int, default=20)
        parser.add_argument('--a2a_radius', type=float, default=80)
        parser.add_argument('--l2a_radius', type=float, default=80)
        parser.add_argument('--num_visible_steps', type=int, default=3)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_attn_layers', type=int, default=3)
        parser.add_argument('--num_hops', type=int, default=4)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--warmup_epochs', type=int, default=4)
        parser.add_argument('--T_max', type=int, default=100)
        return parent_parser
