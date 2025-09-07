from losses import HuberTrajLoss
import torch.nn as nn

class JointLoss(nn.Module):
    def __init__(self, lambda_violation=0.1):
        super(JointLoss, self).__init__()
        self.huber_loss = HuberTrajLoss()
        self.lambda_violation = lambda_violation

    def forward(self, traj_pro, predictions, targets, violation_scores, gt_violation_scores):
        # 计算 Huber Loss
        pro_loss = self.huber_loss(traj_pro, targets)
        refine_loss = self.huber_loss(predictions, targets)
        
        # 计算违规损失
        violation_loss = self.huber_loss(violation_scores, gt_violation_scores)
        
        # 联合损失
        total_loss = pro_loss + refine_loss + self.lambda_violation * violation_loss
        return total_loss, pro_loss, refine_loss, violation_loss
