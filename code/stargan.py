import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunc(nn.Module):
    def __init__(self, adv_weight, cls_weight, rec_weight, is_discriminator=False):
        super(LossFunc, self).__init__()
        self.adv_weight = adv_weight
        self.cls_weight = cls_weight
        self.rec_weight = rec_weight
        self.is_discriminator = is_discriminator
        self.adv_loss = nn.MSELoss()

    def classification_loss(self, logit, label):
        loss = F.cross_entropy(logit, label)
        return loss

    def cyc_loss(self, x_real, x_rec):
        return torch.mean(torch.abs(x_real - x_rec))

    def forward(self, x_real, x_rec, origin_label, target_label, d_tar_logit, d_real, d_rec, d_ori_logit):
        if not self.is_discriminator:
            # gan loss
            return self.adv_weight * self.adv_loss(d_rec, torch.ones_like(d_rec, dtype=torch.float)) + \
                self.cls_weight * self.classification_loss(d_tar_logit, target_label) + \
                self.rec_weight * self.cyc_loss(x_real, x_rec)
        else:
            return self.adv_weight * self.adv_loss(d_rec, torch.zeros_like(d_rec, dtype=torch.float)) + \
                self.adv_weight * self.adv_loss(d_real, torch.ones_like(d_rec, dtype=torch.float)) + \
                self.cls_weight * self.classification_loss(d_ori_logit, origin_label)



