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


import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
