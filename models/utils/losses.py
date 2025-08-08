import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, logit, target):
        return focal_loss(F.cross_entropy(logit, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super().__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(cls_num_list))
        m_list = m_list * (max_m / torch.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s

    def forward(self, logit, target):
        index = torch.zeros_like(logit, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        logit_m = logit - batch_m * self.s  # scale only the margin, as the logit is already scaled.

        output = torch.where(index, logit_m, logit)
        return F.cross_entropy(output, target)


class ClassBalancedLoss(nn.Module):
    def __init__(self, cls_num_list, beta=0.9999):
        super().__init__()
        per_cls_weights = (1.0 - beta) / (1.0 - (beta ** cls_num_list))
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class GeneralizedReweightLoss(nn.Module):
    def __init__(self, cls_num_list, exp_scale=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        per_cls_weights = 1.0 / (cls_num_ratio ** exp_scale)
        per_cls_weights = per_cls_weights / torch.mean(per_cls_weights)
        self.per_cls_weights = per_cls_weights
    
    def forward(self, logit, target):
        logit = logit.to(self.per_cls_weights.dtype)
        return F.cross_entropy(logit, target, weight=self.per_cls_weights)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num

    def forward(self, logit, target):
        logit_adjusted = logit + self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)


class LogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0, factor=1.):
        super().__init__()
        if torch.sum(cls_num_list) < 10:
            # self.log_cls_num = cls_num_list
            log_cls_num = torch.log(cls_num_list + 1e-9)
            self.log_cls_num = log_cls_num
        else:
            cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
            log_cls_num = torch.log(cls_num_ratio + 1e-9)
            self.log_cls_num = log_cls_num
        self.tau = tau
        self.factor = factor

    def forward(self, logit, target):
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0)
        return F.cross_entropy(logit_adjusted * self.factor, target)

class DualLogitAdjustedLoss(nn.Module):
    def __init__(self, cls_num_list1, cls_num_list2, bias1, bias2, tau=1.0):
        super().__init__()
        cls_num_ratio1 = cls_num_list1 / torch.sum(cls_num_list1)
        log_cls_num1 = torch.log(cls_num_ratio1)
        self.log_cls_num1 = log_cls_num1

        cls_num_ratio2 = cls_num_list2 / torch.sum(cls_num_list2)
        log_cls_num2 = torch.log(cls_num_ratio2)
        self.log_cls_num2 = log_cls_num2

        self.tau = tau

        self.bias1, self.bias2 = bias1, bias2

    def forward(self, logit1, target1, logit2, target2):
        logit_adjusted1 = logit1 + self.tau * self.log_cls_num1.unsqueeze(0) - self.bias1
        logit_adjusted2 = logit2 + self.tau * self.log_cls_num2.unsqueeze(0) - self.bias2

        logit_adjusted = torch.cat([logit_adjusted1, logit_adjusted2], dim=0)
        target = torch.cat([target1, target2], dim=0)
        # logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0) + self.tau * self.log_prior_ratio.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)


class LADELoss(nn.Module):
    def __init__(self, cls_num_list, remine_lambda=0.1, estim_loss_weight=0.1):
        super().__init__()
        self.num_classes = len(cls_num_list)
        self.prior = cls_num_list / torch.sum(cls_num_list)

        self.balanced_prior = torch.tensor(1. / self.num_classes).float().to(self.prior.device)
        self.remine_lambda = remine_lambda

        self.cls_weight = cls_num_list / torch.sum(cls_num_list)
        self.estim_loss_weight = estim_loss_weight

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)
 
        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, logit, target):
        logit_adjusted = logit + torch.log(self.prior).unsqueeze(0)
        ce_loss =  F.cross_entropy(logit_adjusted, target)

        per_cls_pred_spread = logit.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (logit - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)
        estim_loss = -torch.sum(estim_loss * self.cls_weight)

        return ce_loss + self.estim_loss_weight * estim_loss


# class MultiExpertLoss(nn.Module):
#     def __init__(self, cls_num_list, tau=1.0):
#         super().__init__()
#         cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
#         log_cls_num = torch.log(cls_num_ratio)
#         self.log_cls_num = log_cls_num
#         self.tau = tau

#     def forward(self, logit_dict, target_dict):
#         coarse_res, fine0_res, fine1_res, fine2_res = logit_dict['coarse_res'], logit_dict['fine0_res'], logit_dict['fine1_res'], logit_dict['fine2_res']
#         group0, group1, group2 = logit_dict['group0'], logit_dict['group1'], logit_dict['group2']
#         coarse_target, label_target = target_dict['coarse_target'], target_dict['label_target']
        

#         loss_coarse = F.cross_entropy(coarse_res, coarse_target)

#         logit_adjusted1 = F.cross_entropy(fine0_res + self.tau * self.log_cls_num.unsqueeze(0), label_target[group0])
#         logit_adjusted2 = F.cross_entropy(fine1_res + self.tau * self.log_cls_num.unsqueeze(0), label_target[group1])
#         logit_adjusted3 = F.cross_entropy(fine2_res + self.tau * self.log_cls_num.unsqueeze(0), label_target[group2])
#         # print(loss_coarse, logit_adjusted1, logit_adjusted2, logit_adjusted3)
#         if torch.isnan(logit_adjusted1):
#             logit_adjusted1 = 0
#         if torch.isnan(logit_adjusted2):
#             logit_adjusted2 = 0
#         if torch.isnan(logit_adjusted3):
#             logit_adjusted3 = 0
#         return (logit_adjusted1 + logit_adjusted2 + logit_adjusted3) / 3 + loss_coarse

class MultiExpertLoss(nn.Module):
    def __init__(self, cls_num_list, prior, tau=1.0):
        super().__init__()
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau

        prior_ratio = prior / torch.sum(prior)
        log_prior_ratio = torch.log(prior_ratio)
        self.log_prior_ratio = log_prior_ratio


    def forward(self, logit, target):
        # fine1_res, fine2_res = logit_dict['fine1_res'], logit_dict['fine2_res']
        logit_adjusted = logit + self.tau * self.log_cls_num.unsqueeze(0) +  self.log_prior_ratio.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)


class GlobalLogitAdjustment(nn.Module):
    def __init__(self, prior, tau=1.0):
        super().__init__()
        self.tau = tau

        self.prior_ratio = prior / torch.sum(prior, dim=1)[:, None]
        
        # log_prior_ratio = torch.log(prior_ratio)
        # self.log_prior_ratio = log_prior_ratio


    def forward(self, res, label):

        base_prior = self.prior_ratio[label]
        label = F.one_hot(label, len(self.prior_ratio))
        bias = torch.log(base_prior) - torch.log(1-base_prior)
        res = res + bias
        # print(res)
        loss = F.binary_cross_entropy_with_logits(
                    res, label.float(),
                    reduction='sum') / label.shape[-1]
        # res = res + self.tau * self.log_prior_ratio.unsqueeze(0)
        # label = F.one_hot(label, len(self.log_prior_ratio))
        # res = res.sigmod()
        # res = res * label + res * (1 - label)
        # loss = -1 * torch.log(res + 1e-9)

        return loss.mean()