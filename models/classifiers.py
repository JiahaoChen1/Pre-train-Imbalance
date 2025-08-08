import torch
import torch.nn as nn
import torch.nn.functional as F


class _Classifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError

    def apply_weight(self, weight):
        self.weight.data = weight.clone()
    

class LinearClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        nn.init.kaiming_normal_(self.weight.data)
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class CosineClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight) * self.scale


class L2NormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
    
    def forward(self, x):
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)


class LayerNormedClassifier(_Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.ln = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-12, dtype=dtype)

    def forward(self, x):
        x = self.ln(x)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)

class MultiExpertClassifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, coarse_num_classes=3, **kwargs) -> None:
        super().__init__()

        self.weight_coarse = nn.Parameter(torch.empty(coarse_num_classes, feat_dim, dtype=dtype))
        self.weight_coarse.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        self.weight_fine0 = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight_fine0.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        self.weight_fine1 = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight_fine1.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        self.weight_fine2 = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight_fine2.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        self.scale = scale

    @property
    def dtype(self):
        return self.weight_coarse.dtype
    
    def forward(self, x):
        x = F.normalize(x, dim=-1)
        coarse_weight = F.normalize(self.weight_coarse, dim=-1)
        fine0_weight = F.normalize(self.weight_fine0, dim=-1)
        fine1_weight = F.normalize(self.weight_fine1, dim=-1)
        fine2_weight = F.normalize(self.weight_fine2, dim=-1)

        coarse_res = F.linear(x, coarse_weight) * self.scale
        select = torch.argmax(coarse_res, dim=1)

        group0 = (select == 0)
        group1 = (select == 1)
        group2 = (select == 2)
        group0_x, group1_x, group2_x = x[group0],  x[group1],  x[group2]

        fine0_res = F.linear(group0_x, fine0_weight) * self.scale
        fine1_res = F.linear(group1_x, fine1_weight) * self.scale
        fine2_res = F.linear(group2_x, fine2_weight) * self.scale
        
        indices = torch.tensor([i for i in range(x.shape[0])]).to(fine0_res.device).long()
        indices0, indices1, indices2 = indices[group0].unsqueeze(-1).repeat(1, fine2_res.shape[-1]), indices[group1].unsqueeze(-1).repeat(1, fine2_res.shape[-1]), indices[group2].unsqueeze(-1).repeat(1, fine2_res.shape[-1])

        output = torch.zeros((x.shape[0], fine1_res.shape[1]), dtype=fine0_res.dtype).to(fine0_res.device)
        # print(output.dtype, fine0_res.dtype)
        output.scatter_(0, indices0, fine0_res)
        output.scatter_(0, indices1, fine1_res)
        output.scatter_(0, indices2, fine2_res)
        # output = fine0_res * group0[:, None] + fine1_res * group1[:, None] + fine2_res * group2[:, None]


        return {'coarse_res':coarse_res, 
                'fine0_res':fine0_res, 
                'fine1_res':fine1_res, 
                'fine2_res':fine2_res,
                'group0':group0,
                'group1':group1,
                'group2':group2,
                'output':output}
    
    def apply_weight(self, weight):
        pass


class CoarseClassifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=-1, dtype=None, num_coarse=3, scale=30, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_coarse, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        self.scale = scale
        print('scale value is {scale}')

    @property
    def dtype(self):
        return self.weight_fine1.dtype
    
    def forward(self, x):
        x = F.normalize(x, dim=-1)

        weight = F.normalize(self.weight, dim=-1)

        res = F.linear(x, weight) * self.scale

        return res
    


class TwoExpertClassifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, **kwargs) -> None:
        super().__init__()

        self.weight_fine1 = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight_fine1.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        self.weight_fine2 = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight_fine2.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        self.scale = scale

    @property
    def dtype(self):
        return self.weight_fine1.dtype
    
    def forward(self, x):
        x = F.normalize(x, dim=-1)

        fine1_weight = F.normalize(self.weight_fine1, dim=-1)
        fine2_weight = F.normalize(self.weight_fine2, dim=-1)

        fine1_res = F.linear(x, fine1_weight) * self.scale
        fine2_res = F.linear(x, fine2_weight) * self.scale
        
        
        return {'fine1_res':fine1_res, 
                'fine2_res':fine2_res,}
    
    def apply_weight(self, weight):
        self.weight_fine1.data = weight.clone()
        self.weight_fine2.data = weight.clone()