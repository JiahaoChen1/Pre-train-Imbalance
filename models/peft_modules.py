import math
from operator import mul
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout
from torch.nn.modules.utils import _pair

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip import clip
_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        n_txt_ctx = cfg.txt_num_tokens
        txt_ctx_dim = clip_model.ln_final.weight.shape[0]

        if cfg.dual_prompt:
            n_vis_ctx = cfg.vis_num_tokens
            vis_ctx_dim = clip_model.visual.conv1.weight.shape[0]
            clip_imsize = clip_model.visual.input_resolution
            cfg_imsize = cfg.resolution
            patch_size = clip_model.visual.conv1.weight.shape[-1]
            assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

            self.vpt_dropout = Dropout(0.)
            vpt_dim = vis_ctx_dim
            clip_patchsize = _pair(patch_size)
            val = math.sqrt(6. / float(3 * reduce(mul, clip_patchsize, 1) + vpt_dim))
            self.vis_ctx = nn.Parameter(torch.zeros(1, n_vis_ctx, vpt_dim, dtype=dtype)) # [1, n_ctx, dim] = [1, 16, 768]
            nn.init.uniform_(self.vis_ctx.data, -val, val)
    
        print("Initializing a generic context")
        txt_ctx_vectors = torch.empty(n_txt_ctx, txt_ctx_dim, dtype=dtype)
        nn.init.normal_(txt_ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_txt_ctx)
        self.txt_ctx = nn.Parameter(txt_ctx_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_txt_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        text_prompts = [prompt_prefix + " " + name + "." for name in classnames] # NOTE: 'X X X X X {cls}'

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in text_prompts])  # NOTE: [cls, 77]

        with torch.no_grad():
            txt_embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", txt_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", txt_embedding[:, 1 + n_txt_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_txt_ctx = n_txt_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def forward_vis(self, x):
        vis_ctx = self.vis_ctx
        B = x.shape[0]
        ctx = self.vpt_dropout(vis_ctx.expand(B, -1, -1)).to(x.dtype)
        prefix = x[:, :1, :]
        suffix = x[:, 1:, :]

        prompt = torch.cat(
            [
                prefix, # [B, 1, dim] 
                ctx,    # [B, n_txt_ctx, dim]
                suffix, # [B, patches, dim]
            ],
            dim=1,
        )

        return prompt

    def forward_txt(self):
        ctx = self.txt_ctx  # [TXT_NUM_TOKENS, dim] = [16, 512] (default)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_txt_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class VPT(nn.Module):
    def __init__(self, vpt_len, seq_len, patch_size, emb_dim, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, emb_dim, dtype=dtype))
        init_val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + emb_dim))
        nn.init.uniform_(self.prompt, -init_val, init_val)
    
    @property
    def dtype(self):
        return self.prompt.dtype

    def forward(self, x):
        x = x[:, :self.seq_len, :]
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt], dim=1)
        return x


class Adapter(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    @property
    def dtype(self):
        return self.ln.weight.dtype
    
    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x


class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))

        # self.gate = nn.Linear(in_dim, in_dim, dtype=dtype)
        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def forward(self, x):

        # route = self.gate(x)

        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        x = x * self.scale

        # x = x * route.sigmoid()
        return x


class LoRA(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_dim, bottle_dim, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(bottle_dim, in_dim, dtype=dtype))
        self.scaling = 1.0 / bottle_dim
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def dtype(self):
        return self.lora_A.dtype

    def forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = self.scaling * x
        return x


class SSF(nn.Module):
    def __init__(self, in_dim, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_dim, dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(in_dim, dtype=dtype))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    @property
    def dtype(self):
        return self.scale.dtype

    def forward(self, x):
        if len(x.shape) == 4:  # for CNN
            return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        else:
            return x * self.scale + self.shift

class MaskedLinear(nn.Module):
    def __init__(self, weight, bias, ratio=0.0, generator=None):
        super().__init__()
        # weight: (out_dim, in_dim)
        # bias: (out_dim)
        out_dim, in_dim = weight.shape
        num_params = out_dim * in_dim + out_dim
        ratio = float(eval(ratio)) if isinstance(ratio, str) else float(ratio)
        num_masked = int(num_params * ratio)

        # randomly select the optimized parameters
        masked_indexs = torch.randperm(num_params, generator=generator)[:num_masked]
        mask = torch.zeros(num_params, dtype=bool).scatter(dim=0, index=masked_indexs, value=True)
        mask = mask.reshape(out_dim, in_dim + 1)
        self.mask_weight = mask[:,:-1]
        self.mask_bias = mask[:,-1]

        self.optimized_weight = nn.Parameter(torch.masked_select(weight.detach(), mask=self.mask_weight))
        self.optimized_bias = nn.Parameter(torch.masked_select(bias.detach(), mask=self.mask_bias))

    def forward(self, x, weight, bias):
        self.mask_weight = self.mask_weight.to(weight.device)
        self.mask_bias = self.mask_bias.to(bias.device)

        if self.mask_weight.sum() > 0:
            weight = torch.masked_scatter(weight, mask=self.mask_weight, source=self.optimized_weight)
        if self.mask_bias.sum() > 0:
            bias = torch.masked_scatter(bias, mask=self.mask_bias, source=self.optimized_bias)
        return F.linear(x, weight, bias)