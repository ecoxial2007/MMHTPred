import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from segment_anything.modeling.common import LayerNorm2d, LayerNorm
from collections import OrderedDict



class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None, attn_type='mh'):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, attn_mask=self.attn_mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context, attn = self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v))
        q = q + context
        q = q + self.mlp(self.ln_2(q))
        return q


class HTModel(nn.Module):
    def __init__(self, args):
        super(HTModel, self).__init__()
        self.use_blood_embedding = args.use_blood_embedding
        self.args = args

        mask_in_chans = args.mask_channels
        embed_dim = args.embed_dim
        dose_channels = args.dose_channels

        self.dose_downscaling = nn.Sequential(
            nn.Conv2d(dose_channels, embed_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 16),
            nn.GELU(),
            nn.Conv2d(embed_dim // 16, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=1),
        )

        if self.use_blood_embedding =="meta":
            self.meta_embedding = nn.Linear(args.meta_dim, embed_dim)
        elif self.use_blood_embedding =="blood":
            self.meta_embedding = nn.Linear(args.blood_dim, embed_dim)
        elif self.use_blood_embedding =="meta+blood":
            self.meta_embedding = nn.Linear(args.blood_dim + args.meta_dim, embed_dim)

        if args.num_mask > 0:
            self.conv_mask = nn.Conv2d(args.num_mask, embed_dim, kernel_size=1)


        # 自注意力层
        self.self_attention = ResidualCrossAttentionBlock(embed_dim, args.num_heads, 0.1)

        # 自适应池化
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.atten_pool_t = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=-2))

        # 分类器
        self.classifier = nn.Linear(embed_dim, args.num_classes)

    def forward(self, batch_dict):
        blood = batch_dict['blood']
        meta = batch_dict['meta']

        if 'doses' in self.args.method or 'images' in self.args.method:
            doses = batch_dict['doses']
            images = batch_dict['images']

            B, T, C, H, W = images.shape
            doses = rearrange(doses.unsqueeze(dim=2), 'b t c h w -> (b t) c h w')
            doses = self.dose_downscaling(doses)
            doses = rearrange(doses, '(b t) c h w -> b t c h w', b=B)

            if self.args.method == "doses":
                combined = doses
            elif self.args.method == "images":
                combined = images
            elif self.args.method == "doses+images":
                combined = doses + images



            if self.args.num_mask > 0:
                mask = []
                if self.args.use_ptv_mask:
                    mask.append(batch_dict['ptv'])
                if self.args.use_bm_mask:
                    mask.append(batch_dict['bm'])
                if self.args.use_fh_mask:
                    mask.append(batch_dict['fh'])
                if self.args.use_ub_mask:
                    mask.append(batch_dict['ub'])
                mask = rearrange(torch.stack(mask, dim=2), 'b t c h w -> (b t) c h w')
                mask = F.softmax(self.conv_mask(mask), dim=1)
                mask = rearrange(mask, '(b t) c h w -> b t c h w', b=B)
                combined = combined + combined * mask

            combined = rearrange(combined, 'b t c h w -> (b t) c h w')
            pooled = self.adaptive_pool(combined).squeeze(-1).squeeze(-1)  # (T, 256)
            pooled = rearrange(pooled, '(b t) c -> b t c', b=B)

        if self.use_blood_embedding == "meta":
            meta_emb = self.meta_embedding(meta)
        elif self.use_blood_embedding == "blood":
            meta_emb = self.meta_embedding(blood)
        elif self.use_blood_embedding == "meta+blood":
            emb = torch.cat((meta, blood), dim=1)
            meta_emb = self.meta_embedding(emb)

        if 'dose' in self.args.method or 'image' in self.args.method:
            seq_emb = pooled
            if self.use_blood_embedding != 'None':
                seq_emb = torch.cat([seq_emb, meta_emb.unsqueeze(dim=1)], dim=1)
        else:
            seq_emb = meta_emb.unsqueeze(dim=1)

        seq_emb = rearrange(seq_emb, 'b t c -> t b c')
        seq_emb = self.self_attention(seq_emb, seq_emb, seq_emb)
        seq_emb = rearrange(seq_emb, 't b c -> b t c')

        att_t = self.atten_pool_t(seq_emb)
        seq_emb = torch.sum(att_t * seq_emb, dim=1)
        logits = self.classifier(seq_emb)

        return logits
