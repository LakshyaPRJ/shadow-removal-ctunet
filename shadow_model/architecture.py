"""
architecture.py
CTU-Net+ Shadow Detection & Removal — all model classes.
Drop this file into your shadow_model/ folder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────
#  Utility blocks
# ─────────────────────────────────────────────────────────────────
class ConvBnReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnReLU(ch, ch), ConvBnReLU(ch, ch)
        )
    def forward(self, x): return x + self.block(x)


# ─────────────────────────────────────────────────────────────────
#  Chromaticity-Invariant Multi-Representation Encoder
#  (Section III-B2: RGB + log-chromaticity + normalized RGB)
# ─────────────────────────────────────────────────────────────────
class MultiRepEncoder(nn.Module):
    def __init__(self, out_ch=64):
        super().__init__()
        self.enc_rgb   = self._make_enc(3, out_ch)
        self.enc_logch = self._make_enc(3, out_ch)
        self.enc_norm  = self._make_enc(3, out_ch)
        self.attn = nn.Sequential(
            nn.Conv2d(out_ch * 3, 3, 1),
            nn.Softmax(dim=1)
        )
        self.merge = nn.Conv2d(out_ch, out_ch, 1)

    def _make_enc(self, in_ch, out_ch):
        return nn.Sequential(
            ConvBnReLU(in_ch, out_ch, k=7, p=3),
            ConvBnReLU(out_ch, out_ch)
        )

    @staticmethod
    def log_chromaticity(x):
        x = x * 0.5 + 0.5 + 1e-6
        log_x = torch.log(x.clamp(min=1e-6))
        mean  = log_x.mean(dim=1, keepdim=True)
        return log_x - mean

    @staticmethod
    def norm_rgb(x):
        n = x.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return x / n

    def forward(self, x):
        f_rgb   = self.enc_rgb(x)
        f_logch = self.enc_logch(self.log_chromaticity(x))
        f_norm  = self.enc_norm(self.norm_rgb(x))
        cat   = torch.cat([f_rgb, f_logch, f_norm], dim=1)
        w     = self.attn(cat)
        fused = (f_rgb   * w[:,0:1] +
                 f_logch * w[:,1:2] +
                 f_norm  * w[:,2:3])
        return self.merge(fused)


# ─────────────────────────────────────────────────────────────────
#  CNN Encoder Backbone (U-Net style)
# ─────────────────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(ConvBnReLU(64, 64),  ResBlock(64))
        self.stage2 = nn.Sequential(nn.MaxPool2d(2),
                                    ConvBnReLU(64, 128), ResBlock(128))
        self.stage3 = nn.Sequential(nn.MaxPool2d(2),
                                    ConvBnReLU(128, 256), ResBlock(256))
        self.stage4 = nn.Sequential(nn.MaxPool2d(2),
                                    ConvBnReLU(256, 512), ResBlock(512))

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return s1, s2, s3, s4


# ─────────────────────────────────────────────────────────────────
#  Lightweight Transformer Block (CTFA)
# ─────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                           dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim), nn.Dropout(drop)
        )

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────
#  Material-Aware Branch (Section III-B1)
# ─────────────────────────────────────────────────────────────────
NUM_MATERIALS = 4  # brick, vegetation, metal, fabric

class MaterialAwareBranch(nn.Module):
    def __init__(self, in_ch=512, num_materials=NUM_MATERIALS):
        super().__init__()
        self.mat_head = nn.Sequential(
            ConvBnReLU(in_ch, 256),
            ConvBnReLU(256, 128),
            nn.Conv2d(128, num_materials, 1),
            nn.Sigmoid()
        )
        self.fuse_w = nn.Parameter(torch.ones(1))

    def forward(self, feat):
        mat_map  = self.mat_head(feat)
        mat_feat = mat_map.mean(dim=1, keepdim=True)
        enhanced = feat + self.fuse_w * (feat * mat_feat)
        return enhanced, mat_map


# ─────────────────────────────────────────────────────────────────
#  U-Net Decoder  (size-safe via F.interpolate)
# ─────────────────────────────────────────────────────────────────
class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = ConvBnReLU(512+256, 256)
        self.conv2 = ConvBnReLU(256+128, 128)
        self.conv1 = ConvBnReLU(128+64,  64)
        self.out   = nn.Conv2d(64, 1, 1)

    def forward(self, s1, s2, s3, s4):
        x = F.interpolate(s4, size=s3.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conv3(torch.cat([x, s3], dim=1))
        x = F.interpolate(x,  size=s2.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conv2(torch.cat([x, s2], dim=1))
        x = F.interpolate(x,  size=s1.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conv1(torch.cat([x, s1], dim=1))
        return self.out(x)


# ─────────────────────────────────────────────────────────────────
#  CTU-Net+ Full Detector  (Section III-A & III-B)
# ─────────────────────────────────────────────────────────────────
class CTUNetPlus(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.multi_rep  = MultiRepEncoder(out_ch=64)
        self.cnn_enc    = CNNEncoder()
        self.mat_branch = MaterialAwareBranch(in_ch=512)
        self.proj_in    = nn.Linear(512, embed_dim)
        self.tf_blocks  = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(2)
        ])
        self.proj_out   = nn.Linear(embed_dim, 512)
        self.decoder    = UNetDecoder()

    def forward(self, x):
        x_feat = self.multi_rep(x)
        s1, s2, s3, s4 = self.cnn_enc(x_feat)
        s4_mat, mat_map = self.mat_branch(s4)
        B, C, H, W = s4_mat.shape
        tokens = rearrange(s4_mat, 'b c h w -> b (h w) c')
        tokens = self.proj_in(tokens)
        tokens = self.tf_blocks(tokens)
        tokens = self.proj_out(tokens)
        s4_tf  = rearrange(tokens, 'b (h w) c -> b c h w', h=H, w=W)
        s4_fused    = s4_mat + s4_tf
        mask_logits = self.decoder(s1, s2, s3, s4_fused)
        return mask_logits, mat_map, s1, s2, s3, s4_fused


# ─────────────────────────────────────────────────────────────────
#  LICM — Learned Illumination Compensation Module (Section III-C)
# ─────────────────────────────────────────────────────────────────
class LICM(nn.Module):
    def __init__(self, feat_ch=64):
        super().__init__()
        self.feat_proj = nn.Conv2d(512, feat_ch, 1)
        in_ch = 3 + 1 + feat_ch
        self.dil_block = nn.Sequential(
            ConvBnReLU(in_ch, 64, dilation=1, p=1),
            ResBlock(64),
            ConvBnReLU(64,    64, dilation=2, p=2),
            ResBlock(64),
            ConvBnReLU(64,    64, dilation=4, p=4),
            ResBlock(64),
        )
        self.illum_head = nn.Sequential(
            nn.Conv2d(64, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 3, 1),  nn.Sigmoid()
        )
        self.refl_head = nn.Sequential(
            nn.Conv2d(64, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 3, 1),  nn.Sigmoid()
        )

    def forward(self, img, mask, feat_s4):
        feat = self.feat_proj(feat_s4)
        feat = F.interpolate(feat, size=img.shape[-2:],
                             mode='bilinear', align_corners=True)
        x     = torch.cat([img, mask, feat], dim=1)
        h     = self.dil_block(x)
        illum = self.illum_head(h)
        refl  = self.refl_head(h)
        return illum, refl


# ─────────────────────────────────────────────────────────────────
#  ARSM — Adaptive Region Segmentation & Learned Matching (III-D)
# ─────────────────────────────────────────────────────────────────
class SuperpixelNet(nn.Module):
    def __init__(self, in_ch=512, K=16):
        super().__init__()
        self.K = K
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 128, 1), nn.ReLU(),
            nn.Conv2d(128, K, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, feat): return self.head(feat)


class PatchEncoder(nn.Module):
    def __init__(self, in_ch=512, embed_dim=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp  = nn.Sequential(
            nn.Linear(in_ch, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.accept_thresh = nn.Parameter(torch.tensor(0.5))

    def encode_superpixel(self, feat, assignments):
        B, K, H, W = assignments.shape
        C = feat.shape[1]
        feat_flat   = feat.view(B, C, -1)
        assign_flat = assignments.view(B, K, -1)
        numer   = torch.bmm(assign_flat, feat_flat.permute(0,2,1))
        denom   = assign_flat.sum(dim=-1, keepdim=True) + 1e-6
        sp_feat = numer / denom
        return self.mlp(sp_feat)


class ARSM(nn.Module):
    def __init__(self, K=16, feat_ch=512, embed_dim=256):
        super().__init__()
        self.sp_net = SuperpixelNet(feat_ch, K)
        self.enc    = PatchEncoder(feat_ch, embed_dim)
        self.K      = K

    def forward(self, feat, mask_pred):
        B, C, H, W = feat.shape
        mask_ds     = F.interpolate(mask_pred.detach(), size=(H, W),
                                    mode='bilinear', align_corners=True)
        assign      = self.sp_net(feat)
        sp_emb      = self.enc.encode_superpixel(feat, assign)
        mask_flat   = mask_ds.view(B, 1, -1)
        assign_flat = assign.view(B, self.K, -1)
        sp_shadow_score = (assign_flat * mask_flat).mean(dim=-1)
        is_shadow   = sp_shadow_score > 0.5
        sp_emb_n    = F.normalize(sp_emb, dim=-1)
        sim_mat     = torch.bmm(sp_emb_n, sp_emb_n.permute(0,2,1))
        transfer    = feat.clone()
        for b in range(B):
            sh_idx  = is_shadow[b].nonzero(as_tuple=True)[0]
            nsh_idx = (~is_shadow[b]).nonzero(as_tuple=True)[0]
            if sh_idx.numel() == 0 or nsh_idx.numel() == 0:
                continue
            for si in sh_idx:
                sims       = sim_mat[b, si, nsh_idx]
                best       = nsh_idx[sims.argmax()]
                sh_mask_k  = assign[b, si].unsqueeze(0)
                nsh_feat_k = (feat[b] * assign[b, best]).sum((-1,-2),
                              keepdim=True) / (assign[b,best].sum()+1e-6)
                transfer[b] += sh_mask_k * nsh_feat_k
        return transfer


# ─────────────────────────────────────────────────────────────────
#  PBR — Perceptual Boundary Refinement (Section III-F)
# ─────────────────────────────────────────────────────────────────
class NonLinearCompOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3+3+64, 128), nn.ReLU(),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, 3),       nn.Tanh()
        )
        self.feat_proj = nn.AdaptiveAvgPool2d(1)

    def forward(self, img, illum, feat):
        B, _, H, W = img.shape
        patch_emb = F.adaptive_avg_pool2d(
            feat.detach(), 1).expand(-1, -1, H, W)
        patch_emb = F.interpolate(
            patch_emb, size=(H, W), mode='bilinear', align_corners=True)
        patch_emb = patch_emb[:, :64]
        inp      = torch.cat([img, illum, patch_emb], dim=1)
        inp_flat = inp.permute(0,2,3,1).reshape(-1, 3+3+64)
        out_flat = self.mlp(inp_flat)
        return out_flat.reshape(B, H, W, 3).permute(0,3,1,2)


class PBRGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBnReLU(4,  64), ConvBnReLU(64, 128),
            nn.MaxPool2d(2),
            ConvBnReLU(128, 256), ResBlock(256),
            nn.MaxPool2d(2),
        )
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBnReLU(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBnReLU(128, 64),
            nn.Conv2d(64, 3, 1), nn.Tanh()
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        return self.dec(self.enc(x))


class PBRDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1),
        )

    def forward(self, x): return self.net(x)


# ─────────────────────────────────────────────────────────────────
#  Full End-to-End Model (Section III-A)
# ─────────────────────────────────────────────────────────────────
class FullModel(nn.Module):
    """
    Full CTU-Net+ pipeline:
      Image → CTU-Net+ → shadow mask
            → LICM     → illumination map, reflectance
            → ARSM     → feature transfer
            → NonLinearCompOp → initial removal
            → PBRGenerator    → final shadow-free image
    """
    def __init__(self, cfg):
        super().__init__()
        self.detector = CTUNetPlus(embed_dim=cfg['embed_dim'],
                                   num_heads=cfg['num_heads'])
        self.licm     = LICM(feat_ch=64)
        self.arsm     = ARSM(K=cfg['K_superpixels'],
                              feat_ch=512, embed_dim=cfg['embed_dim'])
        self.comp_op  = NonLinearCompOp()
        self.pbr_gen  = PBRGenerator()

    def forward(self, img):
        # 1. Shadow detection
        mask_logits, mat_map, s1, s2, s3, s4 = self.detector(img)
        mask_prob = torch.sigmoid(mask_logits)

        # 2. De-normalize for pixel-space operations
        img_01 = img * 0.5 + 0.5

        # 3. Illumination estimation
        illum, refl = self.licm(img, mask_prob, s4)

        # 4. ARSM feature transfer
        s4_transfer = self.arsm(s4, mask_prob)

        # 5. Non-linear pixel compensation
        corrected = self.comp_op(img, illum, s4_transfer)
        corrected = corrected * 0.5 + 0.5

        # 6. Blend: corrected in shadow regions, original elsewhere
        mask_up = F.interpolate(mask_prob, size=img.shape[-2:],
                                mode='bilinear', align_corners=True)
        initial_removed = img_01 * (1 - mask_up) + corrected * mask_up

        # 7. PBR refinement
        inp_gen    = initial_removed * 2 - 1
        refined    = self.pbr_gen(inp_gen, mask_up)
        refined_01 = refined * 0.5 + 0.5

        return {
            'mask_logits':     mask_logits,
            'mask_prob':       mask_prob,
            'mat_map':         mat_map,
            'illum':           illum,
            'refl':            refl,
            'img_01':          img_01,
            'initial_removed': initial_removed,
            'refined':         refined_01,
            's1':              s1,
        }
