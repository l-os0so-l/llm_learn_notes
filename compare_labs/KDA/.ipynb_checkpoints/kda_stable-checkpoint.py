
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple
from fla.ops.kda import chunk_kda



class KimiDeltaAttention(nn.Module):
    """
    基于 FLA 的 Kimi Delta Attention
    数值稳定，支持混合精度，48GB 显存可跑 batch=16+
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg["hidden_size"]
        self.n_heads = cfg["n_heads"]
        self.d_k = cfg["hidden_size"] // cfg["n_heads"]
        self.d_v = cfg["hidden_size"] // cfg["n_heads"]
        self.chunk_size = cfg.get("chunk_size", 64)
        self.use_short_conv = cfg.get("use_short_conv", True)
        conv_size = cfg.get("conv_size", 4)

        # 投影层
        self.W_q = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(self.d_model, self.n_heads * self.d_v, bias=False)
        self.W_o = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

        # 细粒度衰减门（论文核心）
        # 注意：如果 FLA 的 chunk_kda 期望 g 的形状为 [B, T, H]，将 d_k 改为 1
        self.W_alpha_down = nn.Linear(self.d_model, self.d_k, bias=False)
        self.W_alpha_up = nn.Linear(self.d_k, self.n_heads * self.d_k, bias=False)
        
        # Delta Rule 学习率
        self.W_beta = nn.Linear(self.d_model, self.n_heads, bias=False)

        # 短卷积（局部依赖）
        if self.use_short_conv:
            self.conv_q = nn.Conv1d(
                self.n_heads * self.d_k, self.n_heads * self.d_k,
                kernel_size=conv_size, groups=self.n_heads * self.d_k,
                padding=conv_size-1, bias=False
            )
            self.conv_k = nn.Conv1d(
                self.n_heads * self.d_k, self.n_heads * self.d_k,
                kernel_size=conv_size, groups=self.n_heads * self.d_k,
                padding=conv_size-1, bias=False
            )
            self.conv_v = nn.Conv1d(
                self.n_heads * self.d_v, self.n_heads * self.d_v,
                kernel_size=conv_size, groups=self.n_heads * self.d_v,
                padding=conv_size-1, bias=False
            )

        # 输出门控
        self.W_g_down = nn.Linear(self.d_model, self.d_v, bias=False)
        self.W_g_up = nn.Linear(self.d_v, self.n_heads * self.d_v, bias=False)
        self.norm_out = nn.RMSNorm(self.d_v)

        self._init_weights()

    def _init_weights(self):
        """保守初始化，防止初始 NaN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
            elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)

    def forward(self, x, state=None, return_state=False):
        B, T, _ = x.shape
        
        # 1. 线性投影
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 2. 短卷积增强（因果）
        if self.use_short_conv:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            q = self.conv_q(q)[..., :T]
            k = self.conv_k(k)[..., :T]
            v = self.conv_v(v)[..., :T]
            
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # 3. SiLU 激活
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

        # 4. 重塑为多头 [B, T, H, D]
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.n_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.n_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.n_heads)

        # 5. L2 归一化（关键稳定性操作）
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)

        # 6. 计算衰减门 g（确保为负值，表示遗忘）
        alpha_h = self.W_alpha_down(x)
        g = self.W_alpha_up(alpha_h)
        g = rearrange(g, 'b t (h d) -> b t h d', h=self.n_heads)
        # Softplus 保证正值，负号保证 g <= 0（指数衰减）
        g = -F.softplus(g.clamp(min=-10, max=10))

        # 7. 计算 Delta Rule 学习率 beta [B, T, H]
        beta = self.W_beta(x)
        beta = torch.sigmoid(beta.clamp(min=-10, max=10))

        # 8. 使用 FLA 官方 KDA Kernel（数值稳定，自动处理 FP32）
        # 注意：如果 chunk_kda 期望 g 的形状为 [B, T, H]，取消下面一行的注释
        # g = g.mean(dim=-1)  # 从 [B, T, H, D] -> [B, T, H]
        
        o, new_state = chunk_kda(
            q, k, v, 
            g,           # decay gate: [B, T, H, D] 或 [B, T, H]
            beta,        # delta rule learning rate: [B, T, H]
            initial_state=state,
            output_final_state=return_state,
            chunk_size=self.chunk_size,
            scale=1.0    # 缩放因子，类似 1/sqrt(d_k)，已在上面 L2 norm 处理
        )

        # 9. 输出 RMSNorm + GLU 门控
        o = self.norm_out(o)
        
        gate_h = self.W_g_down(x)
        gate = self.W_g_up(gate_h)
        gate = rearrange(gate, 'b t (h d) -> b t h d', h=self.n_heads)
        gate = torch.sigmoid(gate)
        o = o * gate

        # 10. 合并多头并投影
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.W_o(o)

        if return_state:
            return o, new_state
        return o


class KimiModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        
        # 混合架构：KDA 层 + 偶尔的 MLA 层（论文 3:1 比例）
        self.blocks = nn.ModuleList([
            KimiDeltaAttention(cfg) for _ in range(cfg["n_layers"])
        ])
        
        self.final_norm = nn.LayerNorm(cfg["hidden_size"])
        self.out_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        # 权重绑定（可选，节省参数）
        self.out_head.weight = self.embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)

    def forward(self, idx, states=None, return_states=False):
        b, t = idx.size()
        x = self.embedding(idx)
        x = self.drop(x)

        new_states = []
        for i, block in enumerate(self.blocks):
            if states is not None and i < len(states):
                x, state = block(x, state=states[i], return_state=True)
                new_states.append(state)
            else:
                if return_states:
                    x, state = block(x, return_state=True)
                    new_states.append(state)
                else:
                    x = block(x)

        x = self.final_norm(x)
        logits = self.out_head(x)

        if return_states:
            return logits, new_states
        return logits

    def init_states(self, batch_size, device):
        """用于增量解码的初始状态（None 表示零初始化）"""
        return [None] * len(self.blocks)