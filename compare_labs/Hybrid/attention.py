import torch
import torch.nn as nn
from kda_stable import KimiDeltaAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 兼容KDA和GPT的配置命名
        self.num_heads = cfg.get("num_heads", cfg.get("n_heads", 12))
        self.head_dim = cfg["hidden_size"] // self.num_heads
        self.hidden_size = cfg["hidden_size"]
        
        # 兼容两种配置格式
        qkv_bias = cfg.get("qkv_bias", False)
        drop_rate = cfg.get("drop_rate", 0.1)
        context_size = cfg.get("context_size", cfg.get("max_position_embeddings", 1024))
        
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size, bias=qkv_bias)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size, bias=qkv_bias)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size, bias=qkv_bias)
        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=qkv_bias)
        self.dropout = nn.Dropout(drop_rate)
        
        # 注册mask，支持动态序列长度
        self.register_buffer("mask", 
                torch.triu(torch.ones(context_size, context_size), diagonal=1).bool())
    
    def forward(self, x, state=None, return_state=False):
        """
        兼容KDA的接口：接受state和return_state参数
        state: 传统Attention不需要状态，但为了兼容接口保留参数
        """
        b, t, c = x.size()
        
        # 线性投影
        q = self.w_q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attn_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_scores = attn_scores.masked_fill(self.mask[:t, :t], float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 合并头
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)
        output = self.w_o(attn_output)
        
        # 兼容KDA的返回格式：如果要求返回状态，返回None作为状态
        if return_state:
            return output, None
        return output
    

class HybridKimiModel(nn.Module):
    """
    混合架构：每4层为一个周期，包含3层KDA + 1层Attention
    8层模型结构：[KDA, KDA, KDA, Attn, KDA, KDA, KDA, Attn]
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        
        self.blocks = nn.ModuleList()
        n_layers = cfg["n_layers"]
        
        # 构建混合层：每第4层使用Attention，其余使用KDA
        for i in range(n_layers):
            if (i + 1) % 4 == 0:  # 第4, 8, 12...层使用传统Attention
                print(f"Layer {i}: MultiHeadAttention")
                self.blocks.append(MultiHeadAttention(cfg))
            else:
                print(f"Layer {i}: KimiDeltaAttention")
                self.blocks.append(KimiDeltaAttention(cfg))
        
        self.final_norm = nn.LayerNorm(cfg["hidden_size"])
        self.out_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        
        # 权重绑定
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
                # 注意：传统Attention返回的state是None，但KDA返回真实状态
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
        """混合初始化：KDA层需要初始化，Attention层返回None"""
        return [None] * len(self.blocks)