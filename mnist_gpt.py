import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GPTConfig:
    block_size: int = 786
    vocab_size: int = 320
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 320
    dropout: float = 0.1
    bias: bool = True


class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None

    def forward(self, x):
        return F.layer_norm(x, normalized_shape=self.weight.shape, weight = self.weight, bias=self.bias, eps=1e-5)
        

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias = config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size, config.block_size))
        

    def forward(self, x):

        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # q,k,v each have shape of B, T, C

        h_dim = C // self.n_head

        # B, T, n_h, n_head
        q = q.view(B, T, self.n_head, -1).transpose(1, 2) # (B, n_h, T, h_dim)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)# (B, n_h, T, h_dim)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2) # (B, n_h, T, h_dim)

        att_scores = (q @ k.transpose(2, 3)) / math.sqrt(h_dim) # (B, n_h, T, h_dim) x (B, n_h, h_dim, T) --> (B, n_h, T, T)
        att_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # att_scores = att_scores.masked_fill(self.bias[None, None, :T, :T] == 0, float('-inf'))

        att_scores = F.softmax(att_scores, dim=-1)
        att_scores = self.attn_dropout(att_scores) # (B, n_h, T, T)

        # (B, n_h, T, T) * (B, n_h, T, h_dim) --> (B, n_h, T, h_dim)
        out = att_scores @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.resid_dropout(self.c_proj(out))
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)


    def forward(self, x):
        x += self.attn(self.ln_1(x))
        x += self.mlp(self.ln_2(x))
        return x



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            drop = nn.Dropout(self.config.dropout),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f = LayerNorm(self.config)
        ))

        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight # weight tying :0


    def get_num_params(self, non_embedding=True):
        total_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total_params -= self.transformer.wte.weight.numel() # subtract embdding parameters
        return total_params 
    

    def forward(self, idx):
        device = next(self.parameters()).device
        
        if device != idx.device:
            print(f"Moving input idx for forward() to {device}")
            idx = idx.to(device)
        
        b, t = idx.size()

        pos_emb = self.transformer.wpe(torch.arange(t).to(device)) # (1, t) (get's broadcasted to (b,t))
        tok_emb = self.transformer.wte(idx)

        x = self.transformer.drop(tok_emb + pos_emb)
        for layer in self.transformer.h:
            x = layer(x)

        x = self.transformer.ln_f(x) # (B, T)
        
        logits = self.lm_head(x)[:, -1] # (B,1)
        return logits 
    

    def generate(self, target, top_k=None, return_loss_matrix=False):
        idx = torch.tensor([target + 256])[None, :] #(1, seq_len)
        device = next(self.parameters()).device
        idx = idx.to(device)

        while idx.size(1) < self.config.block_size:
            print(idx.size(1))
            
            logits = self.forward(idx) # (1, vocab_size)

            logits = torch.softmax(logits, dim=-1)

            print(logits.size())
            next_idx = torch.multinomial(logits, num_samples=1) #(1, 1)
            idx = torch.cat([idx, next_idx], dim=-1) # (1, seq_len + 1)
        
        return idx.cpu()
