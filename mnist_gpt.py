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
        att_scores = att_scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

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
    

    def generate(self, targets, temperature = 1.0, top_k=None, top_p=None):
        idx = torch.tensor([[target + 256] for target in targets])
        device = next(self.parameters()).device

        idx = idx.to(device)

        while idx.size(1) < self.config.block_size:

            with torch.no_grad():
                logits = self.forward(idx) # (1, vocab_size)
                
                # apply temperature scaling
                assert temperature >= 0, "temperature must be non-negative value"
                
                if temperature > 0:
                    logits /= temperature
                
                elif temperature == 0:
                    max_values, max_indices = torch.max(logits, dim=-1)
                    
                    new_logits = torch.ones_like(logits) * float('-inf')
                    new_logits[torch.arange(logits.size(0)), max_indices] = max_values
                    logits = new_logits
                
                
                # apply sampling strategy (if specified)
                probs = torch.softmax(logits, dim=-1) # (1, vocab_size)
                assert not (top_k is not None and top_p is not None), "You must choose either top_p or top_k sampling (or neither)"
                
                if top_k is not None:
                    top_k_values, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
                    threshold = top_k_values[:, -1] # (B,)
                    probs[torch.where(probs < threshold[:, None])] = 0 # mask out the logits that aren't contributing
                    
                    # renormalize probabilities
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_idx = torch.multinomial(probs, num_samples=1)


                elif top_p is not None:
                    # sort the probabilities (in descending order!!!)
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    
                    # find the cumsum
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum <= top_p
                    mask[:, 0] = True # doing this to make sure we have at least one token
                    
                    sorted_probs[~mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                    next_intermediate_idx = torch.multinomial(sorted_probs, num_samples=1) # (B, 1)
                    next_idx = torch.gather(sorted_indices, 1, next_intermediate_idx)
                    
                
                else:
                    next_idx = torch.multinomial(probs, num_samples=1)#(1, 1) (just regular random sampling)
                

                idx = torch.cat([idx, next_idx], dim=-1) # (1, seq_len + 1)
                next_idx = next_idx.view(-1)

        return idx.cpu()