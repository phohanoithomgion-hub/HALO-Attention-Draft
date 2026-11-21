import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def ssd_forward(u, dt, A, B, C, chunk_size=16):
    b, n, h = u.shape
    s = B.shape[-1]
    
    pad = (chunk_size - n % chunk_size) % chunk_size
    if pad > 0:
        u = F.pad(u, (0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, pad))
    n_p = n + pad
    nc = n_p // chunk_size
    
    u = u.view(b, nc, chunk_size, h)
    dt = dt.view(b, nc, chunk_size, h)
    B = B.view(b, nc, chunk_size, s)
    C = C.view(b, nc, chunk_size, s)
    
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=u.device))
    
    h_start = torch.zeros(b, h, s, device=u.device, dtype=u.dtype)
    y_out = torch.zeros(b, nc, chunk_size, h, device=u.device, dtype=u.dtype)
    
    for c in range(nc):
        u_c = u[:, c]
        dt_c = dt[:, c]
        B_c = B[:, c]
        C_c = C[:, c]
        
        dtA = dt_c.unsqueeze(-1) * A.view(1, 1, h, s)
        cumA = torch.cumsum(dtA, dim=1)
        
        diff = cumA.unsqueeze(2) - cumA.unsqueeze(1)
        L = torch.exp(torch.clamp(diff, max=0)) * causal_mask.view(1, chunk_size, chunk_size, 1, 1)
        
        dBu = dt_c.unsqueeze(-1) * B_c.unsqueeze(2) * u_c.unsqueeze(-1)
        h_chunk = torch.einsum('bqkhs,bkhs->bqhs', L, dBu)
        
        h_c = h_chunk + torch.exp(torch.clamp(cumA, max=0)) * h_start.unsqueeze(1)
        y_out[:, c] = (h_c * C_c.unsqueeze(2)).sum(-1)
        h_start = h_c[:, -1]
        
    return y_out.view(b, n_p, h)[:, :n]


def sliding_window_attention(q, k, v, window, attention_mask=None):
    b, h, n, d = q.shape
    out = torch.zeros_like(q)

    for start in range(0, n, window):
        end = min(start + window, n)
        k_start = max(0, start - window + 1)

        q_blk = q[:, :, start:end]
        k_blk = k[:, :, k_start:end]
        v_blk = v[:, :, k_start:end]

        scores = torch.einsum('bhid,bhjd->bhij', q_blk, k_blk) / math.sqrt(d)

        q_idx = torch.arange(start, end, device=q.device).unsqueeze(1)
        k_idx = torch.arange(k_start, end, device=q.device).unsqueeze(0)
        mask = (k_idx <= q_idx) & (k_idx > q_idx - window)
        
        if attention_mask is not None:
            mask_k = attention_mask[:, k_start:end].unsqueeze(1).unsqueeze(2)
            mask = mask & mask_k

        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out[:, :, start:end] = torch.einsum('bhij,bhjd->bhid', attn, v_blk)

    return out


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))

        A_init = torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A_init).unsqueeze(0).expand(self.d_inner, -1).contiguous())
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u, attention_mask=None):
        b, n, _ = u.shape

        xzBCdt = self.in_proj(u)
        x, z, B, C, dt_raw = torch.split(
            xzBCdt,
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.d_inner],
            dim=-1,
        )

        x = self.conv1d(x.transpose(1, 2))[:, :, :n].transpose(1, 2)
        x = F.silu(x)

        dt = F.softplus(self.dt_proj(dt_raw) + self.dt_bias)
        
        if attention_mask is not None:
            dt = dt * attention_mask.unsqueeze(-1).float()
            
        A = -torch.exp(self.A_log)

        y = ssd_forward(x, dt, A, B, C)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        y = y * F.silu(z)

        out = self.out_proj(y)
        return out, dt


class MultiScalePooler(nn.Module):
    def __init__(self, d_model, out_dim):
        super().__init__()
        self.score_proj = nn.Linear(d_model, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 7, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, mask):
        b, n, d = x.shape
        imp = self.score_proj(x).squeeze(-1)
        if mask is not None:
            imp = imp.masked_fill(~mask, -1e9)
            
        _, idx = torch.sort(imp, dim=-1, descending=True)
        
        sizes = [256, 128, 64, 32, 16, 8, 4]
        pooled = []
        for s in sizes:
            k = min(s, n)
            top_idx = idx[:, :k]
            x_top = torch.gather(x, 1, top_idx.unsqueeze(-1).expand(-1, -1, d))
            
            if mask is not None:
                m_top = torch.gather(mask, 1, top_idx).unsqueeze(-1)
                x_top = x_top * m_top
                denom = m_top.sum(dim=1).clamp(min=1)
                p = x_top.sum(dim=1) / denom
            else:
                p = x_top.mean(dim=1)
            pooled.append(p)
            
        return self.mlp(torch.cat(pooled, dim=-1))


class HyperLoRAGenerator(nn.Module):
    def __init__(self, d_model, n_blocks, rank=4, n_sinks=8, hidden=128):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.rank = rank
        self.n_sinks = n_sinks

        self.head_A_size = 3 * d_model * rank
        self.head_B_size = 3 * d_model * rank + 2 * n_sinks * d_model

        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.block_emb = nn.Parameter(torch.zeros(n_blocks, hidden))
        
        self.head_A = nn.Linear(hidden, self.head_A_size)
        self.head_B = nn.Linear(hidden, self.head_B_size)

        nn.init.kaiming_uniform_(self.head_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.head_A.bias)
        
        nn.init.zeros_(self.head_B.weight)
        nn.init.zeros_(self.head_B.bias)

    def forward(self, g, block_idx):
        b = g.shape[0]
        h = self.trunk(g) + self.block_emb[block_idx].unsqueeze(0)
        
        out_A = self.head_A(h)
        out_B = self.head_B(h)

        d = self.d_model
        r = self.rank
        ns = self.n_sinks

        A_K = out_A[:, :d * r].view(b, r, d)
        A_V = out_A[:, d * r:2 * d * r].view(b, r, d)
        A_O = out_A[:, 2 * d * r:3 * d * r].view(b, r, d)

        B_K = out_B[:, :d * r].view(b, d, r)
        B_V = out_B[:, d * r:2 * d * r].view(b, d, r)
        B_O = out_B[:, 2 * d * r:3 * d * r].view(b, d, r)
        
        dZ_K = out_B[:, 3 * d * r:3 * d * r + ns * d].view(b, ns, d)
        dZ_V = out_B[:, 3 * d * r + ns * d:].view(b, ns, d)

        return {
            'A_K': A_K, 'B_K': B_K,
            'A_V': A_V, 'B_V': B_V,
            'A_O': A_O, 'B_O': B_O,
            'dZ_K': dZ_K, 'dZ_V': dZ_V,
        }


class HALOLayer(nn.Module):
    def __init__(self, d_model, n_heads, window, n_sinks, d_state, expand, lora_rank, chunk_size=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.n_sinks = n_sinks
        self.chunk_size = chunk_size
        self.block_size = chunk_size

        self.mamba = Mamba2Block(d_model, d_state, expand=expand)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.scale_K = nn.Parameter(torch.ones(1))
        self.scale_V = nn.Parameter(torch.ones(1))
        self.scale_O = nn.Parameter(torch.ones(1))

        self.imp_proj = nn.Linear(d_model, 1, bias=False)
        self.imp_lambda = nn.Parameter(torch.zeros(1))

        self.sink_K = nn.Parameter(torch.randn(n_sinks, d_model) * 0.02)
        self.sink_V = nn.Parameter(torch.randn(n_sinks, d_model) * 0.02)

        self.qc_A = nn.Linear(d_model, lora_rank, bias=False)
        self.qc_B = nn.Linear(lora_rank, d_model, bias=False)

        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.ffn = SwiGLU(d_model, d_model * 2)

    def forward(self, x, halora_list, attention_mask=None):
        b, n, d = x.shape
        n_chunks = len(halora_list)

        x_norm = self.norm1(x)
        H, dt = self.mamba(x_norm, attention_mask)
        x = x + H

        x_norm = self.norm2(x)

        q_flat = self.q_proj(x_norm)
        k_flat = self.k_proj(x_norm)
        v_flat = self.v_proj(x_norm)

        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min((c+1) * self.chunk_size, n)
            h_c = halora_list[c]
            x_c = x_norm[:, start:end]
            
            delta_k = torch.einsum('bnr,bdr->bnd', torch.einsum('bnd,brd->bnr', x_c, h_c['A_K']), h_c['B_K']) * self.scale_K
            delta_v = torch.einsum('bnr,bdr->bnd', torch.einsum('bnd,brd->bnr', x_c, h_c['A_V']), h_c['B_V']) * self.scale_V
            
            k_flat[:, start:end] = k_flat[:, start:end] + delta_k
            v_flat[:, start:end] = v_flat[:, start:end] + delta_v

        q = q_flat.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_flat.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v_flat.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        A_local = sliding_window_attention(q, k, v, self.window, attention_mask)

        imp = torch.sigmoid(self.imp_proj(H) + self.imp_lambda * dt.mean(-1, keepdim=True)).squeeze(-1)
        if attention_mask is not None:
            imp = imp.masked_fill(~attention_mask, -1e9)

        q_corrected_flat = q_flat + self.qc_B(F.silu(self.qc_A(q_flat)))
        q_corrected = q_corrected_flat.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        A_long_list = []
        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min((c+1) * self.chunk_size, n)
            q_chunk = q_corrected[:, :, start:end]
            h_c = halora_list[c]
            
            sink_K = (self.sink_K.unsqueeze(0) + h_c['dZ_K']).view(b, self.n_sinks, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            sink_V = (self.sink_V.unsqueeze(0) + h_c['dZ_V']).view(b, self.n_sinks, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

            if start == 0:
                K_long = sink_K.expand(b, self.n_heads, end - start, self.n_sinks, self.head_dim)
                V_long = sink_V.expand(b, self.n_heads, end - start, self.n_sinks, self.head_dim)
            else:
                n_blk = start // self.block_size
                imp_past = imp[:, :start].view(b, n_blk, self.block_size)
                
                mask_past = None
                if attention_mask is not None:
                    mask_past = attention_mask[:, :start].view(b, n_blk, self.block_size)
                    imp_past = imp_past.masked_fill(~mask_past, -1e9)
                    
                blk_max = imp_past.max(dim=-1)[0]
                
                if mask_past is not None:
                    blk_sum = imp_past.masked_fill(~mask_past, 0).sum(dim=-1)
                    blk_count = mask_past.sum(dim=-1).clamp(min=1)
                    blk_mean = blk_sum / blk_count
                else:
                    blk_mean = imp_past.mean(dim=-1)
                    
                nb = min(8, n_blk)
                _, top_max = torch.topk(blk_max, nb, dim=-1)
                _, top_mean = torch.topk(blk_mean, nb, dim=-1)
                
                sel_mask = torch.zeros(b, n_blk, device=x.device, dtype=torch.bool)
                sel_mask.scatter_(1, top_max, True)
                sel_mask.scatter_(1, top_mean, True)
                
                sel_score = sel_mask.float() * blk_max + (~sel_mask).float() * -1e9
                n_sel = min(16, n_blk)
                sel_val, sel_idx = torch.topk(sel_score, n_sel, dim=-1)
                
                if n_sel < 16:
                    pad_idx = torch.zeros(b, 16 - n_sel, dtype=torch.long, device=x.device)
                    sel_idx = torch.cat([sel_idx, pad_idx], dim=-1)
                    pad_val = torch.full((b, 16 - n_sel), -1e9, device=x.device)
                    sel_val = torch.cat([sel_val, pad_val], dim=-1)
                    
                tok_idx = sel_idx.unsqueeze(-1) * self.block_size + torch.arange(self.block_size, device=x.device)
                tok_idx = tok_idx.view(b, -1)
                
                valid_tok = (sel_val.unsqueeze(-1) > -1e8) & (tok_idx < start)
                if attention_mask is not None:
                    m_tok = torch.gather(attention_mask, 1, tok_idx.clamp(max=n-1))
                    valid_tok = valid_tok & m_tok
                    
                idx_exp = tok_idx.unsqueeze(1).unsqueeze(-1).expand(b, self.n_heads, -1, self.head_dim)
                k_sel = torch.gather(k, 2, idx_exp)
                v_sel = torch.gather(v, 2, idx_exp)
                
                sink_K_exp = sink_K.expand(b, self.n_heads, end - start, self.n_sinks, self.head_dim)
                sink_V_exp = sink_V.expand(b, self.n_heads, end - start, self.n_sinks, self.head_dim)
                
                K_long = torch.cat([k_sel, sink_K_exp], dim=2)
                V_long = torch.cat([v_sel, sink_V_exp], dim=2)
                
                mask_long = torch.cat([
                    valid_tok.unsqueeze(1).expand(b, self.n_heads, -1),
                    torch.ones(b, self.n_heads, self.n_sinks, dtype=torch.bool, device=x.device)
                ], dim=-1)
                
                scores_long = torch.einsum('bhqd,bhkd->bhqk', q_chunk, K_long) / math.sqrt(self.head_dim)
                scores_long = scores_long.masked_fill(~mask_long.unsqueeze(2), -1e9)
                
                attn_long = F.softmax(scores_long, dim=-1)
                attn_long = torch.nan_to_num(attn_long, nan=0.0)
                A_long_chunk = torch.einsum('bhqk,bhkd->bhqd', attn_long, V_long)
                A_long_list.append(A_long_chunk)
                continue

            scores_long = torch.einsum('bhqd,bhkd->bhqk', q_chunk, K_long) / math.sqrt(self.head_dim)
            attn_long = F.softmax(scores_long, dim=-1)
            A_long_chunk = torch.einsum('bhqk,bhkd->bhqd', attn_long, V_long)
            A_long_list.append(A_long_chunk)
            
        A_long = torch.cat(A_long_list, dim=2)

        A_local_flat = A_local.transpose(1, 2).reshape(b, n, d)
        A_long_flat = A_long.transpose(1, 2).reshape(b, n, d)

        gate = torch.sigmoid(self.gate(x_norm))
        A_flat = gate * A_local_flat + (1 - gate) * A_long_flat

        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min((c+1) * self.chunk_size, n)
            h_c = halora_list[c]
            A_c = A_flat[:, start:end]
            delta_o = torch.einsum('bnr,bdr->bnd', torch.einsum('bnd,brd->bnr', A_c, h_c['A_O']), h_c['B_O']) * self.scale_O
            A_flat[:, start:end] = A_flat[:, start:end] + delta_o

        A_out = self.o_proj(A_flat)
        x = x + A_out
        x = x + self.ffn(self.norm3(x))
        return x


class HALOModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, window=256, n_sinks=8, d_state=128, expand=2, lora_rank=4, group_size=4, chunk_size=128):
        super().__init__()
        assert n_layers % group_size == 0
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_blocks = n_layers // group_size
        self.group_size = group_size
        self.chunk_size = chunk_size

        self.embed = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            HALOLayer(d_model, n_heads, window, n_sinks, d_state, expand, lora_rank, chunk_size)
            for _ in range(n_layers)
        ])

        self.hyperlora = HyperLoRAGenerator(d_model, self.n_blocks, rank=lora_rank, n_sinks=n_sinks)
        self.pooler = MultiScalePooler(d_model, d_model)
        self.pool_norm = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            
        x = self.embed(input_ids)
        b, n, d = x.shape
        n_chunks = (n + self.chunk_size - 1) // self.chunk_size

        for block_idx in range(self.n_blocks):
            g_list = []
            for c in range(n_chunks):
                start = c * self.chunk_size
                if start == 0:
                    g_c = self.pool_norm(x[:, 0])
                else:
                    past_end = start
                    past_start = max(0, past_end - 256)
                    x_past = x[:, past_start:past_end]
                    m_past = attention_mask[:, past_start:past_end] if attention_mask is not None else None
                    g_c = self.pool_norm(self.pooler(x_past, m_past))
                g_list.append(g_c)
                
            halora_list = [self.hyperlora(g_c, block_idx) for g_c in g_list]
            
            for layer_in_block in range(self.group_size):
                layer_idx = block_idx * self.group_size + layer_in_block
                x = self.layers[layer_idx](x, halora_list, attention_mask)

        x = self.final_norm(x)
        return self.lm_head(x)
