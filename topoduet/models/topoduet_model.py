import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange



class RevIN(nn.Module):
    """Reversible Instance Normalization"""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.weight + self.bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.bias) / self.weight
            return x * self.stdev + self.mean
        else:
            raise ValueError(f"Unsupported mode: {mode}")


class RTDSmoother(nn.Module):

    def __init__(self, num_chem_vars, kernel_size=5):
        super().__init__()
        self.padding = kernel_size - 1
        self.num_chem_vars = num_chem_vars
        self.raw_weights = nn.Parameter(torch.zeros(num_chem_vars, 1, kernel_size))

    def forward(self, chem_seq):
        x = chem_seq.transpose(1, 2)  # (B, C_chem, L)
        x = F.pad(x, (self.padding, 0), mode='replicate')  # 因果Padding，阻断未来信息
        w = F.softmax(self.raw_weights, dim=2)
        x = F.conv1d(x, weight=w, groups=self.num_chem_vars)
        return x.transpose(1, 2)  # (B, L, C_chem)


class TrendSmoother(nn.Module):
   

    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1

    def forward(self, x):
        x_t = x.transpose(1, 2)  # (B, N, L)
        x_t = F.pad(x_t, (self.padding, 0), mode='replicate')
        return F.avg_pool1d(x_t, kernel_size=self.kernel_size, stride=1).transpose(1, 2)



class PatchEmbedding(nn.Module):
 

    def __init__(self, seq_len, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model) * 0.02)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B, N, P, patch_len)
        return self.proj(x) + self.pos_enc  # (B, N, P, d_model)


class TemporalAttention(nn.Module):
  

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B_N, P, _ = x.shape
        qkv = self.qkv(x).reshape(B_N, P, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        scores = torch.matmul(qkv[0], qkv[1].transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, qkv[2]).transpose(1, 2).reshape(B_N, P, self.d_model)
        return self.out(out)


class SpatialAttention(nn.Module):
 

    def __init__(self, d_model, n_heads, dropout=0.1, topk=3):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.topk = topk
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spatial_mask=None):
        B_P, N, _ = x.shape
        d_model = self.n_heads * self.d_k

        
        qkv = self.qkv(x).reshape(B_P, N, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        scores = torch.matmul(qkv[0], qkv[1].transpose(-2, -1)) / math.sqrt(self.d_k)  # (B_P, n_heads, N, N)


        if spatial_mask is not None:
            scores = scores + spatial_mask.unsqueeze(0).unsqueeze(0)

       
        if self.topk is not None and self.topk < N:
            topk_values, topk_indices = torch.topk(scores, k=self.topk, dim=-1)
            sparse_scores_mask = torch.full_like(scores, -1e9)
           
            sparse_scores_mask.scatter_(dim=-1, index=topk_indices, src=topk_values)
            scores = sparse_scores_mask

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, qkv[2]).transpose(1, 2).reshape(B_P, N, d_model)
        return self.out(out)


class STEncoderLayer(nn.Module):
   

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, spatial_mode="masked", topk=3):
        super().__init__()
        self.spatial_mode = spatial_mode
        self.temp_attn = TemporalAttention(d_model, n_heads, dropout)

       
        if self.spatial_mode != "none":
            self.spat_attn = SpatialAttention(d_model, n_heads, dropout, topk=topk)
            self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, spatial_mask=None):
        B, N, P, D = x.shape

        z = x.reshape(B * N, P, D)
        z = z + self.temp_attn(self.norm1(z))
        x = z.reshape(B, N, P, D)

   
        if self.spatial_mode != "none":
            z = x.transpose(1, 2).reshape(B * P, N, D)
            mask = None if self.spatial_mode == "coupled" else spatial_mask
            z = z + self.spat_attn(self.norm3(z), spatial_mask=mask)
            x = z.reshape(B, P, N, D).transpose(1, 2)

        return x + self.ffn(self.norm2(x))



class GroupRTDTracker(nn.Module):


    def __init__(self, config, n_vars=39, d_model=128, Q_mean=1452.0, min_flow=800.0, beta_floor=0.15):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.min_flow = float(min_flow)
        self.max_flow = float(getattr(config, "max_flow", 3000.0))
        self.flow_valid_min = float(getattr(config, "flow_valid_min", 100.0))
        self.short_flow_window = int(
            getattr(config, "short_flow_window", getattr(config, "flow_window", 24))
        )
        if self.short_flow_window < 1:
            raise ValueError("short_flow_window must be at least 1")
        if self.max_flow < self.min_flow:
            raise ValueError("max_flow must be greater than or equal to min_flow")

        
        self.tracker_window = int(getattr(config, "tracker_window", 16))
        if self.tracker_window < 1:
            raise ValueError("tracker_window must be at least 1")

        self.beta_floor = float(beta_floor)
        self.flow_inverse_standardized = bool(
            getattr(config, "flow_inverse_standardized", False)
        )
        self.register_buffer(
            "flow_scaler_mean",
            torch.tensor(float(getattr(config, "flow_scaler_mean", 0.0))),
        )
        self.register_buffer(
            "flow_scaler_scale",
            torch.tensor(float(getattr(config, "flow_scaler_scale", 1.0))),
        )
        self.export_tracker_debug = bool(getattr(config, "export_tracker_debug", False))
        self.print_tracker_debug = bool(getattr(config, "print_tracker_debug", False))
        self._debug_printed = False
        self.register_buffer('Q_mean', torch.tensor(float(Q_mean)))

        
        self.group_indices = {
            0: getattr(config, "raw_indices", [6, 7, 8]),
            1: getattr(config, "chem_indices", [36, 37, 38]),
            2: getattr(config, "yiqi_indices", [0, 1] + list(range(9, 21))),
            3: getattr(config, "erqi_indices", [2] + list(range(21, 34))),
            4: getattr(config, "out_indices", [3, 4, 5, 34, 35])
        }
        self.G = len(self.group_indices)

        pool_matrix = torch.zeros(n_vars, self.G)
        for g_idx, vars_in_group in self.group_indices.items():
            for v in vars_in_group:
                if v < n_vars:
                    pool_matrix[v, g_idx] = 1.0
        pool_norm = pool_matrix / (pool_matrix.sum(dim=0, keepdim=True) + 1e-8)
        self.register_buffer('pool_matrix', pool_norm)
        self.register_buffer('scatter_matrix', pool_matrix.t())

       
        self.register_buffer('group_penalty', self._build_group_topology())

     
        group_has_incoming = (self.group_penalty == 0).any(dim=1).float()
        variable_has_incoming = torch.einsum(
            'g,gn->n', group_has_incoming, self.scatter_matrix
        ).clamp(max=1.0)
        self.register_buffer('variable_has_incoming', variable_has_incoming)

        
        tau_ref = torch.ones(self.G, self.G)
        tau_ref[2, 0] = float(getattr(config, "tau_raw_to_yiqi", 1.0))
        tau_ref[3, 0] = float(getattr(config, "tau_raw_to_erqi", 1.0))
        tau_ref[2, 1] = float(getattr(config, "tau_chem_to_yiqi", 1.0))
        tau_ref[3, 1] = float(getattr(config, "tau_chem_to_erqi", 1.0))
        tau_ref[4, 2] = float(getattr(config, "tau_yiqi_to_out", 3.0))
        tau_ref[4, 3] = float(getattr(config, "tau_erqi_to_out", 3.0))
        tau_ref[4, 0] = float(getattr(config, "tau_raw_to_out", 4.0))
        tau_ref[4, 1] = float(getattr(config, "tau_chem_to_out", 4.0))
        self.learned_V_matrix = nn.Parameter(torch.log(torch.expm1(tau_ref.clamp_min(1e-4))))

       
        _sigma_target = float(getattr(config, "sigma_init", 1.0))
        self.learned_sigma_matrix = nn.Parameter(
            torch.log(torch.expm1(torch.ones(self.G, self.G) * _sigma_target))
        )
   

        self.var_proj = nn.Linear(1, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        
        self.time_bias_scale = nn.Parameter(
            torch.tensor(float(getattr(config, "time_bias_scale_init", 4.0)))
        )

        self.transport_adapter = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )

    def _build_group_topology(self):
        penalty = torch.full((self.G, self.G), -1e9)
     
        edges = [
            (0, 2), (1, 2),
            (0, 3), (1, 3),
            (2, 4), (3, 4),
      
            (0, 4), (1, 4),
        ]
        for src, dst in edges:
            penalty[dst, src] = 0.0
        return penalty

    def _to_physical_flow(self, flow_seq):
    
        if self.flow_inverse_standardized:
            return flow_seq * self.flow_scaler_scale + self.flow_scaler_mean
        return flow_seq

    def _estimate_recent_flow(self, raw_flow_seq):
      
        physical_flow_seq = self._to_physical_flow(raw_flow_seq)
        window = min(self.short_flow_window, physical_flow_seq.size(1))
        recent = physical_flow_seq[:, -window:]
        valid = torch.isfinite(recent) & (recent >= self.flow_valid_min)
        count = valid.sum(dim=1)
        total = torch.where(valid, recent, torch.zeros_like(recent)).sum(dim=1)
        estimated_flow = total / count.clamp_min(1).to(total.dtype)
        estimated_flow = torch.where(
            count > 0,
            estimated_flow,
            self.Q_mean.expand_as(estimated_flow),
        )
        estimated_flow = estimated_flow.clamp(min=self.min_flow, max=self.max_flow)
        return physical_flow_seq, estimated_flow

    @staticmethod
    def _finite_min_max_mean(tensor):
        values = tensor.detach()
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return float("nan"), float("nan"), float("nan")
        return tuple(float(v.cpu()) for v in (values.min(), values.max(), values.mean()))

    def forward(self, x_norm, raw_flow_seq, tracker_mode="dynamic"):
        B, L_full, N = x_norm.shape

        if tracker_mode not in {"dynamic", "fixed"}:
            raise ValueError(
                f"Unsupported tracker_mode={tracker_mode!r}; expected 'dynamic' or 'fixed'."
            )

       
        W = min(self.tracker_window, L_full)
        x_norm = x_norm[:, -W:, :]
        L = W

        
        physical_flow_seq, estimated_flow = self._estimate_recent_flow(raw_flow_seq)
        if tracker_mode == "dynamic":
            actual_flow_vec = estimated_flow
        else:
            actual_flow_vec = self.Q_mean.expand(B)
        actual_flow = actual_flow_vec.view(B, 1, 1)

        V_matrix = F.softplus(self.learned_V_matrix)
        tau_matrix = V_matrix.unsqueeze(0) * (self.Q_mean / actual_flow)
        sigma_matrix = F.softplus(self.learned_sigma_matrix) + 1e-3

        t_idx = torch.arange(L, device=x_norm.device).view(1, 1, 1, L)
        dist = (L - 1) - t_idx

        
        gaussian_mask = torch.exp(-(dist - tau_matrix.unsqueeze(-1)) ** 2 / (2 * sigma_matrix.unsqueeze(-1) ** 2))
        soft_gaussian = self.beta_floor + (1 - self.beta_floor) * gaussian_mask
        time_mask = torch.log(soft_gaussian + 1e-9)

        strict_history_penalty = torch.where(dist == 0, torch.tensor(-1e9, device=x_norm.device),
                                             torch.tensor(0.0, device=x_norm.device))

        x_emb = self.var_proj(x_norm.unsqueeze(-1))
        group_feats = torch.einsum('blnd,ng->blgd', x_emb, self.pool_matrix)

        q = self.W_q(group_feats[:, -1, :, :])
        k = self.W_k(group_feats)
        v = self.W_v(group_feats)

        scores = torch.einsum('bgd, bljd -> bgjl', q, k) / math.sqrt(self.d_model)

        
        time_bias = F.softplus(self.time_bias_scale) * time_mask
        logits = scores + time_bias + strict_history_penalty
        

        
        valid_edges = (self.group_penalty == 0).view(1, self.G, self.G, 1)
        masked_logits = logits.masked_fill(~valid_edges, -1e9)
        attn_weights = torch.softmax(masked_logits.view(B, self.G, -1), dim=-1)
        attn_weights = attn_weights.view(B, self.G, self.G, L)
        attn_weights = attn_weights * valid_edges.to(attn_weights.dtype)
        normalizer = attn_weights.sum(dim=(2, 3), keepdim=True)
        attn_weights = attn_weights / normalizer.clamp_min(1e-12)

        hrt_group_feat = torch.einsum('bgjl, bljd -> bgd', attn_weights, v)
        hrt_var_broadcast = torch.einsum('bgd,gn->bnd', hrt_group_feat, self.scatter_matrix)
        tracked_feat = self.transport_adapter(hrt_var_broadcast)

        
        tracked_feat = tracked_feat * self.variable_has_incoming.view(1, N, 1)

        if self.print_tracker_debug and not self._debug_printed:
            raw_stats = self._finite_min_max_mean(raw_flow_seq)
            physical_stats = self._finite_min_max_mean(physical_flow_seq)
            estimated_stats = self._finite_min_max_mean(estimated_flow)
            actual_stats = self._finite_min_max_mean(actual_flow_vec)
            tau_stats = self._finite_min_max_mean(tau_matrix)
            raw_out_stats = self._finite_min_max_mean(tau_matrix[:, 4, 0])
            chem_out_stats = self._finite_min_max_mean(tau_matrix[:, 4, 1])
            print("\n[GroupRTDTracker Debug]")
            print("tracker_mode =", tracker_mode)
            print("tracker_window(W) =", W, "/ seq_len =", L_full)
            print("time_bias_scale(softplus) =", float(F.softplus(self.time_bias_scale).detach().cpu()))
            print("sigma_matrix(min/max/mean) =", self._finite_min_max_mean(sigma_matrix))
            print("flow_inverse_standardized =", self.flow_inverse_standardized)
            print("flow_scaler_mean =", float(self.flow_scaler_mean.detach().cpu()))
            print("flow_scaler_scale =", float(self.flow_scaler_scale.detach().cpu()))
            print("Q_mean =", float(self.Q_mean.detach().cpu()))
            print("raw_flow_seq min/max/mean =", raw_stats)
            print("physical_flow_seq min/max/mean =", physical_stats)
            print("estimated_flow min/max/mean =", estimated_stats)
            print("actual_flow min/max/mean =", actual_stats)
            print("tau_matrix min/max/mean =", tau_stats)
            print("tau_raw_to_out min/max/mean =", raw_out_stats)
            print("tau_chem_to_out min/max/mean =", chem_out_stats)
            print("[End GroupRTDTracker Debug]\n")
            self._debug_printed = True

        if self.export_tracker_debug:
            self.tracker_debug = {
                "tracker_mode": tracker_mode,
                "tracker_window": W,
                "time_bias_scale": float(F.softplus(self.time_bias_scale).detach().cpu()),
                "raw_flow_seq": raw_flow_seq.detach().cpu(),
                "physical_flow_seq": physical_flow_seq.detach().cpu(),
                "estimated_flow": estimated_flow.detach().cpu(),
                "actual_flow": actual_flow_vec.detach().cpu(),
                "valid_edges": valid_edges.detach().cpu(),
                "attention": attn_weights.detach().cpu(),
                "tau_matrix": tau_matrix.detach().cpu(),
                "sigma_matrix": sigma_matrix.detach().cpu(),
                "tracker_output": tracked_feat.detach().cpu(),
            }
        return tracked_feat




def resolve_ablation(raw_mode, topk=3, force_topk=None):
    """消融解析 v2：以旧 no_topk 配置为正式基线。

    基线 full = 物理掩码空间注意力（可行域内全连接，无 Top-K）
              + 动态 Tracker + Trend + Shortcut + 药剂平滑 + Patch 主干

    与 v1 的区别：除 with_topk / global_topk 外，所有模式一律关闭 Top-K，
    保证每个消融相对基线只改变一个因素（消除 v1 中组件消融驮着 Top-K 的混杂）。

    模式一览（相对基线的单一变化）：
      full / no_topk(旧名兼容)  : 基线本身
      with_topk                 : + Top-K 稀疏路由（= v1 的旧完整模型，用于记录其负贡献）
      dense_spatial             : - 物理掩码（可行域约束移除，全通道稠密注意力）
      no_spatial                : - 整个空间注意力
      global_topk               : - 物理掩码 且 + Top-K（双因素，仅为兼容 v1 结果保留）
      no_tracker                : - 拉格朗日 Tracker
      fixed_hrt / mean_flow     : Tracker 滞后固定为 Q_mean 工况（去流量条件化）
      no_trend / no_smooth / no_shortcut : 去对应 Eulerian 组件
      no_patch_backbone         : - Patch 主干（保留 trend/tracker/shortcut）
      shortcut_only             : 仅保留线性 shortcut

    force_topk=True/False 可强制覆盖 Top-K 开关（例如复现 v1 的
    no_tracker+TopK 组合：ablation_mode='no_tracker', force_topk=True）。
    """
    raw_mode = str(raw_mode).lower()
    if raw_mode == "no_topk":  
        raw_mode = "full"

    dynamic_tracker_modes = {"full", "dynamic", "reachability_dynamic", "with_topk"}
    fixed_tracker_modes = {"fixed_hrt", "mean_flow"}
    eulerian_ablation_modes = {
        "no_trend", "no_shortcut", "no_patch_backbone", "dense_spatial",
        "global_topk", "no_spatial", "no_smooth", "shortcut_only", "with_topk",
    }
    supported = (dynamic_tracker_modes | fixed_tracker_modes
                 | {"no_tracker"} | eulerian_ablation_modes)
    if raw_mode not in supported:
        raise ValueError(
            f"Unsupported ablation_mode={raw_mode!r}. 基线为 'full'（=旧 no_topk）；"
            "Tracker 消融: 'fixed_hrt' / 'no_tracker'；Eulerian 消融: "
            f"{sorted(eulerian_ablation_modes)}."
        )

    # Tracker 
    if raw_mode in fixed_tracker_modes:
        use_tracker, tracker_mode = True, "fixed"
    elif raw_mode in {"no_tracker", "shortcut_only"}:
        use_tracker, tracker_mode = False, "none"
    else:
        use_tracker, tracker_mode = True, "dynamic"

    # Eulerian
    use_trend = raw_mode not in ["no_trend", "shortcut_only"]
    use_shortcut = raw_mode not in ["no_shortcut"]
    use_backbone = raw_mode not in ["no_patch_backbone", "shortcut_only"]

    if raw_mode in ["no_spatial", "shortcut_only", "no_patch_backbone"]:
        spatial_mode = "none"
    elif raw_mode in ["dense_spatial", "global_topk"]:
        spatial_mode = "coupled"
    else:
        spatial_mode = "masked"

    topk_on_modes = {"with_topk", "global_topk"}
    if force_topk is not None:
        effective_topk = topk if bool(force_topk) else None
    else:
        effective_topk = topk if raw_mode in topk_on_modes else None

    return {
        "mode": raw_mode, "use_tracker": use_tracker, "tracker_mode": tracker_mode,
        "use_trend": use_trend, "use_shortcut": use_shortcut,
        "use_backbone": use_backbone, "spatial_mode": spatial_mode,
        "effective_topk": effective_topk,
    }


class TopoDUETModel(nn.Module):
    """ELFormer 主模型：解耦工艺组内瞬时共态与跨工艺组异步拉格朗日追踪"""

    def __init__(self, config):
        super(TopoDUETModel, self).__init__()
        self.n_vars = config.enc_in
        self.seq_len = config.seq_len
        
        self.pred_len = getattr(config, "pred_len", None)
        if self.pred_len is None:
            self.pred_len = getattr(config, "horizon", None)
        if self.pred_len is None:
            raise ValueError("TopoDUETModel requires pred_len (or legacy horizon).")
        self.d_model = config.d_model

        self.flow_idx = getattr(config, "flow_idx", 6)
        self.chem_indices = getattr(config, "chem_indices", [36, 37, 38])
        self.raw_indices = getattr(config, "raw_indices", [6, 7, 8])
        self.yiqi_indices = getattr(config, "yiqi_indices", [0, 1] + list(range(9, 21)))
        self.erqi_indices = getattr(config, "erqi_indices", [2] + list(range(21, 34)))
        self.out_indices = getattr(config, "out_indices", [3, 4, 5, 34, 35])
        self.topk = getattr(config, "topk", 3)

        flags = resolve_ablation(
            getattr(config, "ablation_mode", "full"),
            topk=self.topk,
            force_topk=getattr(config, "force_topk", None),
        )
        self.ablation_mode = flags["mode"]
        self.ablation_scheme = "v2_no_topk_baseline"
        self.use_tracker = flags["use_tracker"]
        self.tracker_mode = flags["tracker_mode"]
        self.use_trend = flags["use_trend"]
        self.use_shortcut = flags["use_shortcut"]
        self.use_backbone = flags["use_backbone"]
        self.spatial_mode = flags["spatial_mode"]
        self.effective_topk = flags["effective_topk"]
        print(
            f"[ELFormer-v2] ablation_mode={self.ablation_mode} | "
            f"spatial={self.spatial_mode}, topk={self.effective_topk}, "
            f"tracker={self.tracker_mode}, trend={self.use_trend}, "
            f"shortcut={self.use_shortcut}, smooth={self.ablation_mode != 'no_smooth'}, "
            f"backbone={self.use_backbone}"
        )

       
        self.revin = RevIN(self.n_vars)

        if len(self.chem_indices) > 0:
            self.rtd_smoother = RTDSmoother(num_chem_vars=len(self.chem_indices), kernel_size=5)

        self.trend_smoother = TrendSmoother(kernel_size=getattr(config, "trend_kernel", 25))
        self.trend_proj = nn.Linear(self.seq_len, self.d_model)

        self.patch_len = getattr(config, "patch_len", 16)
        self.stride = getattr(config, "stride", 8)
        self.patch_embedding = PatchEmbedding(self.seq_len, self.patch_len, self.stride, self.d_model)
        self.flatten_proj = nn.Linear(self.patch_embedding.num_patches * self.d_model, self.d_model)

       
        self.register_buffer('zero_delay_mask', torch.full((self.n_vars, self.n_vars), -1e9))

        def allow_instant_interaction(group_indices):
            for i in group_indices:
                for j in group_indices:
                    if i < self.n_vars and j < self.n_vars:
                        self.zero_delay_mask[i, j] = 0.0

        for i in range(self.n_vars):
            self.zero_delay_mask[i, i] = 0.0

        allow_instant_interaction(self.raw_indices + self.chem_indices)
        allow_instant_interaction(self.yiqi_indices)
        allow_instant_interaction(self.erqi_indices)
        allow_instant_interaction(self.out_indices)

        self.st_encoder = nn.ModuleList([
            STEncoderLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout,
                spatial_mode=self.spatial_mode, topk=self.effective_topk
            )
            for _ in range(config.e_layers)
        ])

        self.water_tracker = GroupRTDTracker(
            config=config, n_vars=self.n_vars, d_model=self.d_model,
            Q_mean=getattr(config, "Q_mean", 1452.0),
            min_flow=getattr(config, "min_flow", 800.0),
            beta_floor=getattr(config, "beta_floor", 0.15),
        )

        self.linear_head = nn.Sequential(
            nn.Linear(config.d_model, self.pred_len), nn.Dropout(config.fc_dropout)
        )
        self.linear_shortcut = nn.Linear(config.seq_len, self.pred_len)
        self.trend_gate = nn.Parameter(torch.tensor(0.5))
        self.raw_hrt_gate = nn.Parameter(
            torch.tensor(float(getattr(config, "hrt_gate_init", -2.0)))
        )

    def forward(self, input):
        safe_input = torch.nan_to_num(input, nan=0.0, posinf=0.0, neginf=0.0)

        if len(self.chem_indices) > 0 and self.ablation_mode != "no_smooth":
            chem_seq = safe_input[:, :, self.chem_indices]
            smoothed = self.rtd_smoother(chem_seq)
            safe_input = safe_input.clone()
            safe_input[:, :, self.chem_indices] = smoothed

        norm_input = self.revin(safe_input, 'norm')

        if self.ablation_mode == "shortcut_only":
            output = self.linear_shortcut(norm_input.transpose(1, 2)).transpose(1, 2)
            return self.revin(output, 'denorm'), torch.tensor(0.0, device=output.device)

        if self.use_trend:
            trend_out = self.trend_smoother(norm_input)
            trend_feat = self.trend_proj(trend_out.transpose(1, 2))
        else:
            trend_feat = 0.0

        if not self.use_backbone:
            temporal_patch_feat = 0.0
        else:
            patches = self.patch_embedding(norm_input.transpose(1, 2))
            mask_to_pass = self.zero_delay_mask if self.spatial_mode == "masked" else None

            for layer in self.st_encoder:
                patches = layer(patches, spatial_mask=mask_to_pass)

            B, N, P, D = patches.shape
            temporal_patch_feat = self.flatten_proj(patches.reshape(B, N, P * D))

        euler_feat = temporal_patch_feat + (self.trend_gate * trend_feat if self.use_trend else 0)

        if self.use_tracker:
            raw_flow_seq = input[:, :, self.flow_idx]
            tracked_feat = self.water_tracker(norm_input, raw_flow_seq, tracker_mode=self.tracker_mode)
            temporal_feature = euler_feat + torch.sigmoid(self.raw_hrt_gate) * tracked_feat
        else:
            temporal_feature = euler_feat

        output = self.linear_head(temporal_feature)
        output = rearrange(output, "b n l -> b l n")

        if self.use_shortcut:
            output = output + self.linear_shortcut(norm_input.transpose(1, 2)).transpose(1, 2)

        return self.revin(output, 'denorm'), torch.tensor(0.0, device=output.device)