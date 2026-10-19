"""Semantic residual calibration head (PSRC components 2 + 3).

Given the numerical backbone's base forecast and the physical semantic
state tokens, this module predicts a bounded residual correction and a
per-horizon confidence gate:

    yhat = yhat0 + g * dyhat,   g in [0, 1]

The probe evidence behind this design: physical state tokens carry
residual-predictive signal (I(S_physical; e) > 0), the signal concentrates
in hard regimes (cloud transitions, ramps, high volatility), and a learned
gate that decides *when to trust* the correction beats applying it
unconditionally.  This module is deliberately independent of the TGSM
prompt machinery: PSRC consumes structured physical state, not language.
"""

from __future__ import annotations

import torch
from torch import nn


def _semantic_state_memory(
    semantic_state: torch.Tensor,
    *,
    batch: int,
    token_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if semantic_state.ndim != 3:
        raise ValueError("semantic state tokens must be [batch, tokens, token_dim]")
    if semantic_state.shape[0] != batch or semantic_state.shape[-1] != token_dim:
        raise ValueError(
            f"expected semantic state batch={batch}, width={token_dim}, "
            f"got {tuple(semantic_state.shape)}"
        )
    return semantic_state.to(device=device, dtype=dtype)


class SemanticResidualCalibration(nn.Module):
    """Confidence-gated residual correction of the base forecast.

    Horizon queries are conditioned on the backbone context and the base
    forecast itself, cross-attend to the physical state tokens, and emit a
    tanh-bounded residual dyhat plus a sigmoid confidence gate g.  The
    residual head is zero-initialized so the branch starts as an exact
    no-op (identity-safe): the PSRC model reproduces the numerical-only
    baseline at initialization and only departs from it as the calibration
    loss drives learning.
    """

    def __init__(
        self,
        d_model: int,
        token_dim: int,
        pred_len: int,
        heads: int = 4,
        max_correction: float = 0.5,
        max_gate: float = 1.0,
        use_gate: bool = True,
        dropout: float = 0.0,
        semantic_variant: str = "cross",
        stat_width: int = 36,
        mlp_width: int | None = None,
        concat_width: int | None = None,
        shared_retrieval: bool = False,
    ) -> None:
        super().__init__()
        self.shared_retrieval = shared_retrieval
        if semantic_variant not in ("cross", "mlp", "concat"):
            raise ValueError(f"unknown semantic_variant {semantic_variant!r}")
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by calibration attention heads")

        self.d_model = d_model
        self.token_dim = token_dim
        self.pred_len = pred_len
        self.max_correction = max_correction
        if not 0.0 <= max_gate <= 1.0:
            raise ValueError("max_gate must be between zero and one")
        self.max_gate = float(max_gate)
        self.use_gate = bool(use_gate)
        self.semantic_variant = str(semantic_variant)

        self.context_projection = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
        )
        self.prediction_projection = nn.Linear(1, d_model, bias=False)
        self.horizon_embedding = nn.Parameter(torch.empty(1, pred_len, d_model))
        nn.init.normal_(self.horizon_embedding, std=0.02)

        self.semantic_projection = nn.Linear(token_dim, d_model, bias=False)
        self.semantic_norm = nn.LayerNorm(d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1) if self.use_gate else None

        # Matched-capacity control branches. Built ONLY for non-cross
        # variants so the formal "cross" state dict stays identical to the
        # locked checkpoints (strict loading elsewhere is unaffected).
        # Hidden widths are solved from the ACTUAL parameter count of the
        # retrieval path these branches replace (cross-attention +
        # semantic projection), keeping total instantiated parameters equal
        # across variants to within a handful of parameters.
        retrieval_params = sum(
            p.numel() for p in self.cross_attention.parameters()
        ) + sum(p.numel() for p in self.semantic_projection.parameters())
        self.control_target_params = retrieval_params
        self.numeric_mlp = None
        self.concat_mlp = None
        self.control_actual_params = 0
        if semantic_variant == "mlp":
            # Linear(D,h) + Linear(h,D) parameter count = h(2D+1) + D.
            if mlp_width is None:
                unit, const = 2 * d_model + 1, d_model
                cands = [(retrieval_params - const) // unit,
                         (retrieval_params - const) // unit + 1]
                mlp_width = min(cands, key=lambda h: abs(
                    h * unit + const - retrieval_params))
            self.numeric_mlp = nn.Sequential(
                nn.Linear(d_model, mlp_width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_width, d_model),
            )
            self.control_actual_params = sum(
                p.numel() for p in self.numeric_mlp.parameters())
        elif semantic_variant == "concat":
            # No LayerNorm: the flattened statistics are already standardized
            # with training-window token statistics.
            # Linear(S,h) + Linear(h,D) parameter count = h(S+D+1) + D.
            if concat_width is None:
                unit, const = stat_width + d_model + 1, d_model
                cands = [(retrieval_params - const) // unit,
                         (retrieval_params - const) // unit + 1]
                concat_width = min(cands, key=lambda h: abs(
                    h * unit + const - retrieval_params))
            self.concat_mlp = nn.Sequential(
                nn.Linear(stat_width, concat_width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(concat_width, d_model),
            )
            self.control_actual_params = sum(
                p.numel() for p in self.concat_mlp.parameters())

        if semantic_variant in ("mlp", "concat"):
            # Retrieval modules are graph-unreachable in these arms; freeze
            # them so trainable parameter counts match the cross arm exactly
            # (AdamW would otherwise decay parameters with zero gradient).
            for module in (self.cross_attention, self.semantic_projection,
                           self.semantic_norm):
                for p in module.parameters():
                    p.requires_grad_(False)

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        if self.confidence_head is not None:
            nn.init.zeros_(self.confidence_head.weight)
            nn.init.constant_(self.confidence_head.bias, -2.0)

    def forward(
        self,
        variables: torch.Tensor,
        state: torch.Tensor,
        base_prediction: torch.Tensor,
        semantic_state: torch.Tensor | None,
        *,
        numeric_only: bool = False,
        semantic_raw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (corrected forecast, raw residual dyhat, confidence gate).

        ``corrected = base_prediction + gate * dyhat``; with the gate
        disabled the gate term is exactly one (unconditional correction).

        Control variants (supplementary matched-capacity study):
          * numeric_only ("numeric_residual"): cross-attention self-attends
            to the numerical horizon queries, no physical tokens;
          * semantic_variant="mlp": a parameter-matched per-horizon MLP
            replaces cross-attention, no physical tokens;
          * semantic_variant="concat": flattened standardized physical
            statistics feed a parameter-matched MLP broadcast across
            horizons, without any attention over token categories.

        """
        batch = variables.shape[0]
        if base_prediction.shape != (batch, self.pred_len):
            raise ValueError("base prediction must be [batch, pred_len]")

        target = variables[:, -1]
        context = variables.mean(dim=1)
        sample_context = self.context_projection(
            torch.cat([target, context, state], dim=-1)
        )
        queries = sample_context.unsqueeze(1) + self.horizon_embedding
        queries = queries + self.prediction_projection(base_prediction.unsqueeze(-1))
        query_norm = self.query_norm(queries)
        if self.semantic_variant == "mlp":
            if semantic_state is not None or semantic_raw is not None:
                raise ValueError("mlp residual variant consumes no semantic input")
            attended = self.numeric_mlp(query_norm)
        elif self.semantic_variant == "concat":
            if semantic_raw is None:
                raise ValueError("concat residual variant requires semantic_raw")
            if semantic_raw.shape != (batch, 36):
                raise ValueError("semantic_raw must be [batch, 36]")
            attended = self.concat_mlp(
                semantic_raw.to(variables.dtype)
            ).unsqueeze(1).expand(-1, self.pred_len, -1)
        elif numeric_only:
            # Self-attention over the numerical queries only; no physical
            # semantic token and no extra parameters are introduced.
            attended, _ = self.cross_attention(
                query_norm, query_norm, query_norm, need_weights=False
            )
        else:
            memory = _semantic_state_memory(
                semantic_state,
                batch=batch,
                token_dim=self.token_dim,
                device=variables.device,
                dtype=variables.dtype,
            )
            memory = self.semantic_norm(self.semantic_projection(memory))
            attended, _ = self.cross_attention(
                query_norm.mean(1, keepdim=True) if self.shared_retrieval else query_norm,
                memory, memory, need_weights=False
            )
            if self.shared_retrieval:
                attended = attended.expand(-1, self.pred_len, -1)
        fused = self.fusion(torch.cat([queries, attended, queries * attended], dim=-1))

        residual = self.max_correction * torch.tanh(
            self.residual_head(fused).squeeze(-1)
        )
        if self.confidence_head is None:
            gate = torch.full_like(residual, self.max_gate)
        else:
            gate = self.max_gate * torch.sigmoid(
                self.confidence_head(fused).squeeze(-1)
            )
        return base_prediction + gate * residual, residual, gate
