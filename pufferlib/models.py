import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────

CHESS_BOARD_OBS_SIZE = 775
CHESS_ACTION_SPACE_SIZE = 64 * 73


def _select_num_heads(hidden_size, requested=4):
    num_heads = min(requested, hidden_size)
    while num_heads > 1 and hidden_size % num_heads != 0:
        num_heads -= 1
    return max(1, num_heads)


def _make_transformer_encoder(layer, num_layers):
    try:
        return nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=num_layers)


def _extract_chess_obs(observations):
    if isinstance(observations, dict):
        observations = observations["obs"]
    return observations[..., :CHESS_BOARD_OBS_SIZE].float()


def _extract_chess_action_mask(observations):
    if observations is None:
        return None
    if isinstance(observations, dict):
        return observations.get("action_mask")

    if observations.shape[-1] < CHESS_BOARD_OBS_SIZE + CHESS_ACTION_SPACE_SIZE:
        return None
    return observations[..., CHESS_BOARD_OBS_SIZE:CHESS_BOARD_OBS_SIZE + CHESS_ACTION_SPACE_SIZE]


def _apply_chess_action_mask(logits, observations):
    mask = _extract_chess_action_mask(observations)
    if mask is None:
        return logits

    mask = mask.to(device=logits.device, dtype=logits.dtype)
    if mask.shape[-1] != logits.shape[-1]:
        return logits

    invalid = torch.full_like(logits, -1e9)
    return torch.where(mask > 0.0, logits, invalid)


#  Chess board observation layout (first 775 floats of the flat env obs)
#    obs[0  :768]  →  12 piece-type planes × 64 squares  (0/1 bitboard)
#    obs[768:775]  →  7 auxiliary scalars
#                     [WK_castle, WQ_castle, BK_castle, BQ_castle,
#                      ep_file/7 (−1=none), halfmove/100, side_to_move]
# ─────────────────────────────────────────────────────────────────────────────


class ChessRelativeAttention(nn.Module):
    """
    Multi-head self-attention for a [CLS + 64 squares] token sequence, with a
    learnable 2-D relative-position bias for every square → square pair.

    Design rationale
    ────────────────
    Standard learned absolute-position embeddings treat every square as an
    independent ID.  A 2-D relative bias instead gives each attention head a
    (15 × 15) table indexed by (Δrank, Δfile) ∈ [-7, 7]².  The model can
    learn that bishops favour diagonal offsets, rooks orthogonal ones, and
    knights L-shaped ones — without any hand-crafted inductive bias.

    The CLS token gets its own scalar biases (cls→sq, sq→cls, cls→cls) so it
    can attend to the whole board independently of geometry.

    QKV uses no bias terms — saves parameters and, empirically, works as well
    as with bias for attention-only layers.  The output projection keeps bias
    for the additive shift after aggregation.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must divide evenly by num_heads {num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # 2-D relative-position bias:  (2R-1)² entries × num_heads, R=8
        self.rel_bias_table = nn.Parameter(torch.zeros(15 * 15, num_heads))

        # Precompute and cache the 64×64 index tensor (never changes)
        self._build_rel_index()

        # Scalar biases for the three CLS blocks
        # [cls_to_sq,  sq_to_cls,  cls_to_cls]  ×  num_heads
        self.cls_bias = nn.Parameter(torch.zeros(3, num_heads))

        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

    def _build_rel_index(self):
        """Precompute (Δrank+7)*15 + (Δfile+7) for all 64×64 square pairs."""
        ranks = torch.arange(8).repeat_interleave(8)  # [64]
        files = torch.arange(8).repeat(8)              # [64]
        dr = (ranks.unsqueeze(1) - ranks.unsqueeze(0)) + 7  # [64,64]
        df = (files.unsqueeze(1) - files.unsqueeze(0)) + 7  # [64,64]
        self.register_buffer("rel_idx", dr * 15 + df)       # [64,64] ∈ [0,224]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape   # N = 65 (1 CLS + 64 squares)

        # ── QKV projection ────────────────────────────────────────────────
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)           # [3, B, H, N, D]
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # ── 2-D relative-position bias for the 64×64 square subblock ─────
        rel_bias = self.rel_bias_table[self.rel_idx]          # [64, 64, H]
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)     # [1,  H, 64, 64]
        attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:] + rel_bias

        # ── CLS token scalar biases ───────────────────────────────────────
        b = self.cls_bias                                           # [3, H]
        attn[:, :, 0:1, 1:] += b[0].view(1, self.num_heads, 1, 1) # CLS→sq
        attn[:, :, 1:, 0:1] += b[1].view(1, self.num_heads, 1, 1) # sq→CLS
        attn[:, :, 0:1, 0:1]+= b[2].view(1, self.num_heads, 1, 1) # CLS→CLS

        attn = F.softmax(attn, dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(out)


class ChessTransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block with GELU FFN.

    Pre-norm (norm before attention/FFN, not after) stabilises training with
    deep networks and large learning rates — critical for PPO where the policy
    gradient can produce large parameter updates early in training.

    FFN width = hidden_size × expansion_factor.  With expansion_factor=1 the
    FFN is a pair of square matrices (no bottleneck, no expansion).  For chess,
    expansion_factor ≥ 2 tends to help because the FFN stores pattern-matching
    knowledge (e.g. "two rooks on 7th rank → dangerous") that attention cannot
    easily capture on its own.
    """

    def __init__(self, hidden_size: int, num_heads: int, expansion_factor: int):
        super().__init__()
        ffn_dim = max(hidden_size, hidden_size * expansion_factor)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn  = ChessRelativeAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ChessEncoder(nn.Module):
    """
    Spatial encoder that converts one chess observation into a single hidden
    vector suitable for the PufferLib Policy.  This vector is the CLS token
    output of a custom Transformer whose tokens are the 64 board squares.

    Design choices
    ──────────────
    • Square-as-token:  treating each of the 64 squares as a separate token
      (rather than flattening the whole board) lets every attention head focus
      on subsets of squares — e.g. one head may learn "which squares are
      controlled by the king", another "which diagonals are open".

    • 2-D relative-position bias (see ChessRelativeAttention):  the model
      learns chess geometry (ranks, files, diagonals) directly from data
      instead of needing it hard-coded.

    • CLS token:  a learnable global-summary token that can attend to all 64
      squares simultaneously.  The decoder uses only this token, so the
      Transformer is free to route all "decision-relevant" information there.

    • Aux broadcast:  castling rights, en-passant square, halfmove clock, and
      side-to-move are injected into every token (including CLS) so that every
      attention head has direct access without needing to "find" these scalars
      through cross-token attention.

    • Separate value head weights (in ChessDecoder):  policy and value have
      different credit-assignment targets in PPO; sharing weights can cause the
      gradient from the value loss to corrupt the policy representation.

    Parameters (from [policy] section in .ini)
    ──────────────────────────────────────────
    hidden_size      : width of every token embedding and attention layer
    num_layers       : number of stacked Transformer blocks
    expansion_factor : FFN width multiplier (recommend ≥ 2 for chess)
    """

    def __init__(
        self,
        obs_size: int,
        hidden_size: int = 256,
        num_layers: int = 6,
        expansion_factor: int = 4,
        **kwargs,
    ):
        super().__init__()

        # Pick num_heads so that head_dim ≥ 32 and divides hidden_size cleanly
        num_heads = max(1, hidden_size // 32)
        while hidden_size % num_heads != 0:
            num_heads -= 1

        self.hidden_size = hidden_size

        # ── Input projections ─────────────────────────────────────────────
        # Each square has a 12-dim one-hot piece indicator; project to H.
        # No bias: the positional embedding supplies the additive offset.
        self.piece_proj = nn.Linear(12, hidden_size, bias=False)

        # Aux scalars affect every square equally → project then broadcast.
        self.aux_proj = nn.Linear(7, hidden_size)

        # ── Positional and CLS embeddings ─────────────────────────────────
        # Learnable (not sinusoidal): the 64 chess squares don't have a
        # natural sequential order, so learned embeddings are more flexible.
        self.pos_emb   = nn.Parameter(torch.zeros(64, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.input_norm = nn.LayerNorm(hidden_size)

        # ── Transformer stack ─────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            ChessTransformerBlock(hidden_size, num_heads, expansion_factor)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_emb,   std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = _extract_chess_obs(obs)
        B = obs.shape[0]

        # [B, 12, 64] → [B, 64, 12]: one 12-dim indicator per square
        piece_planes = obs[:, :768].reshape(B, 12, 64).permute(0, 2, 1)
        aux          = obs[:, 768:775]                           # [B, 7]

        x     = self.piece_proj(piece_planes)                    # [B, 64, H]
        x     = x + self.pos_emb.unsqueeze(0)                   # + square pos
        aux_h = self.aux_proj(aux).unsqueeze(1)                  # [B,  1, H]
        x     = x + aux_h                                        # broadcast

        cls = self.cls_token.expand(B, -1, -1) + aux_h          # [B, 1, H]
        x   = torch.cat([cls, x], dim=1)                        # [B, 65, H]
        x   = self.input_norm(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_norm(x)
        return x[:, 0]   # CLS token → [B, H]


class ChessViTEncoder(nn.Module):
    def __init__(
        self,
        obs_space,
        hidden_size=256,
        num_heads=4,
        num_layers=2,
        expansion_factor=2,
        **kwargs,
    ):
        super().__init__()
        num_heads = _select_num_heads(hidden_size, num_heads)
        ffn_size = hidden_size * max(2, int(expansion_factor))

        self.square_embed = nn.Linear(12, hidden_size)
        self.aux_embed = nn.Linear(7, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 65, hidden_size) * 0.02)
        self.input_norm = nn.LayerNorm(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = _make_transformer_encoder(encoder_layer, num_layers)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, observations):
        obs = _extract_chess_obs(observations)
        B = obs.shape[0]

        board_obs = obs[:, :768].reshape(B, 12, 64).permute(0, 2, 1)
        aux_obs = obs[:, 768:775].reshape(B, 1, 7)

        board_tokens = self.square_embed(board_obs)
        aux_token = self.aux_embed(aux_obs)
        tokens = torch.cat([aux_token, board_tokens], dim=1)
        tokens = self.input_norm(tokens + self.pos_embedding)
        encoded_tokens = self.transformer(tokens)
        return self.output_norm(encoded_tokens[:, 0, :])


class ChessDecoder(nn.Module):
    def __init__(self, nvec, hidden_size: int = 256, **kwargs):
        super().__init__()
        self.nvec = tuple(nvec)
        num_actions = int(np.sum(nvec))
        mid = max(hidden_size // 2, 64)

        self.policy = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_actions),
        )

        self.value = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Linear(mid, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for seq in (self.policy, self.value):
            last = [m for m in seq if isinstance(m, nn.Linear)][-1]
            nn.init.orthogonal_(last.weight, gain=0.01)
            nn.init.zeros_(last.bias)

    def forward(self, hidden: torch.Tensor, observations=None):
        logits = self.policy(hidden)
        logits = _apply_chess_action_mask(logits, observations)
        if len(self.nvec) > 1:
            logits = logits.split(self.nvec, dim=1)
        values = self.value(hidden)
        return logits, values


class ChessDualDecoder(nn.Module):
    def __init__(self, nvec, hidden_size=256, **kwargs):
        super().__init__()
        configured_actions = int(np.sum(nvec))
        if configured_actions != CHESS_ACTION_SPACE_SIZE:
            raise ValueError(
                "ChessDualDecoder requires the chess C env to expose "
                f"{CHESS_ACTION_SPACE_SIZE} actions; got {configured_actions}."
            )

        self.num_actions = CHESS_ACTION_SPACE_SIZE
        self.actor_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.num_actions),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden, observations=None):
        logits = self.actor_net(hidden)
        logits = _apply_chess_action_mask(logits, observations)
        values = self.critic_net(hidden)
        return logits, values


class Policy(nn.Module):
    def __init__(self, encoder, decoder, network):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.network = network
        params = inspect.signature(decoder.forward).parameters
        self._decoder_accepts_observations = any(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in params.values()
        ) or "observations" in params or "observation" in params

    def initial_state(self, batch_size, device):
        return self.network.initial_state(batch_size, device)

    def _decode(self, hidden, observations):
        if self._decoder_accepts_observations:
            return self.decoder(hidden, observations)
        return self.decoder(hidden)

    def forward_eval(self, x, state):
        h = self.encoder(x)
        h, state = self.network.forward_eval(h, state)
        logits, values = self._decode(h, x)
        return logits, values, state

    def forward(self, x):
        B, TT = x.shape[:2]
        flat_x = x.reshape(B*TT, *x.shape[2:])
        h = self.encoder(flat_x)
        h = self.network.forward_train(h.reshape(B, TT, -1))
        logits, values = self._decode(h.reshape(B*TT, -1), flat_x)
        return logits, values.reshape(B, TT)

class DefaultEncoder(nn.Module):
    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        self.encoder = nn.Linear(obs_size, hidden_size)

    def forward(self, observations):
        return self.encoder(observations.view(observations.shape[0], -1).float())

class DefaultDecoder(nn.Module):
    def __init__(self, nvec, hidden_size=128):
        super().__init__()
        self.nvec = tuple(nvec)
        self.is_continuous = sum(nvec) == len(nvec)

        if self.is_continuous:
            num_atns = len(nvec)
            self.decoder_mean = nn.Linear(hidden_size, num_atns)
            self.decoder_logstd = nn.Parameter(torch.zeros(1, num_atns))
        else:
            self.decoder = nn.Linear(hidden_size, int(np.sum(nvec)))

        self.value_function = nn.Linear(hidden_size, 1)

    def forward(self, hidden, observations=None):
        if self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            logits = torch.distributions.Normal(mean, torch.exp(logstd))
        else:
            logits = self.decoder(hidden)
            if len(self.nvec) > 1:
                logits = logits.split(self.nvec, dim=1)

        values = self.value_function(hidden)
        return logits, values


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        max_seq_len=64,
        expansion_factor=2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        num_heads = _select_num_heads(hidden_size, num_heads)
        ffn_size = hidden_size * max(2, int(expansion_factor))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = _make_transformer_encoder(layer, num_layers)

    def _causal_mask(self, length, device):
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def initial_state(self, batch_size, device):
        memory = torch.zeros(batch_size, self.max_seq_len, self.hidden_size, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        return memory, lengths

    def forward_eval(self, h, state):
        memory, lengths = state
        new_memory = torch.roll(memory, shifts=-1, dims=1)
        new_memory[:, -1, :] = h
        new_lengths = torch.clamp(lengths + 1, max=self.max_seq_len)

        positions = torch.arange(self.max_seq_len, device=h.device).unsqueeze(0)
        valid_from = self.max_seq_len - new_lengths.unsqueeze(1)
        padding_mask = positions < valid_from

        out = self.transformer(
            new_memory,
            mask=self._causal_mask(self.max_seq_len, h.device),
            src_key_padding_mask=padding_mask,
        )
        return out[:, -1, :], (new_memory, new_lengths)

    def forward_train(self, h):
        T = h.shape[1]
        return self.transformer(h, mask=self._causal_mask(T, h.device))


class MLP(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def initial_state(self, batch_size, device):
        return ()

    def forward_eval(self, h, state):
        return self.net(h), state

    def forward_train(self, h):
        return self.net(h)

class MinGRU(nn.Module):
    # https://arxiv.org/abs/2410.01201v1
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, 3 * hidden_size, bias=False) for _ in range(num_layers)
        ])

    def _g(self, x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def _log_g(self, x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def _highway(self, x, out, proj):
        g = proj.sigmoid()
        return g * out + (1.0 - g) * x

    def _heinsen_scan(self, log_coeffs, log_values):
        a_star = log_coeffs.cumsum(dim=1)
        return (a_star + (log_values - a_star).logcumsumexp(dim=1)).exp()

    def initial_state(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),)

    def forward_eval(self, h, state):
        state = state[0]
        assert state.shape[1] == h.shape[0]
        h = h.unsqueeze(1)
        state_out = []
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h).chunk(3, dim=-1)
            out = torch.lerp(state[i:i+1].transpose(0, 1), self._g(hidden), gate.sigmoid())
            h = self._highway(h, out, proj)
            state_out.append(out[:, -1:])
        return h.squeeze(1), (torch.stack(state_out, 0).squeeze(2),)

    def forward_train(self, h):
        T = h.shape[1]
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h).chunk(3, dim=-1)
            log_coeffs = -F.softplus(gate)
            log_values = -F.softplus(-gate) + self._log_g(hidden)
            out = self._heinsen_scan(log_coeffs, log_values)[:, -T:]
            h = self._highway(h, out, proj)
        return h

class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])

        for i in range(num_layers):
            cell = self.cell[i]
            w_ih = getattr(self.lstm, f'weight_ih_l{i}')
            w_hh = getattr(self.lstm, f'weight_hh_l{i}')
            b_ih = getattr(self.lstm, f'bias_ih_l{i}')
            b_hh = getattr(self.lstm, f'bias_hh_l{i}')
            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()
            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward_eval(self, h, state):
        assert state[0].shape[1] == state[1].shape[1] == h.shape[0]
        lstm_h, lstm_c = state
        for i in range(self.num_layers):
            h, c = self.cell[i](h, (lstm_h[i], lstm_c[i]))
            lstm_h[i] = h
            lstm_c[i] = c
        return h, (lstm_h, lstm_c)

    def forward_train(self, h):
        # h: [B, T, H]
        h = h.transpose(0, 1)
        h, _ = self.lstm(h)
        return h.transpose(0, 1)

class GRU(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.GRUCell(hidden_size, hidden_size) for _ in range(num_layers)])
        self.norm = torch.nn.RMSNorm(hidden_size)

        for i in range(num_layers):
            cell = self.cell[i]
            w_ih = getattr(self.gru, f'weight_ih_l{i}')
            w_hh = getattr(self.gru, f'weight_hh_l{i}')
            b_ih = getattr(self.gru, f'bias_ih_l{i}')
            b_hh = getattr(self.gru, f'bias_hh_l{i}')
            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()
            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h,)

    def forward_eval(self, h, state):
        assert state[0].shape[1] == h.shape[0]
        state = state[0]
        for i in range(self.num_layers):
            h_in = h
            h = self.cell[i](h, state[i])
            state[i] = h
            h = h + h_in
            h = self.norm(h)
        return h, (state,)

    def forward_train(self, h):
        # h: [B, T, H]
        h = h.transpose(0, 1)
        h_in = h
        h, _ = self.gru(h)
        h = h + h_in
        h = self.norm(h)
        return h.transpose(0, 1)

class NatureEncoder(nn.Module):
    '''NatureCNN encoder (Mnih et al. 2015). Returns [batch, hidden_size].'''
    def __init__(self, env, hidden_size=512, framestack=1, flat_size=64*7*7,
            channels_last=False, downsample=1, **kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.network = nn.Sequential(
            nn.Conv2d(framestack, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(input_shape[0], out_channels, 3, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class ImpalaEncoder(nn.Module):
    '''IMPALA ResNet encoder (Espeholt et al. 2018). Returns [batch, hidden_size].'''
    def __init__(self, env, hidden_size=256, cnn_width=16, **kwargs):
        super().__init__()
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2*cnn_width, 2*cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(shape[0] * shape[1] * shape[2], hidden_size),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, observations):
        return self.network(observations.permute(0, 3, 1, 2).float() / 255.0)
