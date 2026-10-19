"""History-only state vocabulary and frozen GPT-2 representations.

    Text describes measured statistics, not inferred weather causes or laws.
    Thresholds and validity come ONLY from the training cache. Legacy feature
    definitions remain untouched. Invalid cloud proxies are unavailable here.
"""
from itertools import product

import numpy as np
import torch

MODEL_ID = "gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
NAMES = ("power ramp", "power/reference variability", "irradiance-power coupling",
         "temperature state", "diurnal phase", "power variability")
FIELDS = (
    ("1-hour mean absolute 15-minute ramp", "3-hour mean absolute 15-minute ramp",
     "6-hour mean absolute 15-minute ramp",
     "3-interval net change", "11-interval net change"),
    ("power-to-reference ratio mean", "ratio variability", "adjacent ratio change",
     "ratio transition fraction", "last-daylight to first-daylight-quarter ratio difference"),
    ("24-hour correlation", "6-hour correlation", "one-step lag correlation"),
    ("24-hour temperature mean", "recent temperature departure", "temperature-power correlation"),
    ("daily sine", "daily cosine", "solar elevation proxy"),
    ("6-hour residual variability", "12-hour residual variability", "24-hour residual variability"),
)


def fit_vocabulary(raw_train, role_valid):
    raw = np.asarray(raw_train, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (6, 6) or not np.isfinite(raw).all():
        raise ValueError("finite [N,6,6] training descriptors required")
    mean, sd = raw.mean(0), raw.std(0)
    active = sd > 1e-6
    active &= np.asarray(role_valid, dtype=bool)[:, None]
    for k, fields in enumerate(FIELDS):
        active[k, len(fields):] = False
    return dict(mean=mean.tolist(), scale=np.maximum(sd, 1e-3).tolist(),
                edges=np.quantile(raw, [1/3, 2/3], axis=0).transpose(1, 2, 0).tolist(),
                active=active.tolist())


def state_codes(raw, vocabulary):
    raw = np.asarray(raw)
    active, edges = np.asarray(vocabulary["active"]), np.asarray(vocabulary["edges"])
    bins = (raw[..., None] > edges).sum(-1)
    codes = np.zeros((len(raw), 6), dtype=np.int64)
    for k in range(6):
        for j, col in enumerate(np.flatnonzero(active[k])):
            codes[:, k] += bins[:, k, col] * 3**j
    return codes


def prompt_bank(vocabulary):
    """All discrete states are encoded, not a validation/test-fitted vocabulary."""
    prompts, keys = [], []
    for k, mask in enumerate(np.asarray(vocabulary["active"])):
        columns = np.flatnonzero(mask)
        for levels in product(range(3), repeat=len(columns)):
            code = sum(level * 3**j for j, level in enumerate(levels))
            clauses = [f"{FIELDS[k][col]} is {('low', 'middle', 'high')[level]}"
                       for col, level in zip(columns, levels)]
            body = "; ".join(clauses) if clauses else "measurements unavailable"
            prompts.append(f"Historical photovoltaic {NAMES[k]}. {body}. "
                           "Levels are relative to the training reference distribution.")
            keys.append((k, code))
    return prompts, np.asarray(keys, dtype=np.int64)


def last_valid_hidden(hidden, attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=hidden.device)
    index = positions.expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1).amax(1)
    if (index < 0).any():
        raise ValueError("empty prompt")
    return hidden[torch.arange(len(hidden), device=hidden.device), index]


@torch.no_grad()
def encode_prompts(prompts, *, random_weights=False, batch_size=16, device="cpu"):
    """Local-only pinned checkpoint; eval/no-grad, last NON-padding token.

    A matched random-init GPT-2 is encoded once for the pretraining control.
    No text generation, targets or external API calls are involved.
    """
    from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        if random_weights:
            model = GPT2Model(GPT2Config.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True))
        else:
            model = GPT2Model.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    model = model.to(device).eval().requires_grad_(False)
    outputs = []
    for begin in range(0, len(prompts), batch_size):
        encoded = tokenizer(prompts[begin:begin+batch_size], padding=True, return_tensors="pt")
        if encoded.input_ids.shape[1] > model.config.n_positions:
            raise ValueError("prompt exceeds GPT-2 context; refuse silent truncation")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        hidden = model(**encoded).last_hidden_state
        outputs.append(last_valid_hidden(hidden, encoded["attention_mask"]).float().cpu().numpy())
    return np.concatenate(outputs)


def numeric_features(raw, vocabulary):
    """Fixed nonlinear numerical embedding for the equal-trainable-budget control.

    Dense random Fourier features, not zero padding with inactive parameters.
    It contains only the same descriptors; no pretrained model is used.
    """
    z = (raw - np.asarray(vocabulary["mean"])) / np.asarray(vocabulary["scale"])
    z = z * np.asarray(vocabulary["active"])
    rng = np.random.default_rng(2026)
    matrix = rng.normal(size=(6, 384)) / np.sqrt(6)
    phase = rng.uniform(0, 2*np.pi, 384)
    angles = np.clip(z, -10, 10) @ matrix + phase
    return np.concatenate((np.sin(angles), np.cos(angles)), -1).astype(np.float32)
