"""Training-fitted categorical physical states and contextual prompt templates."""
from itertools import product

import numpy as np

from formal.state_fields import FIELDS

CARD = {"level": 3, "dirmag": 7, "dir": 3, "bit": 2}
TASK = ("Task: represent the historical state for correcting the next 16 "
        "numerical forecast steps.")
NAMES = ("ramp", "reference variability", "irradiance-power coupling",
         "thermal state", "diurnal phase", "volatility state")
LEAD = (
    "Recent power ramps and net power changes.",
    "Power relative to its reference and reference variability.",
    "Irradiance-power coupling strength, direction and lag relationship.",
    "Temperature level, recent departure and temperature-power correlation.",
    "Position within the daily cycle and the available solar geometry proxy.",
    "Multi-timescale residual power fluctuation level.",
)

# Own descriptors: (source category, source column, kind, label).
OWN = (
    (  # Ramp: direction (signed net change) + multi-scale strength.
        (0, 0, "level", "1-hour mean absolute 15-minute ramp"),
        (0, 1, "level", "3-hour mean absolute 15-minute ramp"),
        (0, 2, "level", "6-hour mean absolute 15-minute ramp"),
        (0, 3, "dirmag", "3-interval net power change"),
        (0, 4, "dirmag", "11-interval net power change"),
    ),
    (  # Reference variability.
        (1, 0, "level", "power-to-reference ratio level"),
        (1, 1, "level", "ratio variability"),
        (1, 2, "level", "mean absolute adjacent ratio change"),
        (1, 3, "level", "ratio transition fraction"),
        (1, 4, "dirmag", "last-daylight to first-daylight-quarter ratio difference"),
    ),
    (  # Coupling: direction by sign, strength from |corr|.
        (2, 0, "dirmag", "24-hour irradiance-power correlation"),
        (2, 1, "dirmag", "6-hour irradiance-power correlation"),
        (2, 2, "dirmag", "one-step lag irradiance-power correlation"),
    ),
    (  # Thermal.
        (3, 0, "level", "24-hour temperature level"),
        (3, 1, "dirmag", "recent temperature departure"),
        (3, 2, "dirmag", "temperature-power correlation"),
    ),
    (  # Diurnal.
        (4, 0, "level", "daily sine phase coordinate"),
        (4, 1, "level", "daily cosine phase coordinate"),
        (4, 2, "level", "solar elevation proxy"),
    ),
    (  # Volatility.
        (5, 0, "level", "6-hour residual variability"),
        (5, 1, "level", "12-hour residual variability"),
        (5, 2, "level", "24-hour residual variability"),
    ),
)

# Context descriptors per category; src None with kind "bit" reads the role
# mask at role index 2 (irradiance/reference availability).
CONTEXT = (
    ((4, 2, "level", "intraday phase (solar elevation proxy)"),
     (5, 0, "level", "short-term residual fluctuation")),
    ((0, 4, "dir", "power change direction"),
     (None, 1, "bit", "reference measurements available")),
    ((0, 3, "dirmag", "recent 3-interval power change"),
     (4, 2, "level", "intraday phase (solar elevation proxy)")),
    ((0, 4, "dir", "power change direction"),
     (4, 2, "level", "intraday phase (solar elevation proxy)")),
    ((1, 0, "level", "historical power-to-reference ratio level"),
     (0, 3, "dir", "power change direction")),
    ((0, 3, "dir", "ramp direction"),
     (4, 2, "level", "intraday phase (solar elevation proxy)")),
)


def _field_active(field, raw_train, role_valid):
    kind = field[2]
    if kind == "bit":
        return True
    k, j = field[0], field[1]
    if not bool(role_valid[k]) or j >= len(FIELDS[k]):
        return False
    return float(np.std(raw_train[:, k, j])) > 1e-6


def fit_record(raw_train, role_valid):
    """Train-only thresholds. Returns a JSON-safe vocabulary record."""
    raw = np.asarray(raw_train, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (6, 6) or not np.isfinite(raw).all():
        raise ValueError("finite [N,6,6] training descriptors required")
    role_valid = np.asarray(role_valid, dtype=bool)
    if role_valid.shape != (6,):
        raise ValueError("role_valid must be [6]")
    cats = []
    for k in range(6):
        own_entries, ctx_entries = [], []
        for field in OWN[k]:
            sk, sj, kind, label = field
            active = _field_active(field, raw, role_valid)
            edges = None
            if active and kind in ("level", "dirmag"):
                values = np.abs(raw[:, sk, sj]) if kind == "dirmag" else raw[:, sk, sj]
                edges = np.quantile(values, [1/3, 2/3]).round(10).tolist()
            own_entries.append(dict(src=[-1, sj] if sk is None else [sk, sj],
                                    kind=kind, label=label, active=active, edges=edges))
        for field in CONTEXT[k]:
            sk, sj, kind, label = field
            active = _field_active(field, raw, role_valid)
            edges = None
            if active and kind in ("level", "dirmag"):
                values = np.abs(raw[:, sk, sj]) if kind == "dirmag" else raw[:, sk, sj]
                edges = np.quantile(values, [1/3, 2/3]).round(10).tolist()
            ctx_entries.append(dict(src=[-1, sj] if sk is None else [sk, sj],
                                    kind=kind, label=label, active=active, edges=edges))
        cats.append({"own": own_entries, "context": ctx_entries})
    return dict(version="physical_context_v1", names=list(NAMES), categories=cats)


def _digit(value, kind, edges):
    value = np.asarray(value, dtype=np.float64)
    if kind == "level":
        return (value[..., None] > np.asarray(edges)).sum(-1).astype(np.int64)
    if kind == "dirmag":
        direction = (value > 0).astype(np.int64)
        magnitude = (np.abs(value)[..., None] > np.asarray(edges)).sum(-1).astype(np.int64)
        return np.where(np.abs(value) <= 1e-6, 0, 1 + magnitude * 2 + direction)
    if kind == "dir":
        return np.where(np.abs(value) <= 1e-6, 0, np.where(value < 0, 1, 2))
    raise ValueError(kind)


def category_digits(raw, role_mask, entry, n):
    """Digit column for one field; inactive fields are identically zero."""
    kind, active = entry["kind"], entry["active"]
    if not active:
        return np.zeros(n, dtype=np.int64)
    if kind == "bit":
        return np.broadcast_to(np.asarray(role_mask)[entry["src"][1]], n).astype(np.int64).copy()
    sk, sj = entry["src"]
    values = np.asarray(raw)[:, sk, sj]
    return _digit(values, kind, entry["edges"])


def record_codes(raw, role_valid, vocabulary):
    """Mixed-radix code per category for every window: [N,6]."""
    raw = np.asarray(raw)
    n = len(raw)
    role_valid = np.asarray(role_valid, dtype=bool)
    codes = np.zeros((n, 6), dtype=np.int64)
    for k, cat in enumerate(vocabulary["categories"]):
        stride = 1
        for entries in (cat["own"], cat["context"]):
            for entry in entries:
                if not entry["active"]:
                    continue
                digit = category_digits(raw, role_valid, entry, n)
                codes[:, k] += digit * stride
                stride *= CARD[entry["kind"]]
    return codes


def cardinalities(vocabulary):
    cards, strides = [], []
    for cat in vocabulary["categories"]:
        per, st = [], []
        stride = 1
        for entries in (cat["own"], cat["context"]):
            for entry in entries:
                per.append(CARD[entry["kind"]] if entry["active"] else 1)
                st.append(stride)
                stride *= per[-1]
        cards.append(per)
        strides.append(st)
    return cards, strides


def _direction_words(label, direction):
    if "correlation" in label:
        return "negative" if direction == 0 else "positive"
    if "departure" in label:
        return "cool departure" if direction == 0 else "warm departure"
    return "falling" if direction == 0 else "rising"


def _clause(entry, digit):
    label, kind = entry["label"], entry["kind"]
    if kind == "level":
        return f"{label} is {('low', 'middle', 'high')[digit]}"
    if kind == "dirmag":
        if digit == 0:
            return f"{label}: approximately zero"
        magnitude, direction = (digit - 1) // 2, (digit - 1) % 2
        strength = ("weak", "moderate", "strong")[magnitude]
        return f"{label}: {_direction_words(label, direction)}, {strength} relative magnitude"
    if kind == "dir":
        return f"{label}: " + ("approximately unchanged" if digit == 0 else _direction_words(label, digit - 1))
    if kind == "bit":
        return f"{label}: {'yes' if digit else 'no'}"
    raise ValueError(kind)


def _active_entries(vocabulary, k):
    """Every active field of category k, in own-then-context radix order."""
    cat = vocabulary["categories"][k]
    return [e for e in (*cat["own"], *cat["context"]) if e["active"]]


def context_prompts(vocabulary, requested_keys=None):
    """Full Cartesian prompt bank over every reachable state; [keys]=(k,code)."""
    prompts, keys = [], []
    for k, cat in enumerate(vocabulary["categories"]):
        index = _active_entries(vocabulary, k)
        active_own = [e for e in cat["own"] if e["active"]]
        if requested_keys is None:
            combinations = product(*(range(CARD[e["kind"]]) for e in index))
        else:
            combinations = []
            for category, code in requested_keys:
                if int(category) != k:
                    continue
                remainder, digits_for_code = int(code), []
                for entry in index:
                    card = CARD[entry["kind"]]
                    digits_for_code.append(remainder % card)
                    remainder //= card
                if remainder:
                    raise ValueError("State code outside registered vocabulary")
                combinations.append(digits_for_code)
        for levels in combinations:
            digits = {id(e): d for e, d in zip(index, levels)}
            code, stride = 0, 1
            for entry in index:
                code += digits[id(entry)] * stride
                stride *= CARD[entry["kind"]]
            if active_own:
                own_clauses = "; ".join(_clause(e, digits[id(e)])
                                        for e in active_own)
            else:
                own_clauses = "own measurements unavailable"
            ctx_entries = [e for e in cat["context"] if e["active"]]
            parts = [TASK, f" Photovoltaic {NAMES[k]} state. {LEAD[k]}",
                     f" State: {own_clauses}."]
            if ctx_entries:
                parts.append(" Context: "
                             + "; ".join(_clause(e, digits[id(e)]) for e in ctx_entries) + ".")
            parts.append(" Magnitude and level groups use training reference thresholds.")
            prompts.append("".join(parts))
            keys.append((k, code))
    return prompts, np.asarray(keys, dtype=np.int64)


def code_space_size(vocabulary):
    sizes = []
    for cat in vocabulary["categories"]:
        size = 1
        for entries in (cat["own"], cat["context"]):
            for entry in entries:
                if entry["active"]:
                    size *= CARD[entry["kind"]]
        sizes.append(size)
    return sizes


def numeric_context_bank(vocabulary, dim=768, seed=2026, requested_keys=None):
    """Project concatenated FIELD one-hots, preserving shared-state structure."""
    keys, embeddings = [], []
    for k, size in enumerate(code_space_size(vocabulary)):
        codes = (np.arange(size) if requested_keys is None else
                 np.asarray([int(c) for category, c in requested_keys if int(category) == k]))
        if not len(codes):
            continue
        entries = _active_entries(vocabulary, k)
        width = sum(CARD[e["kind"]] for e in entries)
        rng = np.random.default_rng(seed + k)
        projection = rng.normal(size=(width, dim)).astype(np.float32) / np.sqrt(max(1, len(entries)))
        bank = np.zeros((len(codes), dim), dtype=np.float32)
        remainder, offset = codes.copy(), 0
        for entry in entries:
            card = CARD[entry["kind"]]
            bank += projection[offset + remainder % card]
            remainder //= card
            offset += card
        keys.append(np.stack([np.full(len(codes), k), codes], axis=1))
        embeddings.append(bank)
    return np.concatenate(keys), np.concatenate(embeddings)


def mismatch_donors(codes, role_masks, train_codes, train_masks, split, salt=0):
    """Whole-window donors whose six-bit role signature matches exactly.

    Train split: derangement within each signature group. Validation/test:
    donors come ONLY from training windows. Masks are compared, never just
    inferred from zero codes.
    """
    masks = np.asarray(role_masks, dtype=np.uint8)
    tmasks = np.asarray(train_masks, dtype=np.uint8)
    signatures = masks.dot(1 << np.arange(masks.shape[1]))
    train_sig = tmasks.dot(1 << np.arange(tmasks.shape[1]))
    groups = {}
    for i, sig in enumerate(train_sig):
        groups.setdefault(int(sig), []).append(i)
    donor = np.empty(len(codes), dtype=np.int64)
    if split == "train":
        rng = np.random.default_rng(2026 + salt)
        for sig, members in groups.items():
            if len(members) < 2:
                raise ValueError("Mask-aligned train derangement needs >=2 windows per signature")
            order = rng.permutation(len(members))
            rolled = np.roll(order, 1)
            for dst, src in zip(order, rolled):
                donor[members[dst]] = members[src]
        if set(int(s) for s in signatures) - set(groups):
            raise ValueError("A training window signature is missing from the donor pool")
    elif split in ("validation", "test"):
        missing = set(int(s) for s in signatures) - set(groups)
        if missing:
            raise ValueError("No mask-aligned training donor for signature(s); refuse unmasked mismatch")
        for i, sig in enumerate(signatures):
            members = groups[int(sig)]
            rng = np.random.default_rng(2026_000_000 + salt * 7919 + i)
            donor[i] = members[int(rng.integers(len(members)))]
    else:
        raise ValueError(split)
    return train_codes[donor]
