"""Prepare frozen numerical trajectories and training-fitted state encodings."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from formal import foundation as fm
from formal import physical_context as context
from formal.config import load_best_parameters, load_experiment_config
from formal.engine import _seed_everything
from layers.pc_fra import pack_slices
from layers.physical_semantic import compute_physical_tokens

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dev_cache(formal, station):
    return ROOT / fm.prior_cache_path_v4(formal.dataset(station).path, 16, 96, "raw")


def frozen_features(station, split, cfg, formal, *, audit_rows=None):
    """Extract numerical-adapter outputs and historical physical states."""
    if split not in ("train", "validation", "test"):
        raise ValueError(split)
    _seed_everything(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = fm._load_bundle(station, formal)
    starts = getattr(bundle, split).starts
    if split == "train":
        starts = starts[np.linspace(0, len(starts)-1, min(cfg["train_windows"], len(starts)), dtype=int)]
    if audit_rows:
        starts = starts[:audit_rows]
    path = dev_cache(formal, station) if split != "test" else ROOT / (
        f"outputs/chronos/test/{station}.npz")
    fm.build_chronos2_grid_cache(bundle, starts, 16, 96, "raw", out_path=path)
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["dataset_sha256"]) != sha(bundle.path):
            raise ValueError("Chronos cache dataset hash mismatch")
    cached_starts, grid, _ = fm.load_grid_cache(path)
    lookup = {int(s): i for i, s in enumerate(cached_starts)}
    if any(int(s) not in lookup for s in starts):
        raise ValueError("Missing Chronos windows; this study never regenerates old caches")
    records = load_best_parameters(formal.best_parameters)
    state = torch.load(ROOT / fm._checkpoint(station), map_location="cpu", weights_only=True)["state_dict"]
    model, _ = fm._build_frozen(station, formal, bundle, records[station]["params"], state, device)
    batch = int(records[station]["params"]["batch"])
    psrc = fm._infer(model, bundle, starts, batch, device)[1]["pred_std"]
    directory = ROOT / cfg["stations"][station]["adapter_dir"]
    registration = read_json(directory / "validation.json")
    name = registration["locked_prior"]["candidate"]
    scale = registration["locked_prior"]["scale"]
    sd, mu = float(bundle.feature_sd[-1]), float(bundle.feature_mu[-1])
    prior_lookup = fm._calibrated_lookup(name, scale, grid[[lookup[int(s)] for s in starts]],
        np.asarray(fm.CHRONOS2_QUANTILES), bundle, starts, {"pred_phys": psrc * sd + mu})
    # Partner features are unused in the locked PARA path; identity partners
    # avoid introducing unrelated windows and preserve the relevant pack fields.
    pack = fm.build_pack(bundle, starts, 96, 16, prior_lookup,
                         {int(s): psrc[i] for i, s in enumerate(starts)}, starts)
    del model
    model, _ = fm._rebuild_model(station, formal, records, bundle, state,
                                  registration["hyperparameters"], device)
    model.pc_fra_adapter.load_state_dict(torch.load(directory / "adapter_checkpoint.pt",
        map_location=device, weights_only=True)["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    arrays = fm._evaluate(model, fm._loader(bundle, starts, batch, False, 2026, fm_embeddings=pack),
                          bundle, device, collect=True)[1]
    alpha = cfg["stations"][station]["alpha"]
    old = psrc + alpha * (arrays["pred_std"] - psrc)
    raw, physical = [], []
    encoder = model.physical_semantic_encoder
    with torch.no_grad():
        for begin in range(0, len(starts), batch):
            ix = starts[begin:begin+batch, None] + np.arange(96)[None]
            x = torch.as_tensor(bundle.values[ix], device=device)
            raw.append(compute_physical_tokens(x, encoder.roles, encoder.feature_mu,
                       encoder.feature_sd, encoder.target_mu, encoder.target_sd).cpu().numpy())
            physical.append(encoder(x).cpu().numpy())
    roles = encoder.roles
    role_valid = [True, roles["clear_sky"] is not None and roles["clear_sky"] == bundle.clear_sky_index,
                  roles["irradiance"] is not None, roles["temperature"] is not None, True, True]
    prior = (pack.embeddings[:, pack_slices(16)["c_bar"]] - mu) / sd
    result = dict(starts=starts, old=old.astype(np.float32), prior=prior.astype(np.float32),
                  disagreement=(prior-psrc).astype(np.float32), physical=np.concatenate(physical),
                  raw=np.concatenate(raw), target=arrays["target_std"].astype(np.float32),
                  mu=np.float64(mu), sd=np.float64(sd), capacity=np.float64(bundle.capacity),
                  role_valid=np.asarray(role_valid))
    if split == "validation":
        with np.load(directory / "validation_arrays.npz", allow_pickle=False) as saved:
            index = np.searchsorted(bundle.validation.starts, starts)
            reference = saved["psrc_std"][index] + alpha * (
                saved["pred_b_std"][index] - saved["psrc_std"][index])
            np.testing.assert_allclose(old, reference, atol=1e-5, rtol=0,
                                       err_msg="Old formal path did not reproduce; stop before training")
            result["old_max_abs_reconstruction_error"] = np.max(np.abs(old-reference))
    del model
    torch.cuda.empty_cache()
    return result

@torch.no_grad()
def encode_bank(prompts, model, tokenizer):
    outputs = []
    for start in range(0, len(prompts), 8):
        texts = [tokenizer.apply_chat_template([
            {'role':'system','content':'Represent observed photovoltaic states for numerical forecast calibration. Do not infer unobserved weather or future measurements.'},
            {'role':'user','content':p}], tokenize=False, add_generation_prompt=True)
            for p in prompts[start:start+8]]
        tokens = tokenizer(texts, padding=True, return_tensors='pt').to(next(model.parameters()).device)
        if tokens.input_ids.shape[1] > 512:
            raise ValueError('Unexpected prompt length')
        h = model(**tokens, use_cache=False).last_hidden_state
        # Last non-padding position summarizes the full causal prompt.
        ix = tokens.attention_mask.sum(1)-1
        h = h[torch.arange(len(h), device=h.device), ix].float()
        # Fixed low-dimensional sketch keeps state geometry and reduces cache size.
        rng = np.random.default_rng(2026)
        projection = torch.as_tensor(rng.normal(0, 1/8, (h.shape[1],64)),
                                     device=h.device, dtype=h.dtype)
        outputs.append((h@projection).cpu().numpy())
        if start % 256 == 0:
            print(f'[Qwen encode] {start}/{len(prompts)}', flush=True)
    return np.concatenate(outputs)

def state_features(data, vocabulary, encoding='language', *, model=None, tokenizer=None):
    codes = context.record_codes(data['raw'], data['role_valid'], vocabulary)
    if encoding == 'numeric':
        rows = []
        for k in range(6):
            entries = context._active_entries(vocabulary,k)
            digits = [context.category_digits(data['raw'],data['role_valid'],e,len(codes)) /
                      max(context.CARD[e['kind']]-1,1) for e in entries]
            row = np.stack(digits,axis=-1) if digits else np.zeros((len(codes),1))
            rows.append(np.pad(row,((0,0),(0,64-row.shape[1]))))
        state = np.stack(rows,axis=1).astype(np.float64)
        projected = state @ np.random.default_rng(2026).normal(0,1,(64,32))
        return np.concatenate((np.sin(projected),np.cos(projected)),-1)
    if encoding != 'language' or model is None or tokenizer is None:
        raise ValueError('Language encoding requires a frozen Qwen model and tokenizer')
    keys = np.asarray([(k,int(c)) for k in range(6) for c in np.unique(codes[:,k])])
    prompts, keys = context.context_prompts(vocabulary,keys)
    bank = encode_bank(prompts,model,tokenizer)
    lookup = {tuple(k):i for i,k in enumerate(keys)}
    indices = np.array([[lookup[(k,int(c))] for k,c in enumerate(row)] for row in codes])
    return bank[indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--station',required=True)
    parser.add_argument('--split',choices=('train','validation','test'),required=True)
    parser.add_argument('--encoding',choices=('language','numeric'),default='language')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--foundation',type=Path,default=Path('outputs/foundation/selected'))
    args = parser.parse_args()
    formal = load_experiment_config(ROOT/'configs/formal_experiment.json')
    cfg = dict(train_windows=6000,stations={args.station:
        dict(adapter_dir=str((args.foundation/args.station).resolve()),alpha=1.)})
    folder = args.output
    path = folder/f'{args.split}.npz'
    if path.exists():
        raise FileExistsError(f'{path} already exists; select a new output directory')
    vocab_path = folder/'vocabulary.json'
    if args.split != 'train' and not vocab_path.exists():
        raise FileNotFoundError('Prepare training features first to fit the state vocabulary')
    data = frozen_features(args.station,args.split,cfg,formal)
    if args.split == 'train':
        vocabulary = context.fit_record(data['raw'],data['role_valid'])
    else:
        vocabulary = read_json(vocab_path)
    model = tokenizer = None
    if args.encoding == 'language':
        from transformers import AutoModel, AutoTokenizer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
        tokenizer.padding_side = 'right'
        model = AutoModel.from_pretrained(MODEL,revision=REVISION,
            torch_dtype=torch.float16 if device=='cuda' else torch.float32,
            attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    data['language'] = state_features(data,vocabulary,args.encoding,model=model,tokenizer=tokenizer)
    data['available'] = np.broadcast_to(data['role_valid'],(len(data['old']),6)).copy()
    folder.mkdir(parents=True,exist_ok=True)
    if args.split == 'train':
        vocab_path.write_text(json.dumps(vocabulary,indent=2),encoding='utf-8')
    np.savez_compressed(path,**data)
    print(path)


if __name__ == '__main__':
    main()
