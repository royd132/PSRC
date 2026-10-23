"""Physical-language PFA heads, training objectives, and calibrated readout.

Inputs are frozen upstream features prepared by formal.pfa_features. Training
uses training targets and validation selection; inference needs no targets.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from formal.engine import _seed_everything
from formal.metrics import regression_metrics
from models.qwen_guidance import QwenGuidedChronos


def tensors(data, device='cpu'):
    names = ('old', 'prior', 'disagreement', 'physical', 'language', 'available')
    values = tuple(torch.as_tensor(data[k], dtype=torch.float32, device=device) for k in names)
    n = len(values[0])
    expected = ((n, 16), (n, 16), (n, 16), (n, 6, 32), (n, 6, 64), (n, 6))
    if not n or any(tuple(v.shape) != shape or not torch.isfinite(v).all()
                    for v, shape in zip(values, expected)):
        raise ValueError('Expected finite H16 trajectories and six physical/language states')
    if not values[-1].bool().any(1).all():
        raise ValueError('Every window needs at least one available state')
    return values


def statistics(values):
    stats = {}
    for name, x in (('physical', values[3]), ('language', values[4]),
                    ('trajectory', torch.cat(values[:3], -1))):
        mean, sd = x.mean(0), x.std(0, correction=0)
        floor = sd.mean(-1, keepdim=True).clamp_min(1e-3) * .1
        stats[name + '_mu'], stats[name + '_sd'] = mean, torch.maximum(sd, floor)
    return stats


def build_head(config, sigma, stats, device='cpu'):
    _seed_everything(2026)
    return QwenGuidedChronos(max(1e-6, sigma * config['sigma_fraction']), stats,
        gate=config['gate'], temperature=config['temperature']).to(device)


def load_head(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved['state_dict']
    stats = {k: v for k, v in state.items() if k.endswith(('_mu', '_sd'))}
    model = build_head(saved['config'], saved['sigma'], stats, device)
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def predict_head(model, values, batch_size=512):
    model.eval()
    return np.concatenate([model(*(v[i:i+batch_size] for v in values))[0].cpu().numpy()
                           for i in range(0, len(values[0]), batch_size)])


def regime_bins(data, regime):
    edges = np.asarray(regime['edges'])
    if len(regime['scales']) != len(edges) + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError('Regime boundaries must increase and match the scale count')
    if float(data['sd']) <= 0 or float(data['capacity']) <= 0:
        raise ValueError('Positive training power scale and standard deviation required')
    ratio = np.maximum(0, data['old'] * data['sd'] + data['mu']) / float(data['capacity'])
    return np.digitize(ratio, edges)


def regime_readout(raw, data, regime):
    return data['old'] + np.asarray(regime['scales'])[regime_bins(data, regime)] * (raw-data['old'])


def combine_heads(reference, members, weights, reduction='mean'):
    if len(members) != len(weights):
        raise ValueError('Each head needs a readout weight')
    if not members:
        return reference
    values = [reference + s * (p-reference) for p, s in zip(members, weights)]
    if reduction not in ('mean', 'median'):
        raise ValueError('Readout reduction must be mean or median')
    return (np.mean if reduction == 'mean' else np.median)(values, axis=0)


def physical_output(prediction, data):
    return np.maximum(0, prediction * data['sd'] + data['mu'])


def metrics(prediction, data):
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite forecast')
    return regression_metrics(physical_output(prediction, data),
        data['target'] * data['sd'] + data['mu'], float(data['capacity']))


def reference_loss(prediction, target, old, sigma, lower, harm_weight=.25):
    error = (prediction.clamp_min(lower)-target) / max(float(sigma), 1e-6)
    previous = (old.clamp_min(lower)-target) / max(float(sigma), 1e-6)
    square = error.square()
    return square.mean() + harm_weight * F.relu(square-previous.square()).mean()


def refinement_loss(prediction, target, reference, sigma, lower, tuning):
    error = (prediction.clamp_min(lower)-target) / max(float(sigma), 1e-6)
    smooth = tuning.get('smooth', 0.)
    absolute = error.abs() if smooth == 0 else torch.sqrt(error.square()+smooth**2)-smooth
    return (absolute.mean() + tuning['mse_weight'] * error.square().mean()
            + tuning['anchor_weight'] * ((prediction-reference)/max(float(sigma), 1e-6)).square().mean())


def admissible(prediction, data, reference, rmse_guard=1.003):
    target = data['target'].astype(np.float64)*data['sd']+data['mu']
    p, r = (physical_output(v.astype(np.float64), data) for v in (prediction, reference))
    if len(p) < 4:
        raise ValueError('Validation requires at least four chronological windows')
    ratios = [np.mean(abs(p[ix]-target[ix])) / max(np.mean(abs(r[ix]-target[ix])), 1e-12)
              for ix in np.array_split(np.arange(len(p)), 4)]
    ratio = np.sqrt(np.mean((p-target)**2) / max(np.mean((r-target)**2), 1e-16))
    return bool(max(ratios) <= 1.005+1e-12 and ratio <= rmse_guard+1e-12)


def select_readout(raw, data, reference, rmse_guard=1.003):
    choices = []
    for scale in (0., .25, .5, .75, 1., 1.5):
        prediction = reference + scale*(raw-reference)
        if scale == 0 or admissible(prediction, data, reference, rmse_guard):
            score = metrics(prediction, data)
            choices.append((score['mae_physical'], score['rmse_physical'], scale))
    return min(choices)


def fit_head(train, validation, config, tuning, regime, *, initial=None,
             reference=None, epochs=100, patience=10, device='cpu'):
    """Fit one configured head; no historical search folders are needed.

    An initial head optionally warm-starts reference fitting. Refinement starts
    from the fitted reference, uses EMA, and keeps the original 1.02 per-epoch
    RMSE guard. Final selection applies the stricter 1.003 guard separately.
    """
    xtr, xva = tensors(train, device), tensors(validation, device)
    sigma = max(float((train['target']-train['old']).std()), 1e-6)
    model = build_head(config, sigma, statistics(xtr), device)
    if initial is not None:
        model.load_state_dict(initial.state_dict(), strict=True)
    is_refinement = reference is not None
    if is_refinement:
        model.load_state_dict(reference.state_dict(), strict=True)
        model.epsilon = reference.epsilon * tuning.get('bound_factor', 1.)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    language, rest = [], []
    for name, p in model.named_parameters():
        (language if name.startswith('language.') or name == 'gates' else rest).append(p)
    optimizer = torch.optim.AdamW([
        dict(params=rest, lr=tuning['lr']),
        dict(params=language, lr=tuning['lr']*tuning.get('language_lr_factor', 1.))],
        weight_decay=tuning['weight_decay'])
    target = torch.as_tensor(train['target'], dtype=torch.float32, device=device)
    if target.shape != xtr[0].shape or not torch.isfinite(target).all():
        raise ValueError('Finite training targets with shape [N,16] required')
    lower = -float(train['mu'])/float(train['sd'])
    scales = torch.as_tensor(np.asarray(regime['scales'])[regime_bins(train, regime)],
                             dtype=torch.float32, device=device)
    reftr = (regime_readout(predict_head(reference, xtr), train, regime)
             if is_refinement else train['old'])
    refva = (regime_readout(predict_head(reference, xva), validation, regime)
             if is_refinement else validation['old'])
    reference_target = torch.as_tensor(reftr, dtype=torch.float32, device=device)

    def score(head):
        raw = predict_head(head, xva)
        if is_refinement:
            return select_readout(regime_readout(raw, validation, regime), validation, refva, 1.02)
        # Reference fitting selects by validation RMSE with an MAE guard.
        candidates = []
        old_mae = metrics(refva, validation)['mae_physical']
        for scale in (0., .125, .25, .5, .75, 1.):
            m = metrics(refva + scale*(raw-refva), validation)
            if m['mae_physical'] <= old_mae + 1e-12:
                candidates.append((m['rmse_physical'], m['mae_physical'], scale))
        return min(candidates)

    best, state, stale = score(ema), copy.deepcopy(ema.state_dict()), 0
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.as_tensor(np.random.default_rng(2026+epoch).permutation(len(target)), device=device)
        for begin in range(0, len(order), 128):
            ix = order[begin:begin+128]
            raw = model(*(v[ix] for v in xtr))[0]
            prediction = xtr[0][ix] + scales[ix]*(raw-xtr[0][ix]) if is_refinement else raw
            loss = (refinement_loss(prediction, target[ix], reference_target[ix], sigma, lower, tuning)
                    if is_refinement else reference_loss(raw, target[ix], xtr[0][ix], sigma, lower,
                                                        tuning.get('harm_weight', .25)))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite PFA loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if is_refinement:
                with torch.no_grad():
                    for p, q in zip(ema.parameters(), model.parameters()):
                        p.lerp_(q, .02)
        evaluated = ema if is_refinement else model
        current = score(evaluated)
        if current < best:
            best, state, stale = current, copy.deepcopy(evaluated.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    saved_config = dict(config, sigma_fraction=model.epsilon/sigma)
    return model, dict(state_dict={k: v.cpu() for k, v in state.items()},
                      config=saved_config, sigma=sigma)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'predict'))
    parser.add_argument('--station', required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/pfa.json'))
    parser.add_argument('--features', type=Path, required=True, help='directory containing split NPZ files')
    parser.add_argument('--output', type=Path, required=True, help='checkpoint/readout directory')
    parser.add_argument('--reference', type=Path, help='optional fitted reference-head bundle')
    parser.add_argument('--split', choices=('train','validation','test'), default='test')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    cfg = settings['stations'][args.station]
    folder = args.output
    if args.action == 'train':
        if (folder/'selection.json').exists():
            raise FileExistsError('Choose a new output directory; a fitted selection already exists')
        tr, va = [load_arrays(args.features/f'{s}.npz') for s in ('train','validation')]
        if any(float(tr[k]) != float(va[k]) for k in ('mu','sd','capacity')):
            raise ValueError('Train/validation must share training-derived target scaling')
        kwargs = dict(epochs=settings['epochs'], patience=settings['patience'], device=args.device)
        if args.reference:
            reference = load_head(args.reference, args.device)
            bundle = torch.load(args.reference, map_location='cpu', weights_only=True)
        else:
            reference, bundle = fit_head(tr, va, cfg['reference']['config'],
                cfg['reference']['tuning'], cfg['regime'], **kwargs)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, folder/'reference.pt')
        xva = tensors(va, args.device)
        refva = regime_readout(predict_head(reference, xva), va, cfg['regime'])
        records, predictions = [], []
        for i, head_cfg in enumerate(cfg['heads']):
            model, bundle = fit_head(tr, va, head_cfg['config'], head_cfg['tuning'],
                cfg['regime'], reference=reference, **kwargs)
            torch.save(bundle, folder/f'head{i}.pt')
            raw = regime_readout(predict_head(model, xva), va, cfg['regime'])
            score = select_readout(raw, va, refva)
            records.append(dict(file=f'head{i}.pt', scale=score[2]))
            predictions.append(raw)
        choices = [(refva, [])]
        choices += [(combine_heads(refva,[p],[r['scale']]),[r]) for p,r in zip(predictions,records)]
        if len(records)>1:
            choices.append((combine_heads(refva,predictions,[r['scale'] for r in records],cfg['reduction']),records))
        eligible = [(metrics(p,va)['mae_physical'], metrics(p,va)['rmse_physical'], rs)
                    for p,rs in choices if not rs or admissible(p,va,refva)]
        _,_,selected = min(eligible,key=lambda v:v[:2])
        selection = dict(regime=cfg['regime'], reduction=cfg['reduction'], members=selected)
        (folder/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    else:
        data = load_arrays(args.features/f'{args.split}.npz')
        values = tensors(data,args.device)
        selected = json.loads((folder/'selection.json').read_text())
        regime = selected['regime']
        ref = regime_readout(predict_head(load_head(folder/'reference.pt',args.device),values),data,regime)
        predictions = [regime_readout(predict_head(load_head(folder/r['file'],args.device),values),data,regime)
                       for r in selected['members']]
        pred = combine_heads(ref,predictions,[r['scale'] for r in selected['members']],selected['reduction'])
        np.savez_compressed(folder/f'{args.split}_predictions.npz',pred_std=pred,pred_physical=physical_output(pred,data))
        if 'target' in data:
            print(json.dumps(metrics(pred,data),indent=2))


if __name__ == '__main__':
    main()
