import copy

import numpy as np
import pytest
import torch

from formal import pfa, physical_context
from formal.pfa_features import state_features
from formal.chronos_prior import readout, skew_readout, CHRONOS2_QUANTILES


def sample(n=12):
    rng = np.random.default_rng(42)
    old = rng.normal(1, .2, (n, 16)).astype('float32')
    return dict(old=old, prior=old+.1, disagreement=np.full_like(old,.1),
                physical=rng.normal(size=(n,6,32)).astype('float32'),
                language=rng.normal(size=(n,6,64)).astype('float32'),
                available=np.ones((n,6),dtype=bool), target=old+.05,
                mu=0., sd=1., capacity=3.)


def test_head_checkpoint_and_masking(tmp_path):
    data = sample()
    data['available'][:,2] = False
    x = pfa.tensors(data)
    cfg = dict(sigma_fraction=.25,gate=.8,temperature=.1)
    model = pfa.build_head(cfg,1.,pfa.statistics(x))
    assert sum(p.numel() for p in model.parameters()) == 10183
    torch.testing.assert_close(model(*x)[0],x[0])
    assert torch.count_nonzero(model(*x)[-1][...,2]) == 0
    with torch.no_grad():
        model.output.weight.fill_(.01)
    expected = pfa.predict_head(model,x)
    path = tmp_path/'head.pt'
    torch.save(dict(config=cfg,sigma=1.,state_dict=model.state_dict()),path)
    np.testing.assert_array_equal(pfa.predict_head(pfa.load_head(path),x),expected)
    x[-1].zero_()
    with pytest.raises(ValueError):
        model(*x)


def test_regime_and_ensemble_readout():
    data = sample(4)
    data['old'][:] = np.array([.2,.8,1.5,2.5])[:,None]
    raw = data['old']+.1
    regime = dict(edges=[.25,.6],scales=[.5,.125,1.])
    corrected = pfa.regime_readout(raw,data,regime)
    expected = data['old']+np.array([.5,.125,.125,1.])[:,None]*(raw-data['old'])
    np.testing.assert_allclose(corrected,expected)
    ref = data['old']
    np.testing.assert_allclose(pfa.combine_heads(ref,[raw,raw+.2],[1,.5]),
                              np.mean([raw,ref+.5*(raw+.2-ref)],axis=0))


def test_training_vocab_and_numeric_encoding_are_batch_independent():
    raw = np.random.default_rng(1).normal(size=(20,6,6))
    mask = np.array([1,1,1,0,1,1],dtype=bool)
    vocab = physical_context.fit_record(raw[:12],mask)
    before = copy.deepcopy(vocab)
    data = dict(raw=raw,role_valid=mask)
    full = state_features(data,vocab,'numeric')
    parts = np.concatenate([state_features(dict(raw=r,role_valid=mask),vocab,'numeric')
                            for r in (raw[:7],raw[7:])])
    np.testing.assert_array_equal(full,parts)
    assert full.shape == (20,6,64) and vocab == before


def test_loss_and_validation_guards():
    data = sample()
    ref = data['old']
    assert pfa.admissible(data['target'],data,ref)
    bad = data['target'].copy()
    bad[:3] += 10
    assert not pfa.admissible(bad,data,ref)
    assert pfa.select_readout(bad,data,ref)[2] == 0
    pred = torch.tensor([[.1,.3]],requires_grad=True)
    target, reference = torch.tensor([[.2,.1]]),torch.tensor([[.1,.1]])
    tuning=dict(mse_weight=.1,anchor_weight=.01,smooth=0.)
    loss=pfa.refinement_loss(pred,target,reference,.5,0.,tuning)
    error=(pred-target)/.5
    expected=error.abs().mean()+.1*error.square().mean()+.01*((pred-reference)/.5).square().mean()
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_train_reference_then_refine_smoke():
    torch.set_num_threads(2)
    train, val = sample(),sample(8)
    cfg=dict(sigma_fraction=.25,gate=.8,temperature=.1)
    tuning=dict(lr=.001,weight_decay=.001,mse_weight=.1,anchor_weight=.01,bound_factor=1.)
    regime=dict(edges=[],scales=[1.])
    ref,_=pfa.fit_head(train,val,cfg,tuning,regime,epochs=1)
    model,saved=pfa.fit_head(train,val,cfg,tuning,regime,reference=ref,epochs=1)
    assert saved['sigma'] > 0
    assert np.isfinite(pfa.predict_head(model,pfa.tensors(val))).all()


def test_chronos_readouts():
    q=np.broadcast_to(np.array(CHRONOS2_QUANTILES)[None,:,None],(2,len(CHRONOS2_QUANTILES),16)).copy()
    np.testing.assert_allclose(readout(q,np.array(CHRONOS2_QUANTILES),'mean_q_full'),.5)
    np.testing.assert_allclose(skew_readout(q,np.array(CHRONOS2_QUANTILES),.5),.5)
