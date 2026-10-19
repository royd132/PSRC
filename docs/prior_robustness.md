# Fixed-architecture prior robustness follow-up

Status: supplementary development after historical test exposure. No new LLM,
no changes to PSRC, multi-scale backbone, frozen FM, adapter dimensions or
placement. The para adapter still does not use FiLM. Inspired by paired
supervision in ChronoSteer and explicit state/scale correspondence in OpenTSLM;
this is neither implementation nor reproduction of their architectures.

## Unified entry

```powershell
python run_experiment.py --prior-robustness all --dry-run
python run_experiment.py --prior-robustness diagnose
python run_experiment.py --prior-robustness train
python run_experiment.py --prior-robustness all
```

Outputs: `outputs/prior_robustness_v2`. Protocol and input hashes are frozen
before execution. Original formal tables and v1 controls are not overwritten.

The incomplete robustness v1 is retained as an audit trail. Its Site1A
validation reconstruction failed because the rebuilt model did not restore
training-time PSRC requires_grad=False flags. A controlled replay with identical
weights/inputs reduced the maximum output difference from 3.71933e-5 to exactly
zero after restoring those flags. V2 matches the saved validation execution
flags, and separately preserves historical test reconstruction flags. Both
evaluate in no_grad; no parameter updates occur. Numerical tolerance is not
increased. See `outputs/prior_reconstruction_audit.py` for the localization check.

## Fixed-model diagnostic

Use all four stations and all three v1 real-prior adapters, with existing
validation-locked alpha. Evaluate real input and five fixed matched-replacement
seeds on validation and the already-exposed test. Matching uses only historical
month, six-hour bucket, registered operating state, 0.1-capacity power-level bins
and training-volatility terciles. Donor/recipient L96+H16 windows cannot overlap.
Donors come from the same split. This is an offline negative control, not a
deployable forecasting method; donor windows may occur later within that split.
Sampling is with replacement, not a marginal-preserving permutation. Unmatched
windows remain unchanged: report coverage and paired metrics restricted to
matched windows as well as full-split metrics. Recompute disagreement relative
to the recipient PSRC. Do not permute PSRC, historical features or target.

The real-input reconstruction must agree with saved v1 predictions before
interpreting the intervention. Save prediction arrays, donor indices and daily
origin-grouped errors. Overlapping forecast windows are not independent samples;
daily statistics here are descriptive, not a significance claim. A fixed-model
input intervention differs from v1's separately retrained shuffled controls.

## Equal-budget training

Chronos only; 4 stations x 3 arms = 12 runs. Same seed 2026, frozen PSRC, station
hyperparameters, training windows, batch size and optimizer. Each arm completes
100 epochs (patience 101 disables early termination) and restores its minimum
validation-MSE checkpoint. This differs from the formal 100/patience10 protocol
deliberately: the comparison measures equal update budgets. No extra pretraining.

- clean: unchanged training priors.
- noise: static independent noise on 50% of training windows.
- state: same selected windows/sign/amplitude; bias for night_low/regular,
  linear ramp error for ramp state, smooth hump for peak state.

Both augmented arms use pre-projection RMS = 0.03 capacity, halved in low state.
Each row's noise shape is RMS-normalized, so nominal corruption energy is equal.
Project selected rows into the existing capacity/daylight contract; report
actual post-projection RMS because clipping changes energy. Unselected rows are
bitwise unchanged. Labels remain real training futures; no fabricated weather
events or future-derived semantic inputs. Validation priors are never augmented.
The existing forward recomputes disagreement from the modified prior.

Report every validation result, calibrated alpha, epochs and best checkpoint;
do not automatically promote any arm or evaluate these newly trained arms on
test. Full-validation alpha fitting remains development, not independent proof.
Frozen PSRC residuals on training samples remain in-sample: this experiment
does not address the separate need for rolling out-of-fold residual supervision.

## Interpretation boundaries

State labels are registered numerical operating heuristics, not proven cloud
states or causal explanations. State counts and thresholds are audited. Strong
augmentation may erase useful FM information; state-dependent shapes may not
resemble actual forecast errors. If matched replacement removes the apparent
real-prior advantage, narrow the contribution claim rather than retune matching.
Independent untouched time periods remain necessary for fresh generalization
confirmation. Fair baseline selection must not use worst test configurations.
