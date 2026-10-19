# Semantic specificity experiment v2

Entry: `python run_experiment.py --semantic-study plan|train|test|report`.
Training is NOT started by `plan` or `--dry-run`. Resume is automatic within
the immutable protocol; do not edit the implementation or configuration during
a campaign. Existing formal checkpoints, tables and parameters are not replaced.

## Question and scope

Does physically grouped historical information provide a reproducible advantage
over the tested information/layout/retrieval alternatives? This is not a test
of universal irreplaceability or adherence to conservation laws. The existing
test splits have already been exposed; this is supplementary evidence, not a
new independent confirmatory test.

Four stations: DKASC Site31, DKASC Site1A, PVOD Station02, HKUST.
L96 -> H16, 15 minutes, existing chronological 70/15/15 split.
All normalization is fitted on training observations/windows only.
FM is disabled for all arms to isolate PSRC semantics.

## Eight arms

| Arm | Change | Controlled comparison |
| --- | --- | --- |
| physical | Full PSRC, retrained | Paired reference, not the archived seed2026 row |
| regroup_4101/4102/4103 | Fixed coordinate permutations of the same normalized 36 slots, reshaped to 6x6 | Information and parameter count identical; physical grouping broken |
| generic | Six equal chronological blocks of the same standardized historical inputs; each block has channel-averaged mean, population SD, min, max, last value and linear slope | Same 6x6 input shape and projection/attention parameters; alternative generic summary |
| shared_retrieval | Average horizon queries before cross-attention, broadcast the retrieved result | Same parameters and physical information; removes horizon-specific retrieval, retains horizon-specific output head |
| mlp_residual | Existing matched-size MLP residual, no statistics | Added capacity control |
| concat_stats | Existing matched-size direct concatenation of the same statistics | Retrieval/layout control |

Regrouping permutes coordinates WITHIN each sample, not samples or whole-token
order. Three fixed permutations are all reported; no validation/test selection
between permutations. Zero-padded physical slots are preserved. Generic tokens
have 36 populated statistics whereas physical tokens contain padding and missing
roles; same width is not identical information content. The generic comparison
tests one specified alternative, not every possible statistical representation.
MLP/concat parameter counts are approximately matched by existing width solvers;
actual trainable counts are saved in every lock for auditing.

## Budget and selection

Seeds: 2027, 2028, 2029, 2030, 2031, paired across every arm.
Per station/arm, test LR = 0.5x/1x/2x the archived station LR, with every candidate
trained on all five seeds. All other station settings, sampling, optimizer,
loss and batch size remain fixed. Every run uses 100 epochs / patience10 and
validation-only early stopping. Select ONE LR per station/arm by lowest mean
validation standardized MSE across five seeds; ties use the first candidate.
No per-seed LR selection. Total 480 training trials, followed by 160 locked test
passes. Historical full-model tuning is not erased by equal local LR budgets;
these results compare local robustness around the existing station settings.

Training and test are separate commands. `train` never evaluates test, and `test`
requires all 32 station/arm selections plus 160 selected checkpoints first.
The protocol fingerprints source, configuration, parameters and data to prevent
silent mixed-version resume. Only train/validation is used to select checkpoints.

## Reporting

Output: `outputs/semantic_specificity_v2/`.
`protocol.json`, per-trial `lock.json`/`checkpoint.pt`, per-arm `selection.json`,
selected per-seed `test.json`/`predictions.npz`, `comparison.csv/json`,
`summary.csv/json`. Summary reports mean/sample SD of RMSE/MAE/MBE/R2, paired
RMSE change and the number of seeds where full physical PSRC wins.
Report all arms, stations and seeds, including negative results. Seed repeats
measure training variability, not five independent datasets. Do not assign
IID significance to overlapping forecast windows or infer significance merely
from five wins. Temporal block-based uncertainty can be computed from saved
predictions after the campaign, under a separately specified analysis protocol.

If physical beats all three regroupings, generic statistics and shared retrieval
consistently, this supports the tested physical grouping/retrieval design.
If generic is comparable, state that the advantage is structured causal summaries
rather than unique physical semantics. If regrouping is comparable, do not claim
physical category grouping is essential. If shared retrieval is comparable, do
not claim horizon-specific attention is essential.

This primary campaign retrains the entire model; it does not hold the learned
backbone weights fixed. A frozen-common-backbone diagnostic or a new untouched
time period would be a separate experiment, not silently merged into this one.

## Launch

From the repository root:

```powershell
python run_experiment.py --semantic-study plan
python run_experiment.py --semantic-study train
# Only after all validation selections have been locked:
python run_experiment.py --semantic-study test
python run_experiment.py --semantic-study report
```
