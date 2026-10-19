# PSRC + Frozen Chronos 四站正式对比实验

仓库只保留已经锁定的正式实验链路。

## 实验规格

- 站点：DKASC Site31、DKASC Site1A、PVOD Station02、HKUST
- 分辨率：15 min
- 任务：`L96 -> H16`（过去 24 h 预测未来 4 h）
- 划分：按时间顺序 70%/15%/15%，不打乱
- 训练：100 epoch、patience 10、seed 2026
- 指标：RMSE、MAE、MSE、R2，使用各站原始目标量纲
- 输出：`outputs/final_h16/formal_comparison_subset_with_fm.xlsx`

PSRC 参数来自 `configs/best_params.json`。LSTM 和 GRU 使用
`configs/formal_experiment.json` 中明确记录的敏感性配置，不标作公开默认配置。

## 唯一入口

```powershell
python -m pip install -r requirements.txt
python run_experiment.py --dry-run
python run_experiment.py --resume
```

已有完整结果时只重建表：

```powershell
python run_experiment.py --report-only
```

从锁定流程重建 Frozen Chronos 校正：

```powershell
python run_experiment.py --resume --rebuild-fm
```

## 代码边界

物理–语言对齐独立实验（冻结原 PSRC+Chronos，GPT-2 六类模板；不覆盖正式结果）：

```powershell
python run_experiment.py --physical-language plan
python run_experiment.py --physical-language audit
python run_experiment.py --physical-language prepare
python run_experiment.py --physical-language train
```

仅 seed 2026，四站五臂同预算。test 必须另行显式启动；设计与全部命令见
`docs/physical_language_experiment.md`。

统一的补充证据实验（已知测试集上的后续分析，结果不覆盖正式表）：

```powershell
python run_experiment.py --evidence-study audit
python run_experiment.py --evidence-study train --dry-run
python run_experiment.py --evidence-study all
```

`all` 依次训练四站的无先验对照、三种 FM 的真实/打乱先验对照（共 28 次）；
所有验证集锁定完成后才评估 test。可断点续跑，产物在 `outputs/evidence_study_v1`。
具体协议与现有证据边界见 `docs/evidence_study.md`。

固定架构的先验鲁棒性补充实验（先完成上述 28 组）：

```powershell
python run_experiment.py --prior-robustness all --dry-run
python run_experiment.py --prior-robustness all
```

先用固定 checkpoint 做同状态先验替换诊断，再训练四站的 clean/noise/state
三组对照（共 12 次，每次完整 100 epoch）。新训练只评估 validation，
不覆盖正式表。产物在 `outputs/prior_robustness_v2`，详见 `docs/prior_robustness.md`。

- `run_experiment.py`：唯一用户入口
- `configs/formal_experiment.json`：正式实验与基线参数
- `configs/best_params.json`：PSRC 锁定参数档案
- `models/ours.py`：PSRC 主干
- `models/`：正式表中使用的基线
- `layers/`：模型共享层
- `formal/`：数据、训练、Frozen Chronos 校正、指标和表格生成
- `tests/`：正式链路与接口测试

验证：

```powershell
python -m pytest -q
python -m compileall -q formal models layers utils run_experiment.py
```
