# PSRC

光伏功率多步预测的物理状态引导时频框架。公开代码包含时频骨干、多尺度跨变量建模、物理语义残差修正、冻结 Chronos-2 数值适配，以及 Qwen/PFA 状态条件修正。

默认设置：15 min 采样，历史 96 步预测未来 16 步（24 h → 4 h）；四站独立训练，时间划分 70% / 15% / 15%，seed 2026。17 个比较模型共用数据窗口和指标实现。

## 安装

Python 3.10+，先安装适合设备的 PyTorch，再执行：

```bash
python -m pip install -r requirements.txt
# 使用 Chronos / Qwen 时另装：
python -m pip install -r requirements-foundation.txt
```

基础模型首次运行从 Hugging Face 获取。Qwen 固定 revision；建议使用 GPU 提取基础模型特征。核心测试支持 CPU，无需下载大模型。

## 数据

数据不包含在仓库内。默认 CSV 路径在 `configs/formal_experiment.json`：

| 站点标识 | CSV |
| --- | --- |
| dkasc_site31 | data/DKASC_site31_15min.csv |
| dkasc_site1a | data/DKASC_site1a_15min.csv |
| pvod_station02 | data/PVOD_station02_15min.csv |
| hkust | data/HKUST_UG_Hall4_15min_causal_physics.csv |

CSV 包含时间列（如 `date`）、功率列 `Target` 和数值型历史协变量。可选 `TargetObserved` 标记真实观测；确定性时间和太阳位置特征参与物理状态构造。按站点整理 15 min 时间轴并保留缺测标记。加载器排除跨断点窗口，标准化统计只由训练段估计。

原始来源：[DKASC](https://www.dkasolarcentre.com.au/download?location=alice-springs)、[PVOD](https://doi.org/10.11922/sciencedb.01094)、[HKUST](https://doi.org/10.5061/dryad.m37pvmd99)。项目标识 `pvod_station02` 对应论文 PVOD02（原始数据 station07）。

## 运行

在仓库根目录执行。先训练骨干及物理修正部分：

```bash
python run_experiment.py --dry-run
python run_experiment.py --station dkasc_site31 --model psrc --resume --require-artifacts
# 全部四站及比较模型：
python run_experiment.py --resume --require-artifacts
```

单站 PFA 流程：

```bash
python -m formal.foundation --station dkasc_site31
python -m formal.pfa_features --station dkasc_site31 --split train --output outputs/features/dkasc_site31
python -m formal.pfa_features --station dkasc_site31 --split validation --output outputs/features/dkasc_site31
python -m formal.pfa train --station dkasc_site31 --features outputs/features/dkasc_site31 --output outputs/pfa/dkasc_site31
python -m formal.pfa_features --station dkasc_site31 --split test --output outputs/features/dkasc_site31
python -m formal.pfa predict --station dkasc_site31 --features outputs/features/dkasc_site31 --output outputs/pfa/dkasc_site31
```

其他站点替换站点名和输出目录。数值状态对照使用 `pfa_features --encoding numeric`，放入独立目录，并在所有分区采用相同编码。必须先准备训练特征，以训练段阈值拟合类别词表。训练命令只读取训练和验证文件。

`configs/pfa.json` 保留最终采用的先验、参考头、修正头和区间参数。`pfa train --reference reference.pt` 可从已有参考头继续 MAE 修正；未指定时按采用配置拟合新的参考头。公开版本省去历史超参搜索、权重和缓存，固定配置重训不等同逐轮重放论文实验，不保证新训练与已报告结果逐位一致。

## 代码结构

- `models/ours.py`：全局／局部时频骨干和物理修正。
- `layers/`：多尺度、物理 token、残差修正及基线共享层。
- `formal/data.py`、`engine.py`、`metrics.py`：数据、训练、评估。
- `formal/chronos_prior.py`、`foundation.py`：Chronos 先验及数值适配。
- `formal/physical_context.py`、`pfa_features.py`：历史状态、词表与 Qwen 编码。
- `models/qwen_guidance.py`、`physical_foundation.py`、`formal/pfa.py`：PFA 头、损失和最终读出。

张量坐标和训练细节见 [实现说明](docs/implementation.md)。旧调参脚本、过程记录、论文生成脚本、实验输出和权重不属于公开源码。

## 验证

```bash
python -m pytest -q
```

测试使用合成数据，覆盖时间划分、模型接口、物理修正、PFA 编码、损失、校准读出和 checkpoint。发布整理时还对照了四站各 128 个已保存样本，PFA 标准化预测最大差异约 `1.9e-7`。
