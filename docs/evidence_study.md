# PSRC 论文证据整理与统一先验对照

日期：2026-09-28。当前 test 已参与既往分析，本轮属于补充实验，不是新盲测。
核心贡献定位为历史物理状态驱动的逐预测步残差校准；冻结 FM 是可替换扩展。

## 已有证据：使用匹配的 PSRC 层级对照

下表来自 `outputs/ablation_20260920/ablation_table_long.csv`。
结构消融均未接 FM，应与 PSRC（原表 w/o Chronos）比较，而不是直接与 Full Ours 比较。
数值残差对照保留校正功能，但去掉物理编码路径，因此不宣称严格相同有效参数量。

|站点|数值骨架 RMSE|纯数值残差 RMSE|PSRC RMSE|PSRC 相对纯数值残差的 MAE 变化|
|---|---:|---:|---:|---:|
|Site31|0.303760|0.306669|0.291144|-14.35%|
|Site1A|0.739439|0.748435|0.738819|+10.24%|
|PVOD02|1.706421|1.703111|1.675506|+0.34%|
|HKUST|1.996715|2.015389|1.843531|-18.73%|

PSRC 四站 RMSE 改善，但 MAE 不全改善。不能据此称物理语义在所有指标一致获益。
现有数值残差与物理 token 计算路径有结构差异；仍缺相同 PSRC 架构下的物理 token 打乱重训对照。
该项不由本轮 FM 打乱对照代替。本轮先把 FM 归因补齐，不能在结果出来前声称证据链完整。

## 旧 replacement 结果的正确口径

下表为已有 test 产物，属于协议未完全匹配的描述性结果，不是新的训练结果。
每格为“50% 固定混合 / 学习 adapter”的 RMSE；混合前对 FM 做了物理约束。
旧表 prior_only 已含 PSRC，因此这些数值不能叫 raw FM。

|站点|PSRC|Chronos-2|Moirai-2|TimesFM-3|
|---|---:|---:|---:|---:|
|Site31|0.291144|0.338077 / 0.289714|0.489144 / 0.290406|0.284829 / 0.289516|
|Site1A|0.738819|0.910081 / 0.729880|1.157803 / 0.731668|0.760174 / 0.729496|
|PVOD02|1.675506|1.956532 / 1.645699|2.477288 / 1.650257|1.669772 / 1.664210|
|HKUST|1.843531|2.419491 / 1.840210|3.005799 / 1.840576|1.913833 / 1.838683|

Site31 上 TimesFM 固定混合优于学习 adapter；不是 Site1A 上出现 60% 的误差骤降。
旧 Chronos 报告三站 beta=1，替换脚本却使用 beta=sigma_R，因此旧表不能称仅改变先验来源。
原结果与脚本保留作为历史记录，以本轮独立表取代错误的解释，不覆盖正式表。

## 本轮冻结协议

- 四站：Site31、Site1A、PVOD02、HKUST；L96→H16；15 min；seed=2026。
- 三个冻结先验：Chronos-2、Moirai-2、TimesFM-3；相同历史 Target 窗口。
- PSRC 权重固定，复用缓存，不重新下载或训练基础模型。
- 每站一个无先验 adapter，三种 FM 各两个 adapter（真实、打乱）；总计 28 次训练。
- 所有 adapter：同一 para 架构、hidden=96、dropout=0.1、100 epochs、patience=10。
- 同站统一学习率、weight decay、epsilon、损失 beta、正则系数和读出；参数见 protocol.json。
- beta 为全局标准化功率单位：Site31 为 0.03238818049430847，其他站为 1.0。
- 全部使用 validation 的解析 alpha 收缩；包括 HKUST，不再沿用其旧的另一种门控选择规则。
- 所有 28 个验证集锁定先完成，再评估 test；不按 test 挑选保留结果。
- 因协议在 test 已知后建立，本轮不能作为独立确认；新时间段确认仍需另外提供数据。

## 各行的含义

1. PSRC：原始冻结预测。
2. FM_physical_constraints_no_PSRC：FM 分布读出并经过容量/昼夜约束，不含 PSRC。
3. fixed_mix_50pct：0.5×PSRC + 0.5×FM，与旧 prior_only 对应。
4. adapter_none：先验槽直接等于 PSRC，差异为零；输入维度与参数数目完全保留。
5. adapter_shuffle：每个 split 内打乱物理约束后的 FM，再与该样本自己的 PSRC 混合。
6. adapter_real：对应样本的真实先验，按相同设置训练。

打乱发生在混合前，避免把别的样本的 PSRC 也一起打乱。这个阴性对照检验窗口对应信息，
仍可能包含状态失配效应；不单凭它宣称预训练知识的因果机制已被证明。
缓存保留统一 21 分位数接口；替换模型的尾部是常数外延，不能宣称均值读出完全等价于原生预测均值。

## 入口、输出及恢复

```powershell
python run_experiment.py --evidence-study audit
python run_experiment.py --evidence-study train --dry-run
python run_experiment.py --evidence-study all
```

`train` 仅训练/锁定；`test` 要求全部锁定存在。再次 `all` 自动复用完成的产物。
输出目录 `outputs/evidence_study_v1` 中保存协议与数据/权重/缓存/实现哈希、逐配置 checkpoint、
验证数组、alpha 锁定、测试数组及完整 comparison.csv/json；参数或输入变更必须另开版本。
报告 RMSE、MAE、MBE、R²及相对 PSRC 的 RMSE 百分比变化，不平均不同站点的绝对 RMSE。

## 论文写法的边界

主结果回答性能；匹配消融回答物理信息是否有用；FM 表回答不同先验在同一接口下如何迁移。
只有真实先验优于无先验/打乱对照，才将增益解释为与先验内容一致的证据。
如果固定混合获胜、某站 MAE 变差或 alpha 为零，均保留并讨论。
单种子结果为描述性证据；已有多种子 PSRC/FM 结果不能替代本轮各对照的不确定性估计。
