# Physical–language alignment：冻结预测器上的最小对照实验

## 要回答什么

保持现有 PSRC + Chronos 数值先验 + 已锁定 adapter 不变，检验：围绕六类
历史物理状态构建的语言表示，能否通过逐类对齐改善预测空间中的小幅残差校正。
本轮不替换旧模型、不更新正式表、不重新搜索数据段。

这是实际使用冻结 GPT-2 的语言分支，不把 Chronos 称作大语言模型。
语言是历史统计量的确定性表达，不增加新的观测信息；可能的收益来自
预训练表示与物理状态组织方式，而不是 LLM 获得了未来天气或推导了物理定律。

## 模型与接口

旧预测 `y_old` 原样保留。新增模块位于旧预测之后，不改其内部 PC-FRA：

1. 从同一个 96 步历史窗口提取原有六类描述符和冻结的六个 32 维 physical token。
2. 每类描述符按训练集三分位数转为 low/middle/high，生成一条固定英文模板。
3. 冻结 GPT-2 提取每条模板最后一个有效 token 的 768 维表示；按状态码查缓存。
4. 线性映射到 32 维，逐类门控融合：`U_k = P(S_k) + sigmoid(g_k) T(E_k)`。
5. 保留六类顺序，拼接旧预测、Chronos 先验、先验与 PSRC 的差异，经小 MLP 输出
   `delta = epsilon * tanh(head(...))`；最终 `y_new = y_old + a * delta`。

新增可训练参数 **43,670**，五个训练臂完全相同。主干、旧 adapter、物理编码器、
GPT-2 均冻结。新 head 的最后一层零初始化，确保初始输出逐位等于旧预测；
上游门控不同时置零，避免阻断学习。保存的 head checkpoint 与锁定的旧 checkpoint
共同构成完整模型，不能单独拿新 head 当完整 PSRC。

GPT-2 固定为 `gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e`，仅使用本地已缓存权重，
不调用在线 API、不生成自由文本。缓存的是冻结 embedding，投影和融合层仍参与训练。
冻结参数及首次文本编码成本不计入“新增可训练参数”，但应单独说明。

### 物理描述的边界

六类为 ramp、power/reference variability、irradiance–power coupling、thermal、
diurnal、volatility。旧实现的 6×6 张量包含 **22 个实际描述符，其余是填充**，
新模板不把填充值写成信息。阈值、有效列和尺度仅由训练窗口拟合。

- DKASC 的旧 cloud 角色与经过验证的太阳几何代理不一致；新分支将该类别标为
  unavailable 并屏蔽，不将其解释成可信云量。旧 PSRC 内部计算不变。
- HKUST 没有温度列，新 thermal 类别同样屏蔽。
- power/reference ratio 不称为无量纲晴空指数；不从相关性推断天气原因。
- 新模板忠实描述原有统计定义，不修改旧特征以追求更好结果。

## 对照组：四站 × 五臂 × 三个学习率 = 60 次训练

| 臂 | 新头的输入表示 | 逐类对齐损失 | 用途 |
|---|---|---|---|
| 原有 PSRC+Chronos | 无新头 | 无 | 不训练的冻结参照 |
| numeric | 同样历史描述符的固定稠密 Fourier 特征 | 无 | 同可训练参数量的非语言对照 |
| language_unaligned | 冻结预训练 GPT-2 | 无 | 加语言表示本身是否有用 |
| language_aligned | 冻结预训练 GPT-2 | 有 | 拟议方法 |
| language_shuffled | 错配窗口的预训练 GPT-2 表示 | 有 | 样本与语言状态是否必须对应 |
| random_lm_aligned | 同架构随机初始化、冻结的 GPT-2 | 有 | 是否真的需要语言预训练 |

numeric 使用连续描述符，语言使用三档量化描述符；因此语言没有额外输入信息，
但两种编码的信息精度并非完全相同。numeric 的固定映射为稠密 768 维，
不以大量零填充伪造参数匹配。

对齐损失是在**同一物理类别内**的对称、多正例对比损失。同状态码的多个窗口
都是正例，不把同状态误当负例；单一状态批次和缺失类别不产生对齐损失。
shuffle 保留六类联合结构，只错配整窗：训练集内做无自配对置换；验证/test
只从训练窗抽取错误描述，绝不从未来窗口获取 donor。

## 固定训练与选型规则

- 四站：DKASC Site31、DKASC Site1A、PVOD Station02、HKUST。
- 15 min，L96→H16，沿用原始时间顺序 70/15/15 划分；仅 seed **2026**。
- 每站均匀抽取最多 6,000 个训练窗口；验证集使用全部有效窗口。
- AdamW，batch 128，weight decay 0.001；lr 为 `1e-4 / 3e-4 / 1e-3`。
- 最多 100 epoch，patience 10；五臂相同搜索预算。
- 标准化空间 MSE；对齐臂加 `0.01 × alignment_loss`。
- `epsilon = 0.25 × std(y_train - y_old_train)`，只由训练残差确定。
- 先按验证集物理尺度 RMSE 选 epoch（含 epoch 0 的原预测），再按验证集选择
  `a ∈ {0, .25, .5, 1}`，要求 MAE 不劣于旧预测的 1.005 倍；最后选学习率。
- 负功率按旧口径裁到 0 后计算 RMSE/MAE/MBE/R² 与容量归一化指标。
- 必须同时保存未缩放结果与最终结果；`a=0` 明确记为“未获得可用增益”，不是语言有效。
- 旧预测器是在这些训练样本上拟合过的，训练残差不是 out-of-fold；本轮是冻结模型上的
  轻量验证，不把训练残差泛化性作为既有事实。

### 如何判断“语言分支有用”

主要比较 aligned 对旧预测和 numeric，其次比较 aligned 对 unaligned、shuffle、
random-LM。逐站列全，不只展示赢家。若只胜旧预测但不能胜 numeric，不将收益归于
语言；若不能胜 random-LM，不声称语言预训练有贡献；若不胜 unaligned，不声称
对齐损失有效。单 seed 的小差异是探索证据，不等于显著性或“不可替代”。

历史上这些 test 已被多轮查看，本版协议如实标记 `exploratory_after_prior_test_exposure`。
本轮选参不访问 test 指标，但这不能把旧 test 重新变成独立确认集。

## 统一启动入口

在项目根目录的现有 Python 环境执行。先看计划，再准备缓存和训练：

```powershell
python run_experiment.py --physical-language plan
python run_experiment.py --physical-language audit
python run_experiment.py --physical-language prepare
python run_experiment.py --physical-language train
```

`audit` 仅核验四站各 64 个验证窗口的旧预测重建，不训练。
`prepare` 生成训练/验证特征、训练阈值、预训练/随机 GPT-2 缓存，不训练也不评估 test。
`train` 默认跳过已完成且哈希一致的 trial，支持断点续跑，不自动评估 test。

全部 20 个站点×臂完成验证选型后，**需要单独执行**：

```powershell
python run_experiment.py --physical-language test
python run_experiment.py --physical-language report
```

test 解封时冻结全部 selection 哈希；之后本版本拒绝重新训练。
输出仅在 `outputs/physical_language_v1/`，包含协议、审计、每个 trial 的训练记录/
锁定参数/checkpoint、每臂 selection、逐站 test 数组及 `comparison.json`。
不会覆盖 `outputs/final_h16` 或现有正式 Excel。

## 实际保留的旧 adapter

精确路径位于 `configs/physical_language.json`：Site31 使用 round2 t09；
Site1A 使用 e100 重锁版 round1 t11；PVOD02 使用 round1 t11；
HKUST 使用后续重锁版 round1 t00。不是旧截图里的过期配置。
缓存准备阶段会核对完整验证集归档数组；test 阶段会核对当前正式报告中的旧模型指标。
重建不一致立即停止，不用新输出偷偷替换旧基准。
