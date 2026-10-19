# ChronoSteer：与固定 PSRC 框架相关的针对性阅读卡

> Source coverage: Partial paper（HTML 方法与相关实验段落）
> Extraction confidence: High for inspected text; figures not visually audited
> Locator mode: structure-grounded
> Primary analytical lens: methods
> Secondary analytical lens: resource
> Context verification: Targeted external check
> Card completeness: Partial；不声称完成全论文图表审计

术语：LLM 指大型语言模型；TSFM 指时间序列基础模型；Chronos 是 TSFM，不与 ChronoSteer 混称；本项目 FM adapter 指输出端数值残差模块。

## 01 基本信息

[External] Chengsen Wang 等；arXiv:2505.10083v2。作者主页报告 TKDD 2026 接收。原文：https://arxiv.org/html/2505.10083v2 ；官方实现：https://github.com/ForestsKing/ChronoSteer 。阅读日：2026-09-28。DOI、卷期未核实。

## 02 一句话总结

[Analysis] 对本项目最有价值的是训练监督如何约束修正行为，而非增加网络模块。

## 03 研究问题

[Paper: Section 3.2] 研究文本如何指导数值预测，而不要求 LLM 直接输出精确数值。

## 04 背景与发展路径

[Analysis] 本次为针对性方法阅读，不是领域优先权检索；不能据此宣称我们的方案首创。

## 05 核心痛点

[Paper: Section 3.4.1] 指令—序列配对不足、自由文本不易稳定对齐。

## 06 核心思想

[Paper: Sections 3.3–3.5] 采用离散指令锚点、合成监督和后续真实序列微调。

## 07 方法概览

[External] 官方 README 的训练与部署流程：冻结 Chronos-Bolt，仅优化对齐模块；预计算文本嵌入可用于评估。来源：上述官方实现。

## 08 核心模块

[Paper: Section 3.3, Figure 1] 指令经投影后作为前缀 token 进入 TSFM。[Analysis] 这与本项目输出端 adapter 不同，不宜直接移植。

## 09 关键公式

[Analysis] 本项目保持 y_hat = y_PSRC + alpha * epsilon * tanh(f_theta(u))。此式为本项目对应关系，不是 ChronoSteer 原公式。

## 10 实验与证据链

[External] 官方 README 给出 MTSFBench-300 的能源、交通、天气评估。[Analysis] 尚未逐项复核全部图表及消融数值，不能据此为 PV 效果背书。

## 11 结论边界

[Analysis] 对齐监督有效不等于任意合成扰动都有效；合成后的输入必须足以解释监督目标。

## 12 作者明确承认的局限

[Paper: Section 3.5] 函数生成的方向监督不能充分表达真实事件的幅度、延迟和领域效应。

## 13 批判分析

[Analysis] 同一历史输入若对应多个合成未来，却没有可区分的条件，回归器只会折中。这是严格不加指令接口时的重要限制。

## 14 可迁移知识

[Analysis] 借鉴可控训练对，而非照抄九条文本指令。给已有 prior 添加受控偏差，目标仍保持训练集真实未来。

## 15 与当前工作的联系

[Analysis] 当前 formal/fm_campaign.py 固定 para；layers/pc_fra.py 中该变体不读取六个语义 token。不能将此 adapter 写成语言推理模块。

## 16 研究候选

[Hypothesis] Agent-derived research candidates：仅训练阶段做先验偏差增强。缘由：修正器可能依赖训练期先验偏差。Delta：改数据与训练日程，不改接口。如何验证：四站相同更新预算，对照原训练、普通噪声增强、状态条件增强；只在验证时间块选方案，报告 RMSE/MAE/MBE 及逐日配对差。可能失败：合成偏差不真实；增强过强导致忽略有效先验。创新状态：unverified。所有新结果属于历史 test 暴露后的补充实验，独立泛化确认另需未使用数据。
