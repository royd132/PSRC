# OpenTSLM：与固定 PSRC 框架相关的针对性阅读卡

> Source coverage: Partial paper（HTML 方法与相关实验段落）
> Extraction confidence: High for inspected text; figures not visually audited
> Locator mode: structure-grounded
> Primary analytical lens: methods
> Secondary analytical lens: resource
> Context verification: Targeted external check
> Card completeness: Partial；不声称完成全论文图表审计

术语：LLM 指大型语言模型；TSFM 指时间序列基础模型；Chronos 是 TSFM，不与 ChronoSteer 混称；本项目 FM adapter 指输出端数值残差模块。

## 01 基本信息

[External] Patrick Langer 等；arXiv:2510.02410v2。原文：https://arxiv.org/html/2510.02410v2 ；官方实现：https://github.com/OpenTSLM/OpenTSLM 。阅读日：2026-09-28。本卡固定 v2，不混用旧版摘要指标。

## 02 一句话总结

[Analysis] 对本项目最有价值的是语义、量纲与时间信息的可核对对应关系，而非生成一段看似合理的解释。

## 03 研究问题

[Paper: Section 3] 研究多变量时间序列与文字的联合理解。

## 04 背景与发展路径

[Analysis] 针对性方法阅读，未执行完整相关工作综述。

## 05 核心痛点

[Paper: Section 3.1] 归一化后的序列需要保留尺度与时间背景。

## 06 核心思想

[Paper: Sections 3.2–3.3] 比较 soft prompting 与 cross-attention 两种接入 LLM 的路线。

## 07 方法概览

[Paper: Section 4.1] 使用分阶段训练。[Analysis] 不等于我们需要新增语言生成器。

## 08 核心模块

[External] 官方仓库给出 Llama/Gemma 模型及推理入口。[Analysis] 医学任务模型不能直接视为 PV 预测基线。

## 09 关键公式

[Analysis] 对本项目，应核对每一输入的变量身份、单位、采样间隔和归一化逆变换；这些是接口契约，不是新增预测公式。

## 10 实验与证据链

[Paper: Sections 4.3–4.6] 提供分类、解释和扩展性评估。[Analysis] 本次未逐项复核所有图表数值；不从这些任务外推 PV RMSE 收益。

## 11 结论边界

[Analysis] 语言解释正确与预测校准正确是不同命题；前者不能替代后者的消融。

## 12 作者明确承认的局限

[Paper: Section 5, Limitations] 作者指出尺度编码、合成解释质量、答案正确性及仅考察一种预训练 TSFM 等限制。

## 13 批判分析

[Paper: Appendix A.2] 解释生成使用图像与已知标签。[Analysis] 可用于训练监督研究，但不能把依赖未来真值的解释作为 PV 推理输入。

## 14 可迁移知识

[Analysis] 先检查现有物理 token 是否表达声称的状态，再讨论是否需要更复杂对齐。输出端数值 adapter 不应改称跨模态语义推理。

## 15 与当前工作的联系

[Analysis] 不改 models/ours.py 的主干或 PSRC；不打开原本关闭的 FiLM。可先做同状态先验错配的评估，避免全局随机打乱只测出昼夜不相容。

## 16 研究候选

[Hypothesis] Agent-derived research candidates：同状态难负例对照。缘由：全局打乱可能产生过于容易识别的异常。Delta：只改对照配对；按历史日照状态、功率水平、时段配对，不用未来标签。如何验证：固定 checkpoint、多次预注册置换，比较真实与同状态错配先验的逐日误差差；保留原全局 shuffle。可能失败：近似配对仍分布不匹配；真实先验与替代先验本就高度相似。创新状态：unverified。若差异消失，应收窄信息贡献的结论，不能事后改配对追求胜出。
