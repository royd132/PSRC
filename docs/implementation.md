# 实现接口

## 前向链路

`models/ours.py` 执行 RevIN、全局周期检索、局部增量 GRU、频域 Chebyshev 映射和多尺度跨变量建模。物理编码器从同一历史窗口提取六类描述符，经逐步检索后修正预测。逆 RevIN 得到训练集标准化坐标中的预测。

Chronos 提供 `[N,Q,16]` 分位数网格；选定读出经非负／训练最大功率裁剪和确定性日照掩码，再与骨干预测校准混合。数值适配器 `PcFraResidualAdapter` 据此生成预测。

历史描述符按训练段阈值形成类别及上下文记录，Qwen 编码固定文本模板；最后非 padding 位置的隐状态经固定 Gaussian 投影压缩为 64 维。数值对照使用相同记录的固定 Fourier 编码。

`QwenGuidedChronos` 由数值轨迹形成逐步查询，以余弦注意力检索物理／语言状态，生成有界修正。最终经过功率区间缩放、参考预测锚定、多个头组合、逆标准化和非负裁剪。

## PFA 特征文件

`formal.pfa.tensors` 接受以下数组，`N` 是按时间排列的窗口数。

| 键 | 形状 | 含义 |
| --- | --- | --- |
| old | [N,16] | Chronos 条件数值适配预测 |
| prior | [N,16] | 校准后 Chronos 先验 |
| disagreement | [N,16] | 校准先验减去物理修正骨干预测 |
| physical | [N,6,32] | 冻结物理编码器输出 |
| language | [N,6,64] | Qwen sketch 或同信息数值编码 |
| available | [N,6] | 可用类别掩码 |
| target | [N,16] | 训练／评估标签；纯推理不需要 |
| mu, sd | 标量 | 训练目标均值、标准差 |
| capacity | 标量 | 训练功率最大值，用于区间划分；不是额定容量 |

前三项预测与标签使用同一训练目标标准化坐标。六类 token 描述爬坡、参考功率比变化、辐照度耦合、温度、日周期和波动；波动窗口为 6、12、24 h。缺失测量角色按实现填零，语言检索屏蔽不可用类别。

所有分区共用训练段缩放统计和类别词表。`load_head` 加载包含 `state_dict`、`config`、`sigma` 的 PyTorch bundle；旧权重文件须结合其原始配置封装后使用。

## 训练与验证

骨干及物理修正使用预测损失、物理残差监督与骨干锚定。数值先验适配冻结骨干。PFA 参考头使用平方误差及误差增大惩罚；修正头从参考头初始化，采用绝对误差、可选平方误差和参考锚定，EMA 系数为 0.98。

修正头训练保留原实现的验证 RMSE 上限 `1.02 × reference`；最终读出选择收紧至 `1.003 × reference`，四个连续验证块的 MAE 上限为 `1.005 × reference`。两种阈值分别用于 checkpoint 选择和最终校准。最终读出包含参考预测作为回退。

公开命令采用固定配置重训，省去历史搜索与试验分支。论文实验使用单种子，适配开发前已查看相同测试时期，上游训练残差为样本内残差；新评估应独立设定时间留出段。

## 方法来源

全局检索和局部增量分支参考 [Global Temporal Retrieval](https://arxiv.org/abs/2602.10847) 和 [L-Drive](https://arxiv.org/abs/2605.17730)。窗口标准化使用 RevIN，时频协同相关工作包括 [TFCNet](https://doi.org/10.1016/j.energy.2025.139481)。冻结模型为 [Chronos-2](https://huggingface.co/amazon/chronos-2) 和 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，使用时遵循各自模型许可。
