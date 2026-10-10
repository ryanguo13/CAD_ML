# ECG-CAD/OMI 项目模型调研简报（截至 2026-10-09）

本文为文献摘记与候选设计，不是最终实施接口。最终取舍、最新访问限制、PTB 预训练重叠处理和可执行的实验边界以 [模型调研与架构方案](../docs/模型调研与架构方案.md) 为准。尤其不能保证所有原生骨干都输出导联级向量或已校准质量分数，也不能在缺少可靠日期/外部数据时声称已能做时间外/外部验证。

本简报只讨论模型、表示学习和工程拼接方式。论文报告的 AUROC 不能直接解释为冠脉造影确诊 CAD/OMI 的概率；最终概率仍需在本项目按患者级、时间外和外部队列校准。

## 1. 先采用什么判断标准

现有 ECG foundation model（FM）在不同任务、标签来源和人群上的排名差异很大。ICLR 2026 **Benchmarking ECG Foundational Models: A Reality Check Across Clinical Tasks** 在 12 个公开数据集、26 个临床任务和 1,650 个目标上比较了 8 个 ECG FM：成人 ECG 解读通常由 ECGFounder、ECG-JEPA 和 ECG-CPC 领先；急性照护任务由 ECG-CPC 与 ECG-FM 较强；没有任何模型在所有任务均第一。该基准还报告紧凑的结构化状态空间模型（SSM）在多类时序任务中超过大型 Transformer，并且预训练模型相对从头训练可减少约 3.3–9 倍的标注量需求。结果必须按任务相似性解释，不能将通用 ECG 解读分数当作 OMI/CAD 外部证据。另一项独立研究 **BenchECG and xECG** 于 2026-09-14 发表于 npj Digital Medicine，两项基准不能混称。

项目应把“可复用的 ECG 表示”与“冠脉终点监督头”分开。对于本项目的约 2 万条造影配对 ECG，先比较冻结表示、线性/查询注意力头、轻量适配器和全量微调；只有在患者级内部验证稳定后，才解冻最后一至两层。RTX 5060 8 GB 不适合从头复现 90M 参数 FM 的大规模预训练，但适合缓存冻结 embedding、训练小头及在混合精度下进行最后层/LoRA 适配。

## 2. 主要公开 ECG foundation model

### ECG-FM（推荐作为 Transformer 主候选）

- 论文：McKeen et al., *ECG-FM: an open electrocardiogram foundation model*, JAMIA Open 2025，DOI [10.1093/jamiaopen/ooaf122](https://doi.org/10.1093/jamiaopen/ooaf122)，开放代码与权重：[bowang-lab/ECG-FM](https://github.com/bowang-lab/ECG-FM)，Hugging Face [wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm)。
- 规模/架构：作者正文给出约 **90.9M** 参数；wav2vec 2.0 风格的 4 个卷积下采样块（每块 256 通道、stride 2、kernel 2）+ BERT-Base 风格 12 层 Transformer（768 维、12 heads、FFN 3072），相对位置编码。部分基准封装按分类头口径统计 93.9M，工程记录应注明口径。
- 预训练：约 1.405M 条 5 秒片段，来源为 MIMIC-IV-ECG 与 PhysioNet 2021；WCR 混合目标包括 latent span masking/wav2vec 对比、CMSC 相邻时段对比和 Random Lead Masking（每导联独立 mask 概率 0.5）。作者报告 1.5M 12 导联 ECG 的数据集总数，具体训练样本/片段口径见论文。
- 输入和预处理：500 Hz；原始 ECG z-score；10 秒记录切成两个不重叠 5 秒片段。下游标签以每条 ECG 的二元标签为主，作者特别指出片段级信号与记录级标签可能不完全匹配。
- 公开结果：MIMIC/UHN 下游小到中等标注量时优于任务专用基线；示例 AF AUROC 0.996、LVEF≤40% AUROC 0.929。指标不能直接外推到造影证实的 OMI。
- 适用性和风险：WCR 同时包含局部形态和全局/序列对比，适合做主干或冻结特征提取器；模型较大，建议先缓存 768 维 embedding。z-score 会去除每条导联的绝对幅度尺度，疑似缺血的 ST 振幅可另行送入保留 mV 单位的局部分支。

### ECGFounder（强 CNN/Net1D 候选）

- 论文：Li et al., *An Electrocardiogram Foundation Model Built on over 10 Million Recordings*, NEJM AI 2025，DOI [10.1056/AIoa2401033](https://doi.org/10.1056/aioa2401033)，代码/权重：[PKUDigitalHealth/ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder)，Hugging Face [PKUDigitalHealth/ECGFounder](https://huggingface.co/PKUDigitalHealth/ECGFounder)。
- 数据和标签：HEEDB/Harvard-Emory ECG Database 中 10,771,552 条 ECG、1,818,247 名受试者、150 个由专家/设备报告解析出的标签；训练目标是带正标签增强的监督多标签损失，而非完全无标签 SSL。
- 架构：Net1D/RegNet 风格的一维卷积网络，逐 stage 增加宽度，使用 bottleneck、group convolution 和 channel attention；约 33.8M 参数。它不是 Transformer，适合作为与 ECG-FM 互补的局部形态/CNN 表示。
- 输入和预处理：论文描述统一重采样到 500 Hz、滤波并包含 50/60 Hz 陷波；仓库明确要求严格遵循 `dataset.py` 的 filtering 与 z-score 预处理。不同版本滤波细节需以仓库配置为准，不能自行替换。
- 优点/风险：推理和微调成本低于 ECG-FM，公开权重易于做冻结基准；预训练标签来自报告而非冠脉造影，CAD/OMI 需重新监督，且基准研究显示其在部分 outcome 任务弱于 ECG-CPC。

### HuBERT-ECG（大规模 SSL 备选）

- 论文预印本：Coppola et al., *HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications*, medRxiv DOI [10.1101/2024.11.14.24317328](https://doi.org/10.1101/2024.11.14.24317328)；代码：[Edoardo-BS/HuBERT-ECG](https://github.com/Edoardo-BS/HuBERT-ECG)，权重在 [Edoardo-Coppola Hugging Face](https://huggingface.co/Edoardo-Coppola)，许可 CC BY-NC 4.0。
- 规模：约 9.1M 条 12 导联 ECG、四国人群，三个模型大小；作者在 21 个独立数据集、约 1,600 个诊断/预后目标上评估。
- 方法：HuBERT 类离散目标/掩码自监督；可用 AutoModel 加载，并提供预训练和多任务微调脚本。
- 适用性：适合做外部表示比较和低标注消融，但模型约 97M 参数且预处理/版本较多；在 BenchECG 中并非所有任务均超过 S4/CNN。先做冻结 embedding，避免在 8GB 显卡全量微调。

### ST-MEM（空间-时间 MAE）

- 论文：Na et al., ICLR 2024，OpenReview [WcOohbsF4H](https://openreview.net/forum?id=WcOohbsF4H)；代码与 checkpoint：[bakqui/ST-MEM](https://github.com/bakqui/ST-MEM)。仓库声明 VUNO 保留权利，使用、修改、分发必须先按仓库许可证确认，不能默认当作 MIT。
- 架构：ViT-B/75；每个导联独立划分时间 patch，同时将同一时段的多导联 patch 放入空间-时间序列；lead embedding、`[SEP]` 和 lead-wise shared decoder 防止 decoder 直接从另一导联复制答案。75% 随机 patch mask，MAE MSE 重建。
- 数据/预处理：Chapman、Ningbo、CODE-15 的 188,480 条记录用于预训练；统一 250 Hz、0.67–40 Hz 带通、z-normalization。模型能适配 12、6 或 1 导联。
- 适用性：显式跨导联空间关系对多导联 ECG 有吸引力，可作为低成本第二表示或 teacher；但 MAE/z-score 可能弱化绝对 ST 振幅，且约 90M 参数，不适合在 RTX 5060 上重新预训练。其独立 ACS 证据仅来自后述 embedding 融合研究，不能视为 OMI 已验证模型。

### ECG-JEPA（潜空间掩码预测）

- 论文：Kim, *Learning general representation of 12-lead ECG with a joint-embedding predictive architecture*, Neural Networks 2026，DOI [10.1016/j.neunet.2026.109334](https://doi.org/10.1016/j.neunet.2026.109334)，代码/权重：[sehunfromdaegu/ECG_JEPA](https://github.com/sehunfromdaegu/ECG_JEPA)。
- 方法：student-teacher JEPA 在隐藏表示空间预测被 mask 的 patch，不重建含噪原始波形；同步跨导联 masking 和 Cross-Pattern Attention（CroPA）限制注意力只在同导联或同一时间位置交互。预训练约 180k 条开放 ECG；基准统计约 87.2M 参数。
- 适用性：潜空间目标旨在减少对噪声细节的拟合，在极少标注的结构/诊断任务可有较高 label efficiency。模型仍是大 Transformer。已读论文中的预训练来源为 Chapman/Ningbo/CODE，PTB-XL 是其下游数据；不能把 ECG-FM 的 PTB-XL 预训练污染结论套到所有模型，须逐权重版本审计。

### ECG-CPC / S4（本项目最值得保留的轻量 SSM 基线）

- 原始基准：Al-Masud et al., *Benchmarking ECG Foundational Models: A Reality Check Across Clinical Tasks*, ICLR 2026 OpenReview [xXRqWpt3Xr](https://openreview.net/forum?id=xXRqWpt3Xr)，[已读原文](https://arxiv.org/html/2509.25095v2)，代码/权重：[AI4HealthUOL/ecg-fm-benchmarking](https://github.com/AI4HealthUOL/ecg-fm-benchmarking)。仓库同步提供 ECG-CPC、ECGFounder、ECG-JEPA、ST-MEM、HuBERT-ECG、ECG-FM 等 checkpoint。
- 架构：4 个卷积编码层 + 4 个 S4 结构化状态空间层；作者训练的 ECG-CPC 约 **3.8M** 参数，S4-from-scratch 基线约 **2.2M**。ECG-CPC 在 HEEDB（约 10.7M）上以 contrastive predictive coding 训练，输入采样率 240 Hz。
- 结论：该基准中 ECG-CPC 在成人解读、心脏结构/功能和多项 outcome 类别总体强，且单张 NVIDIA L40 可训练约三周；冻结评估应使用 query-attention pooling，简单 mean+linear 会低估 token 表示质量。论文也明确指出模型/数据集/预训练目标纠缠，需按项目数据重新消融。
- 工程价值：3.8M 级参数、SSM 长程记忆和较低显存使其非常适合 RTX 5060 8GB；可作为独立主干或与 ECG-FM embedding 做后期融合。若重新训练，建议从冻结 checkpoint + 小头开始，暂不在 2 万标签集上从头训练。

### ECG-CLIP（截至日期的文本对齐模型，暂不作为即插即用主干）

- 2026 论文：Ko et al., *Development and external validation of a contrastive learning foundation model for ECG-based prediction of cardiovascular diseases and outcomes*, The Lancet Digital Health 2026，DOI [10.1016/j.landig.2026.101092](https://doi.org/10.1016/j.landig.2026.101092)。研究用超过 170 万条 Scripps ECG-报告对预训练，并在独立 MIMIC-IV ECG 上做 AMI、淀粉样变、HCM、AF 及结局任务；在只有 10 个阳性标签时，AMI AUC 报告 0.910。
- 官方仓库：[scripps-research-querlab/ecg_clip](https://github.com/scripps-research-querlab/ecg_clip)。README/代码定义 500 Hz、10 秒、8 导联（I、II、V1–V6）、80 个 500 点 token，12 层 Transformer、512 hidden，并采用“先 MAE、再 ECG-text CLIP”的两阶段训练；文本端为 Bio_ClinicalBERT，只有最后一层和 pooler 解冻。
- 重要限制：仓库的 `dataset.py` 是随机 tensor/占位文本 stub，没有提供真实训练数据读取或官方 checkpoint 下载；README 也没有公布 ECG 振幅归一化细节。其 AMI downstream 生成脚本以 ECG 报告关键词构造标签，不能等同于造影/肌钙蛋白确认的 AMI/OMI。应把 ECG-CLIP 当作文本对齐方法参考或未来复现方向，不应在本项目直接声称可用。

### D-BETA、MERL/KED 等文本-信号模型

- D-BETA：ICML 2025，论文代码/权重 [manhph2211/D-BETA](https://github.com/manhph2211/D-BETA)，[arXiv:2410.02131](https://arxiv.org/abs/2410.02131)。采用 masked ECG-text auto-encoder + 对比学习，示例 encoder 输出 768 维；可在 ECG 报告较丰富时作为跨模态研究基线。其文本监督和报告偏差会被下游学习，不能替代造影标签。
- MERL 与 ECGFM-KED 在 BenchECG 中分别以信号-文本弱监督对比学习、ResNet 信号-文本对比学习为代表；可用于低成本融合消融，但不应堆叠多个大模型而不做增益检验。

## 3. 自监督目标和时序模块的取舍

| 模块/目标 | 能学到什么 | 对本项目的价值 | 主要风险/控制 |
|---|---|---|---|
| 1D CNN/ResNet/Net1D | QRS、ST-T、局部波形和抗噪形态 | 小样本、显存友好；可保留 mV 幅度分支 | 远距离节律/跨导联关系有限；用 dilation 或轻量 S4 补足 |
| ViT/Transformer MAE（ST-MEM） | patch 级空间-时间关系 | 多导联结构、缺失导联鲁棒性 | 大显存；MSE 重建会关注噪声/幅度，需保留原始单位对照 |
| wav2vec/WCR（ECG-FM） | 局部上下文 + 相邻片段一致性 + lead dropout 鲁棒性 | 公开、成熟、适合冻结 embedding | 5 秒分片与记录级冠脉标签错配；预处理必须保持 500 Hz/z-score |
| JEPA（ECG-JEPA） | 潜空间语义、少重建噪声 | 极少标注下的表示学习 | 约 87M Transformer；预训练数据污染和权重版本需记录 |
| CPC + S4/SSM（ECG-CPC） | 长程预测、平滑状态和节律结构 | 8GB 显卡友好；基准跨类别强 | 240 Hz 与项目采样率不一致；需固定重采样和 query pooling |
| 文本-信号对比（D-BETA/ECG-CLIP/MERL/KED） | 报告语义对齐、零/少样本查询 | 把报告标签映射到候选异常的探索工具 | 文本报告噪声、同义词和标签泄漏；不能替代造影终点 |
| Mamba/现代 SSM（通用模块） | 线性复杂度长序列建模 | 可作为 ECG-CPC/S4 的工程替换或小型局部分支 | 截至本项目资料，ECG 公开基准主要直接支持 S4/CPC；Mamba 需在同一数据和切分上验证，不能假定优于 S4 |

Mamba 原始论文：[arXiv:2312.00752](https://arxiv.org/abs/2312.00752)；Mamba-2：[arXiv:2405.21060](https://arxiv.org/abs/2405.21060)。它们是通用序列模块，项目中应作为可插拔实验项，而不是未经验证的临床模型。

## 4. 推荐的可插拔缝合架构（模型层面）

建议定义统一的 `EncoderAdapter` 接口，使 `F_ECGFM`、`F_ECGCPC`、ECGFounder 或本地小型 S4 可以交换：输入为 `(B, 12, T)`、采样率/单位元数据和质量 mask；输出为 token 序列、全局 embedding、lead-wise embedding 和质量分数。第一版只实现两个主干：

1. **全局分支（可选其一）**：冻结 ECG-FM（768 维，5 秒窗口）或 ECG-CPC/S4（512 维，240 Hz）。对 10 秒 ECG 做两个/四个窗口推理，保留每窗 embedding 和一个 query-attention pooling；使用窗口平均只是对照。
2. **局部保幅分支（固定轻量）**：原始 mV 12 导联经带宽受控的 1D Conv/ResNet 或 TCN（例如 3 个 dilation stage）提取 1–2 秒局部 token；不要对该分支做逐记录 z-score，保留校准过的 mV 与 ST 段基线。另一路可输入滤波/z-score 信号，避免量纲和噪声互相干扰。
3. **跨导联融合**：每个局部 token 加 lead identity；用低头数（2–4 heads）的 cross-lead attention 或 gated pooling，将 I/II/aVR/aVL/aVF 与 V1–V6 的互补信息汇总。若显存紧张，先用按导联 attention pooling。
4. **后期门控融合**：将全局向量、局部向量、质量分数及可选 ECG 报告 embedding 做 LayerNorm 后拼接；用一个 2 层、256→128 的 gated MLP，门控权重同时输入 signal-quality mask。训练初始阶段冻结所有大主干，只训练融合器和头；随后只解冻 ECG-FM/SSM 最后 1–2 个 block 或使用 LoRA/adapter。
5. **多任务输出头**：共享融合表示，分别预测 (a) OMI、CTO/明确闭塞标签；(b) ACS/AMI 相关标签；(c) ST elevation/depression、T-wave inversion、节律/传导和伪差质量；(d) CAD 潜变量/长期终点（仅在有可靠标签时）。每个头是独立 sigmoid，使用 masked BCE/focal 或 class-balanced BCE；不要把“正常”当作 CAD 阴性金标准。
6. **校准和不确定性**：冻结训练集外的校准集，按任务做 temperature scaling 或 isotonic；报告 AUROC、AUPRC、敏感度/特异度、Brier、校准曲线和置信区间。用深度 ensemble（例如 3–5 个轻量头）或 MC dropout 形成模型不确定性，低质量 ECG 触发拒答/复核。

## 5. 本项目的最小实验矩阵

1. `raw-local-only`：轻量 1D CNN/TCN，作为幅度和数据管线 sanity check。
2. `ECG-FM-frozen + head`：使用 768 维 embedding + query-attention/MLP。
3. `ECG-CPC-frozen + head`：使用 SSM embedding + query-attention；这是 8GB 显卡上的主效率基线。
4. `ECG-FM + local-mV gated fusion` 与 `ECG-CPC + local-mV gated fusion`：检验全局语义和缺血形态是否互补。
5. 仅在上述结果稳定后做 `ECG-FM ⊕ ECG-CPC` 双 embedding 融合；MEDIC 院前 ACS 研究报告 ST-MEM+ECG-FM 拼接 AUROC 0.843±0.006、AUPRC 0.674±0.012，但它使用 5,813 条院前 ECG、10 次随机 80/20 切分，不能当作本项目造影外部验证。
6. 所有实验按患者分组，保留时间外测试和完全未参与预训练的外部测试；特别是 ECG-FM 公开权重已使用 PhysioNet2021（含 PTB-XL）和 MIMIC。无标签预训练见过测试信号也应透明披露，不能被标记为无污染外部集。其他模型逐 checkpoint 核对，勿一概判定污染。造影配对数据应按患者/记录去重，明确 OMI/CTO 标签来源。

## 6. 推荐落地顺序

第一阶段在 RTX 5060 上下载并核验 ECG-FM 与 ECG-CPC checkpoint、预处理和许可证；缓存训练/验证/测试 embedding。第二阶段训练局部保幅分支和门控融合器，同时保持两个大主干冻结。第三阶段仅解冻最后 block/adapter，比较冻结与微调收益。第四阶段才引入文本分支或超声分支；任何融合必须有同患者配对或清楚的时间窗，不能用跨患者配对制造“多模态”监督。

## 7. 关键原始来源

- ECG-FM：JAMIA Open 2025，DOI [10.1093/jamiaopen/ooaf122](https://doi.org/10.1093/jamiaopen/ooaf122)，[GitHub](https://github.com/bowang-lab/ECG-FM)。
- ECGFounder：NEJM AI 2025，DOI [10.1056/AIoa2401033](https://doi.org/10.1056/aioa2401033)，[GitHub](https://github.com/PKUDigitalHealth/ECGFounder)。
- HuBERT-ECG：medRxiv DOI [10.1101/2024.11.14.24317328](https://doi.org/10.1101/2024.11.14.24317328)，[GitHub](https://github.com/Edoardo-BS/HuBERT-ECG)。
- ST-MEM：ICLR 2024 OpenReview [WcOohbsF4H](https://openreview.net/forum?id=WcOohbsF4H)，[GitHub](https://github.com/bakqui/ST-MEM)。
- ECG-JEPA：Neural Networks 2026，DOI [10.1016/j.neunet.2026.109334](https://doi.org/10.1016/j.neunet.2026.109334)，[GitHub](https://github.com/sehunfromdaegu/ECG_JEPA)。
- ICLR2026 基准与 ECG-CPC：OpenReview [xXRqWpt3Xr](https://openreview.net/forum?id=xXRqWpt3Xr)，[原文](https://arxiv.org/html/2509.25095v2)，[代码](https://github.com/AI4HealthUOL/ecg-fm-benchmarking)。
- 独立的 BenchECG/xECG：npj Digital Medicine 2026，DOI [10.1038/s41746-026-03196-y](https://doi.org/10.1038/s41746-026-03196-y)，[代码/权重入口](https://github.com/dlaskalab/bench-xecg)。本次核实发表日期与官方仓库，未把其模型结果作为主架构证据。
- ECG-CLIP（Scripps）：Lancet Digital Health 2026，DOI [10.1016/j.landig.2026.101092](https://doi.org/10.1016/j.landig.2026.101092)，[GitHub](https://github.com/scripps-research-querlab/ecg_clip)。
- ECG-CLIP（日本，报告零样本版本，勿与上者混淆）：npj Digital Medicine 2025，DOI [10.1038/s41746-025-02074-3](https://doi.org/10.1038/s41746-025-02074-3)。
- ACS embedding 融合先导研究：DOI [10.3233/SHTI250902](https://doi.org/10.3233/SHTI250902)，开放全文 [PMC13382877](https://pmc.ncbi.nlm.nih.gov/articles/PMC13382877/)。
- D-BETA：ICML 2025，[arXiv:2410.02131](https://arxiv.org/abs/2410.02131)，[GitHub](https://github.com/manhph2211/D-BETA)。
- ECG-JEPA/ECG-CPC 预训练和规模研究：[AI4HealthUOL/ecg-pretraining-strategies](https://github.com/AI4HealthUOL/ecg-pretraining-strategies)。
- Mamba、Mamba-2 通用序列模块：[arXiv:2312.00752](https://arxiv.org/abs/2312.00752)、[arXiv:2405.21060](https://arxiv.org/abs/2405.21060)。

### 结论

模型架构应采用“**可插拔全局 FM/SSM + 保留 mV 的局部形态分支 + 跨导联门控融合 + 多任务独立头 + 后验校准**”。第一主干优先 ECG-FM 或 ECG-CPC 二选一，第二主干只用于对照和融合增益验证；不要默认同时堆叠 ECGFounder、ST-MEM、HuBERT、ECG-JEPA 等大型模型。ECG-CPC/S4 是显存和长程建模的低成本基线，ECG-FM 是公开 WCR Transformer 的稳妥主候选，而局部保幅分支负责弥补大多数 FM 预处理对绝对 ST/T 形态的损失。
