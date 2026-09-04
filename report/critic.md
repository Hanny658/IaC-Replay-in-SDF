1. **“stateless SGD” 与实际代码不符**

手稿多次称其为“stateless heavy-ball SGD”或“carrying no state”，但代码默认 `momentum=0.9`，并持续保存一阶动量 `m[l]`：

- [cortex.py:57](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/models/cortex.py:57>)
- [cortex.py:499](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/models/cortex.py:499>)
- [run_seq.py:1466](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/run_seq.py:1466>)

实验配置没有把 momentum 设为 0，因此这些实验比较的是 masked Adam 与 **mask-confined momentum SGD**，不是“有状态 vs 无状态”。

受影响的核心表述包括 [preprint.tex:510](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:510>)、[preprint.tex:523](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:523>)、[preprint.tex:687](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:687>)。

建议二选一：

- 重新跑 `momentum=0` 的真正 SGD，最好补 `0` 对 `0.9` 的消融；
- 或统一改称 “heavy-ball/momentum SGD”，删除 “stateless/state-free” 以及“无状态是性能原因”的解释。

目前数据只能说明 Adam 与 momentum SGD 在该设置中表现不同，不能说明“optimizer state 本身造成差距”。

2. **“exact isolation”证明没有完整覆盖实际实现**

存在三个具体问题：

- 式 (4) 的维度不成立。按正文列向量约定和 `B^\ell` 的定义，应写成  
  `\((B^\ell)^\top[\cdots]\)`，而不是 `\(B^\ell[\cdots]\)`。代码使用的是行向量形式 `err @ B`，也印证需要转置：[preprint.tex:246](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:246>)、[cortex.py:377](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/models/cortex.py:377>)。
- Proposition 2 的扰动只包含 `\(g^\ell\Delta W a\)`，但实现还会更新 asleep unit 的 bias：[preprint.tex:297](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:297>)、[cortex.py:560](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/models/cortex.py:560>)、[run_seq.py:1662](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/run_seq.py:1662>)。margin condition 应加入 `\(\Delta b_i\)`，或 replay 时冻结 bias。
- 正文明确说 readout 保持 fully plastic，[preprint.tex:281](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:281>)；代码也对 readout 使用全 1 mask。因此“network output unchanged”及“present computation is held invariant”并不适用于实际完整网络，只适用于受 mask 保护的隐藏层部分。

建议把主张改成 **support-isolated hidden replay**：隐藏表示在特定条件下严格不变，而完整输出只达到经验上的近似不变；同时明确 readout 是例外。

3. **能耗常数引用错位**

[preprint.tex:226](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:226>) 将 0.9 pJ/event 和 4.6 pJ/MAC 引给 Merolla et al. 2014，但：

- 0.9 pJ 是 Horowitz 的 45 nm、32-bit floating-point add/accumulate proxy；
- 4.6 pJ 可由 0.9 pJ add + 3.7 pJ multiply 得到；
- Merolla 的 TrueNorth 论文报告的是约 26 pJ/synaptic event，而不是 0.9 pJ。

参见 [Horowitz 2014](https://doi.org/10.1109/ISSCC.2014.6757323) 和 [Merolla et al. 2014](https://www.sciencemagazinedigital.org/sciencemagazine/08_august_2014?pg=86)。

另外，当前代码只统计 synaptic operations，没有计入数据移动、膜电位更新、阈值/kWTA 比较和随机脉冲生成。因此 119 nJ 不能写成硬件实测意义上的 “runs at 119 nJ”。建议改为：

> an idealized 45-nm FP32 arithmetic-energy proxy estimates a 20.7× reduction relative to dense-rate inference

并优先报告原始 operation counts。ANN-to-SNN 转换和阈值校准也应补方法引用，例如 [Rueckauer et al.](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00682/full)。

4. **learning-rate appendix 不能支撑正文的 10% substrate 结论**

Appendix E 的 sweep 实际使用 `active_frac=0.05`：[run_seq.py:443](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/run_seq.py:443>)；但正文用它来支持 10% record substrate 下 Adam 已经处于最优点：[preprint.tex:534](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:534>)、[preprint.tex:1173](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:1173>)。

至少应在 Appendix 开头和 caption 明确“这是历史 5% substrate”。更稳妥的是补 10% 下 Adam/SGD learning-rate sweep，否则不能断言主表差异不是交叉超参数造成的。

## Reference 审核

当前共 42 个 `bibitem`，41 个被引用；没有 unresolved citation/reference。未发现明显虚构文献。尤其容易出错的 2025–2026 条目均真实存在，包括 [Driessen et al.](https://www.nature.com/articles/s41593-026-02318-9)、[Bazhenov et al.](https://arxiv.org/abs/2606.08447)、[Kubo et al.](https://arxiv.org/abs/2508.14081)、[McKee et al.](https://arxiv.org/abs/2604.24637) 和 [HiCL](https://ojs.aaai.org/index.php/AAAI/article/view/39411)。

需要处理的引用问题：

- `whittington2017` 未被引用：若 cortical local learner 确实受 predictive coding 启发，可在方法中引用；否则删除。
- HiCL 条目不完整，应补 `AAAI 40(27), 22518–22526` 和 DOI。
- Iyer 条目应补 article number `846219`。
- Abbasi、Kolen–Pollack 等 conference 条目的页码/DOI 完整度不一致，建议统一 bibliography 风格。
- reservoir writing 最好引用 Vitter 1985。
- `sorrenti2024` 的 arXiv 页面显示初稿提交于 2023 年 12 月；可按目标 venue 的格式决定写 2023 或 2024，但建议统一元数据：[arXiv 页面](https://arxiv.org/abs/2401.08623)。

生物学引用的“类比关系”也有几处过强：

- [preprint.tex:179](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:179>) 把 silent-mask 称为 Driessen tonic condition 的对应物并不准确；真正更接近 tonic reduction 的是 soft rotation/gain 0.5。
- “unit-level Process S”建议改成 “a unit-level analogue inspired by Process S”。
- fast/slow error ratio 不是 Hasselmo 2006 推导出的 ACh 模型，应写 “loosely ACh-inspired”。
- Saper 2005 支持 sleep–wake flip-flop 概念，但不直接支持算法中的固定 minimum on-duration。
- “SWR-like bursts”没有真正模拟 sharp-wave ripple 动力学；可改成 “burst replay”，或补引用并明确只是时间结构类比。

## Claim 强度和统计表述

以下几处建议直接降级：

- [preprint.tex:102](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:102>) 写 “three-way trade-off”，公式却有四项，应改为 four-way。
- “statistically neutral”没有给出检验或等效性区间；`27.6±0.4` vs `29.5±2.8` 应写 “inconclusive under the available seeds”。
- “four-fold variance reduction”计算不对：SD 从 1.1 到 0.3 是约 3.7 倍下降，variance ratio 则约 13.4。不要把 SD 当 variance。
- CIFAR substrate 消融只覆盖有限配置，而且 local 是 3 seeds、部分 BP cells 是 6 seeds，因此 “decomposes entirely”/“learning rule gives nothing away”过强。建议写 “the tested decomposition is consistent with activation sparsity accounting for the observed mean gap”。
- controller 的提升范围不是 3–5×：表中 raw CIFAR-100 是约 2.4–2.6×，feature 是约 4.3–4.8×。[preprint.tex:823](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:823>)
- “no dataset-specific knob at all”不准确，因为 learning rate 和部分 target scale 仍随数据集改变；可写 “without a dataset-specific decay coefficient”。
- “every fixed decay”应改成 “every tested fixed-decay setting”。
- Conclusion 的 “pays nothing on learnable i.i.d. streams”与正文报告的约 1.2-point static penalty 冲突：[preprint.tex:940](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:940>)。
- “Consolidation does not require an offline phase”建议限定为 “In the evaluated class-incremental MLP settings…”。
- held-out appendix 仅覆盖单个固定 10% split、3 seeds 和部分 MNIST 结论，所以 “every claim the paper rests on survives”明显过宽：[preprint.tex:1240](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:1240>)。

还有一处内部定义不一致：objective 将 \(R\) 定义为 samples × synapses，但 cost accounting 实际只报告样本数，[preprint.tex:104](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:104>)、[preprint.tex:465](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.tex:465>)。由于 night 和 local 使用的有效 substrate/mask 不同，最好确实补 synaptic-work comparison；否则将 \(R\) 明确定义为 replay samples。

## 用语与读者视角

实验日志式用语偏多，建议统一清理：

- “the project inherited” → “the previously used 5% condition”
- “the strongest schedule from our earlier ablations” → “the best-performing schedule in the tuning sweep”
- “In our first report” → 直接报告旧设置和新设置的差异
- “those numbers are the honest estimates” → “selection-independent replication estimates for this fixed split”
- “benchmark-chasing continual learning” → 删除价值判断，直接说明评价目标
- “wildly unstable” → “highly variable across seeds”
- “yields nothing” → “produced no detectable improvement”
- “fails outright” → 给出具体失败指标
- “cortex receives it as its retina” → “the frozen representation serves as the model input”
- “the sparsity dial, not the gate, is the cure” → “increasing the activity fraction reduced the static penalty more consistently than the tested gates”

Future Work 目前是一整段七项内容，阅读负担较大；建议改成 `enumerate` 或按 representation、control、scaling 三类组织。

## 最值得新增的图

优先级从高到低：

1. **完整训练数据流图**：waking batch、buffer、replay batch、\(W/B\)、awake/asleep masks、optimizer state，以及 fully-plastic readout 例外。现有 Figure 1 还不足以让读者判断 exactness 覆盖范围。
2. **controller × depth/class-count 图**：fixed/controller/skip 随深度或类别数变化的 accuracy，并同时画各层实际 \(\lambda_\ell\)。这是后半篇最重要的机制结论，目前只有表。
3. **CIFAR substrate decomposition waterfall/paired plot**：dense width → sparse wiring → kWTA → local learner。比文字更适合承载“gap 来源”的论证。
4. **energy–accuracy Pareto 图**：横轴 operation/energy proxy，纵轴 accuracy，不同 \(T_s\) 为点；标题中明确 idealized arithmetic proxy。
5. 可选：free readout 与 isolated hidden layers 的 prediction/code drift 分布，用于诚实展示“exact”和“near-exact”的边界。

## PDF 版面检查

我检查了全部 22 页。总体没有明显裁切，表格可读，但：

- PDF 第 8 页 Figure 3 的右侧对数刻度标签有重叠；
- 第 12 页 Figure 4 的直接文字标签互相挤压，建议使用 legend 或自动避让；
- 第 22 页 held-out table 单独占据一张大量留白的 float page；
- LaTeX log 还有 3 个 overfull `\hbox`，最大约 15.6 pt，以及一次 `[h]` 被改为 `[ht]` 的警告：[preprint.log](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/report/preprint.log>)。
