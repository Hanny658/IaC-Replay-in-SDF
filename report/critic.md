先给结论：

**我认为这篇工作的整体新颖度是“中上到较强”，大约 7/10；但当前稿件对几个子机制的新颖性表述得过强。**
真正最有希望站住的 novelty 不是 “refractory rotation 本身”，也不是 “waking replay 本身”，而是：

> **利用当前 wake batch 的稀疏活动，在每一步动态找出对当前计算“不可见”的突触自由度，只允许 replay 在这些自由度里发生；再用 refractory rotation 主动扩大这个可 replay 的隔离通道，并让 optimizer state 也服从同一个 mask。**

我这轮检索里，**没有找到已经把这一整套东西做在一起的先例**。而你稿件现在的核心机制确实是这个：Eq. 6 根据当前 batch 的 pre/post activity 构造 replay mask，并把 optimizer state 也限制在 mask 内；随后用前一 batch 的使用历史强制 recently-active units 在下一次竞争中休息。 

但有几篇非常接近、而且 reviewer 很可能找到的工作，必须认真处理。

| 你稿件中的 claim / 组件                                                         | 我找到的最接近工作                                                                                                                                                                                                                                                                            |       重合程度 | 我对剩余 novelty 的判断                                                                                                                                                         |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Recently used units 下一 batch 不参加 WTA，refractory rotation**             | Kaski & Kohonen 1994 已经构造了会“自动使 winner inactive”的循环 WTA；Maeda & Miyajima 1999 更直接提出 refractory competitive learning：unit firing 后在一段时间内不能再次成为 winner。([ScienceDirect][1])                                                                                                            |      **高** | **单独作为算法 primitive：低，约 2/5。** 不能再说没有 ANN precedent。                                                                                                                      |
| **根据 unit 自身使用历史改变后续竞争，而不是由 task/input context 直接 gate**                 | 2006 SOM habituation/conscience 已经属于 history-dependent competitive suppression；更危险的是 Shen et al. AAAI 2024 的 continual-learning SNN：variable threshold 随历史 activation count 上升，过去经常激活的 neuron 后续更难再激活。([PubMed][2])                                                                  |     **很高** | **“use-history-dependent gating” 本身并不新。** 但你的一步 refractory 用途不同。                                                                                                         |
| **k-WTA + history-dependent selective activation 用于 continual learning** | Shen et al. 2024 直接做了 trace-based K-WTA + variable threshold 的 class-incremental learning；ICLR 2026 又有 RTK-WTA，用 trace-dependent activation + randomized top-k 做 lifelong learning。([AAAI Publications][3])                                                                          |      **高** | Rotation 若定位为“CL 的资源轮换机制”，novelty 被明显削弱；若定位为**扩大 isolated replay channel**，则仍然新。                                                                                         |
| **利用 inactive/sparse subnetworks 避免干扰**                                  | CLNP 用 sparsification 后的 inactive neurons/filters 学后续任务，并宣称对旧任务 zero deterioration；PackNet 也是 free parameters；Lässig et al. 2023 甚至明确“只更新被 WTA inactivate 的 neurons 的 incoming recurrent weights”。([arXiv][4])                                                                       |     **中高** | “inactive units 可安全塑性”不是新思想；**你的方向反过来：保护的是正在处理的 current wake computation，同时让旧 replay 在空闲方向学习**，这是重要区别。                                                                   |
| **activation/null-space 中的 non-interfering learning**                    | OGD 直接把 gradient 投到不会改变 previous-task output 的子空间；Adam-NSCL 在 previous-task feature covariance 的 null space 中更新；GPM 根据过去 activation subspaces 找正交更新。([Proceedings of Machine Learning Research][5])                                                                                  | **概念上很接近** | “存在 function-preserving update directions” 不新；你独特的可能是**不用存/估计全局 subspace，而是利用当前 batch 的精确 sparse support 得到一个离散、廉价、逐 batch 变化的 replay null channel**。                    |
| **k-WTA sparsity + neural activation null space**                        | Abbasi et al., CoLLAs 2022 的标题甚至就是 *Sparsity and Heterogeneous Dropout for Continual Learning in the Null Space of Neural Activations*；使用 k-winner activations + heterogeneous dropout 鼓励 task 间 non-overlapping patterns，并建立在 GPM 上。([Proceedings of Machine Learning Research][6]) | **非常值得警惕** | 这是我觉得 Related Work **必须新增**的一篇。不过其保护方向仍然是 “learn new while preserving old”；你的方案是 “replay old while preserving the live/current computation”。这个 temporal reversal 很好用来区分。 |
| **exact disjoint update + optimizer-state 问题**                           | 你已经引用的 McKee et al. 2026 FTN 实际比当前 Related Work 描述得更接近：disjoint k-WTA masks 给出 exactly disjoint gradient paths，而且论文明确提醒 optimizer state、weight decay、BN 等即使 gradient=0 也可能破坏 guarantee。([arXiv][7])                                                                                  |      **高** | **“exactness”和“optimizer leakage”本身不能作为强 novelty。** 但 “per-synapse current-batch replay mask + same-mask optimizer moments” 仍比较独特。                                       |
| **sparse optimizer 的 stale momentum**                                    | Bricken et al. 的 Sparse Distributed Memory 已明确提出 sparse model 中 momentum optimizer 的 “stale momentum” 问题；Shen 2024 也因此采用 SGD。([ResearchGate][8])                                                                                                                                     |      **高** | 不建议声称发现 optimizer-state leakage；可以说你给出了**针对 isolated replay 的 exact implementation requirement**。                                                                        |
| **wake 中 replay / 不等 offline night**                                     | Kubo et al. 2025 已明确组合 sleep-like consolidation 和 “awake rehearsal/replay”，后者在继续学习新任务时进行；SIESTA 则是明确 wake/sleep 两阶段。([arXiv][9])                                                                                                                                                     |      **中** | “awake replay” 不新；**不暂停 stream，同时 replay 被限制到对当前 computation invisible 的突触**才是你的强 claim。                                                                                 |
| **homeostatic signal 决定何时 replay**                                       | replay scheduling 本身已有系统工作，如 Klasson et al. TMLR 2023 学习何时 replay、哪些任务 replay。([OpenReview][10])                                                                                                                                                                                     |     **中低** | 我没有找到与你这种 **unit-level accumulated use pressure → currently-asleep population pressure → burst replay** 很接近的 ANN/CL 方法。这个具体 controller 我会给 **3.5–4/5**。                  |
| **fast/slow error relative-novelty gate**                                | concept-drift 领域早就使用 EWMA 的 online error monitoring；更值得注意的是一篇 **2026-08-27**、与你稿件只差 4 天的 sLoTh：维护 loss EMA，使用当前 loss / EMA 的 ratio 超阈值触发 novelty event。([ScienceDirect][11])                                                                                                         |     **中高** | fast-vs-slow EMA ratio 和 “unmastered” clause 有具体区别，但不适合把 relative-loss novelty detection 当主要 novelty。最好把 Aug 27 这篇作为 **concurrent work** 加进去。                            |
| **sleep-anchored synaptic down-selection**                               | Robinson et al. 2022 已经在 ANN continual learning 中联合研究 NREM replay、REM replay 和 synaptic downscaling。([arXiv][12])                                                                                                                                                                    |  **高（概念）** | 新意主要在你的**schedule interaction/negative finding**：continuous local、offline night、rare burst 下效果方向不同，而不是 down-selection 本身。                                                |
| **local sleep biological motivation**                                    | Driessen et al. 2026 的结果确实非常强：awake mice 中人为诱导局部 ON/OFF alternation，降低 local sleep pressure / synaptic-strength markers，并恢复 sleep-deprivation 下的 memory consolidation。([Nature][13])                                                                                                 |          — | 这是一个**非常及时且漂亮的 biological motivation**，但更应被称为 algorithmic analogy / inspiration，而不是证明你的算法等价于 biological local sleep。                                                     |

### 我认为当前稿件最容易被 reviewer 抓住的一句话

你 Related Work 现在写的是：已有 gating 方法都从 input identity 导出 gate，而你的 rotation 是由 unit 自身 use history 决定，并称 “to our knowledge … has no ANN precedent”。

**这一句以现在的形式，我认为守不住。**

甚至不需要争论 “SNN 算不算 ANN”：1994 的 Kaski–Kohonen 和 1999 的 Maeda–Miyajima 已经是经典人工 competitive neural networks，而且后者的 refractory rule 和你“刚 firing 的 winner 暂时不能继续成为 winner”在 primitive 层面相当接近。([ScienceDirect][1])

到了 continual learning 场景，Shen et al. 2024 更危险：他们也是 k-WTA，并且明确让 firing threshold 随 activation history 上升，使过去频繁激活的 neuron 后续更不容易激活，从而让较少使用的 neuron 获得学习机会。([AAAI Publications][14])

不过，这**并没有杀掉你们真正的故事**，因为二者的目的明显不同：

Shen 2024 是在做 **representation/resource allocation**——历史上用过的 neuron 不容易被新 task 再用；而你这里让刚刚参与 wake computation 的 neuron **下一 batch 对 wake 输入强制休息，却恰恰因此成为 replay 可以写入的 consolidable substrate**。你的稿件自己显示 rotation 把 isolated channel 从自然 silence 的约 0.2–0.3 扩大到约 0.53。

这其实是比 “我们发明了 refractory rotation” 更好的 novelty story。

---

### 我认为最强的 novelty 是 Eq. 6，但需要重新精准定位

你们的 Eq. 6 很值得保住。它不是一般意义上的 “sparse update”：

当前 wake batch \(x\) 先确定每层 awake set；replay update 只有在 **presynaptic endpoint 对当前输入 silent，或者 postsynaptic endpoint 当前 asleep** 时才能写入，而且 optimizer 的 first/second moment 也只在该 mask 内推进。

随后 Proposition 1 说明，若改动只落在 current input 的 pre-silent coordinates 上，那么无论 update magnitude 多大，对该 \(x\) 的输出都严格不变；post-asleep case 在 margin 条件下不变，而被 refractory mask 强制 suppression 的 unit 又可以把这个条件提升为 magnitude-independent。

这里 reviewer 最可能拿 OGD/GPM/Adam-NSCL/Abbasi/FTN 来对比。([Proceedings of Machine Learning Research][5])

但你们和它们之间有一个我认为**非常干净的 distinction**：

> **Prior null-space / parameter-isolation work generally protects old-task computations while learning the new task. Here the direction is reversed: the ongoing wake computation is protected, while an old-memory replay update is inserted into degrees of freedom that are instantaneously invisible to that current computation.**

而且这个 “null space” 不是通过 SVD、stored gradient bases、task subnetworks 或 task masks 得到的；它是由 **当前 batch 的 exact sparse support** 每步免费暴露出来的。

我认为这才是文章最应该重点宣称的 technical novelty。

甚至标题里的 “local sleep” 可以理解为这个机制的 biological interpretation，而不是 novelty 的数学定义。

---

### 还有一个我觉得你稿件目前低估了的 closest prior：Abbasi et al. 2022

这篇我强烈建议补 citation：

**Abbasi et al., “Sparsity and Heterogeneous Dropout for Continual Learning in the Null Space of Neural Activations,” CoLLAs 2022.** 它明确把 k-winner sparse activation、non-overlapping representations 和 activation null space 放到 continual learning 中。([Proceedings of Machine Learning Research][6])

乍一看，reviewer 很可能会觉得 “这不就是你们的核心概念吗？”

实际不是，但你最好替 reviewer 把区别解释掉：

**Abbasi/GPM：**
past activation subspace → constrain future/new-task gradient → protect the past.

**你们：**
present wake activity support → identify present-invisible coordinates → put replay/old-memory gradient there → protect the present while consolidating the past.

换句话说，两者像是**时间方向相反的 non-interference problem**。这个对比我觉得会让你们的 contribution 反而显得更清楚。

---

### “optimizer state confined inside the mask” 也需要降一点 novelty claim

这件事是对的，而且很重要，但不宜写成仿佛此前没人注意过。

FTN 2026 已经很明确指出：即使 inactive neuron 的 gradient 严格为零，optimizer state、decoupled weight decay、BN statistics 等非 gradient update 仍然能够破坏 structural guarantee。([ResearchGate][15])

Bricken et al. 在 sparse continual learner 中也已经专门讨论了 Adam/RMSProp 对 inactive neurons 的 stale momentum。([ResearchGate][8])

所以更强、更安全的说法不是：

> “we identify optimizer-state leakage”

而是：

> “we enforce the same current-input isolation constraint on both parameter updates and their per-synapse optimizer state, which is necessary for the replay channel to retain its exactness.”

这样既承认 prior，又保住你们 implementation/theory 的贡献。

---

### relative-novelty gate 这里有一个非常新的 concurrent work，需要马上处理

你稿件的 gate 是 fast error EMA / slow error EMA，加上一个 “unmastered” clause；其目的主要是决定 **什么时候 rotation**。

传统 concept drift 早就会用 exponentially weighted moving average 监控 streaming classifier error，所以 EMA-based shift detection 本身并不新。([ScienceDirect][11])

更重要的是，**Nagabhushana et al., arXiv:2608.26720，2026-08-27**，即你这份 Aug 31 draft 前四天，已经在 online continual learning 中写了非常相似的 novelty detector：

$$
\bar L_b=(1-\alpha)\bar L_{b-1}+\alpha L_b,\qquad
r_b=L_b/\bar L_b,
$$

\(r_b>\gamma\) 时触发 novelty event。([arXiv][16])

这当然**不是你们同一个 gate**：你是 fast EMA / slow EMA 的 relative dynamics，trigger 的对象是 rotation，而且还有 unmastered clause；它是 instantaneous loss / EMA，trigger teacher snapshot。但是 reviewer 若看到，会很难接受 “relative novelty via normalized loss dynamics” 是独立 headline novelty。

由于它只早四天，我会明确称 **concurrent work**，不需要把它写得像 priority problem，但应该 cite。

---

## 综合下来，我会这样给 novelty 打分

**Refractory rotation 本身：2/5。** 已有非常明确的早期 refractory WTA，以及近年的 history-dependent sparse CL。([Transactions Online][17])

**“rotation 用来主动制造 replay-safe substrate”：4/5。** 我没找到 prior 用 recently-active → next-step-off 这一机制，目的不是 representation allocation，而是为了增加 concurrent replay 的隔离带宽。

**Exactly isolated replay，按你们当前的精确定义：4–4.5/5。** “non-interfering update/null space” 很成熟，但“基于当前 batch sparse support、逐 synapse、用于同时进行旧记忆 replay，并保持当前 computation invariant”我没有检索到直接前例。

**Per-synapse optimizer-state isolation：3/5。** stale optimizer state 的问题已有 precedent，但你的 exact masking formulation 有价值。

**Unit-level homeostatic pressure → burst replay：3.5–4/5。** adaptive replay scheduling 并不新，但我没有找到很接近这种 use-dependent pressure of currently asleep units 的 controller。稿件中 pressure-triggered burst 与 surprise-triggered replay 的系统性分离也是挺好的 empirical contribution。

**Relative novelty gate：2.5/5。** 有长期 concept-drift precedent，又有 Aug 27 2026 的 concurrent loss-ratio gate。

**整体组合：约 4/5，或者我用 10 分制会给约 7/10。**

这类 paper 的 novelty 很常见的一种形态就是：**各个 primitive 都能找到祖先，但组合后解决了一个此前没有被解决得这么具体的问题。** 你这篇属于这种情况。

---

## 如果我是 reviewer，我会怎么评价这篇

我大概不会因为找到 refractory WTA prior 就认为 paper “not novel”。我反而会接受：

> refractory competition 本身 old；sparse isolation 本身 old；replay 本身 old；adaptive scheduling 本身也 old；
> **但是把 wake-active computation 和 memory consolidation 变成两个在同一网络、同一时间步上结构性不冲突的 plasticity channels，这个 operationalization 看起来是新的。**

而且 Figure 3 / 相关结果比 “91.3 vs 90.7” 那个 headline 数字对 novelty story 更有说服力：masked refractory replay 从 batch 256 降到 8 基本不掉，而 unmasked/silent mask/offline night 在小 batch 下明显退化。 这说明 isolation 不是一个装饰性的 biological metaphor，而是真的改变 replay 的统计性质。

相反，**91.3±0.3 vs 90.7±0.4 只有 0.6 point**，我不建议把“beats the night”本身当最大贡献。文章真正有意思的是 “为什么 micro-batch 可以工作”和 “为什么 consolidation 能嵌在 wake computation 的 unused degrees of freedom 中”。

此外，稿件自己已经很坦率地说明目前只有 MLP scale、MNIST/CIFAR、3 seeds，而且 feature-based CIFAR 上 BP+ER 42.6% 会超过 local learner 37.0%。  所以如果投较强 ML venue，**novelty 本身可能够，empirical generality 反而可能是更大的 reviewer 风险**。

---

## 我会直接改掉 Related Work 中的核心 novelty wording

当前的 “use history has no ANN precedent” 我建议删掉。

可以换成类似下面这一版，基本保住真正的贡献，同时比较难被 prior art 击穿：

> **Use-history-dependent suppression itself has precedents in classical competitive/WTA networks, including refractory and habituation mechanisms, and recent spiking continual-learning models use activity-history-dependent thresholds to diversify neuronal recruitment. Our contribution is different in purpose: recently wake-active units are suppressed for the next competition specifically to enlarge a dynamically reallocated replay channel. Replay updates—and their optimizer state—are then restricted to synapses whose modification is invisible to the ongoing wake computation. Thus, rotation is not introduced as a task-allocation mechanism, but as a mechanism for creating plastic degrees of freedom in which consolidation can proceed without a global offline phase.**

然后 null-space 那段我还会专门加一句：

> **Unlike prior null-space and parameter-isolation methods, which protect representations of previous tasks while learning the current one, our isolation reverses the protected direction: the current wake computation is held invariant while replay updates past memories through synapses exposed as inactive by the current batch.**

我认为这两段一换，**paper 的 novelty defense 会比现在明显强一个档次**。

最后还有一批我建议至少读/加进 Related Work 的文献：Kaski & Kohonen 1994、Maeda & Miyajima 1999、Rizzo & Chella 2006、GPM 2021、Abbasi et al. 2022、Shen et al. AAAI 2024、Shen et al. ICLR 2026，以及刚刚 8 月 27 日出现的 sLoTh；McKee/FTN 2026 虽然你已经 cite 了，但值得扩写，因为它和 “exact isolation + optimizer state” 的关系比稿件现在写出来的更近。([ScienceDirect][1])

**所以最终判断是：我不会把这篇判成“idea 已经有人做过”。但我会把它从“一个全新的 local-sleep/refractory principle”重新定位成“一个新的 concurrent, exactly isolated replay construction”。后一个 claim 经我这轮广搜以后，明显更能站得住。**

[1]: https://www.sciencedirect.com/science/article/abs/pii/S0893608005801546?utm_source=chatgpt.com "Winner-take-all networks for physiological models of competitive learning - ScienceDirect"
[2]: https://pubmed.ncbi.nlm.nih.gov/16722184/?utm_source=chatgpt.com "A comparison between habituation and conscience mechanism in self-organizing maps."
[3]: https://ojs.aaai.org/index.php/AAAI/article/view/27817?utm_source=chatgpt.com "Efficient Spiking Neural Networks with Sparse Selective Activation for Continual Learning | Proceedings of the AAAI Conference on Artificial Intelligence"
[4]: https://arxiv.org/abs/1903.04476?utm_source=chatgpt.com "Continual Learning via Neural Pruning"
[5]: https://proceedings.mlr.press/v108/farajtabar20a.html?utm_source=chatgpt.com "Orthogonal Gradient Descent for Continual Learning"
[6]: https://proceedings.mlr.press/v199/abbasi22a?utm_source=chatgpt.com "Sparsity and Heterogeneous Dropout for Continual Learning in the Null Space of Neural Activations"
[7]: https://arxiv.org/abs/2604.24637?utm_source=chatgpt.com "Cortex-Inspired Continual Learning: Unsupervised Instantiation and Recovery of Functional Task Networks"
[8]: https://www.researchgate.net/publication/369414013_Sparse_Distributed_Memory_is_a_Continual_Learner?utm_source=chatgpt.com "(PDF) Sparse Distributed Memory is a Continual Learner"
[9]: https://arxiv.org/abs/2508.14081?utm_source=chatgpt.com "Toward Lifelong Learning in Equilibrium Propagation: Sleep-like and Awake Rehearsal for Enhanced Stability"
[10]: https://openreview.net/forum?id=Q4aAITDgdP&utm_source=chatgpt.com "Learn the Time to Learn: Replay Scheduling in Continual Learning | OpenReview"
[11]: https://www.sciencedirect.com/science/article/pii/S0167865511002704?utm_source=chatgpt.com "Exponentially weighted moving average charts for detecting concept drift - ScienceDirect"
[12]: https://arxiv.org/abs/2209.05245?utm_source=chatgpt.com "Continual learning benefits from multiple sleep mechanisms: NREM, REM, and Synaptic Downscaling"
[13]: https://www.nature.com/articles/s41593-026-02318-9?utm_source=chatgpt.com "Induction of cortical on/off periods in awake mice fulfills sleep functions | Nature Neuroscience"
[14]: https://ojs.aaai.org/index.php/AAAI/article/download/27817/27664 "Efficient Spiking Neural Networks with Sparse Selective Activation for Continual Learning"
[15]: https://www.researchgate.net/publication/404249315_Cortex-Inspired_Continual_Learning_Unsupervised_Instantiation_and_Recovery_of_Functional_Task_Networks?utm_source=chatgpt.com "(PDF) Cortex-Inspired Continual Learning: Unsupervised Instantiation and Recovery of Functional Task Networks"
[16]: https://arxiv.org/abs/2608.26720?utm_source=chatgpt.com "Parameter Efficient Continual Learning for Sparse Event-Based Transformers"
[17]: https://globals.ieice.org/en_transactions/fundamentals/10.1587/e82-a_9_1825/_p?utm_source=chatgpt.com "Competitive Learning Methods with Refractory and Creative Approaches"
