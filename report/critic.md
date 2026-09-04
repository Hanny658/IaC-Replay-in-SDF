我把这个 15 页版本完整看了一遍，也对照了 **CL4FMAgents 2026 的最新 CFP**，并重新核查了几篇关键/最新 reference。先给结论：

> **这个版本已经明显从“有趣但理论口径没收住的 preprint”，变成了一篇相当适合 CL4FMAgents 的 workshop paper。**
>
> 叙事、ablation、理论边界意识都比上一版成熟很多。**我会建议投。**
>
> 但提交前我认为还有 **2 个 P0 级问题必须修**：一个是我发现的 **replay cost/headline 数值不一致**，另一个是 **Proposition 1 与 plastic readout 之间仍然存在直接逻辑冲突**。前者尤其危险，因为 reviewer 一旦自己算出来，会直接伤害对整篇实验可信度的判断。

按现在的状态，我会大致这样打：

| 维度                           |                          评价 |
| ---------------------------- | --------------------------: |
| Workshop fit                 |                  **8.5/10** |
| 核心 novelty                   |                    **8/10** |
| Narrative                    |                    **8/10** |
| Mathematical rigor           |      **6.5/10 → 修完可到 8/10** |
| Experimental design          |                    **8/10** |
| Reproducibility / statistics |                    **7/10** |
| References                   |                  **8.5/10** |
| 当前总体                         | **Weak Accept / Accept 边缘** |
| 修完下面 P0/P1                   |          **比较有竞争力的 Accept** |

而且 venue 本身其实比我原先想象的还合适。官方 CFP 明确把 **memory/knowledge consolidation、online/test-time/streaming adaptation、catastrophic forgetting、theory and foundations** 都列为 scope，并且 regular paper 正好允许 **8 页正文，不计 references 和 appendix**；你的正文也是 8 页。([NeurIPS 2026 Workshop][1])

---

# 1. 整体叙事：现在已经比较成立了

新标题：

> **Inference as Consolidation: Continual Learning with No Offline Phase by Replay in the Silent Degrees of Freedom**

比上一版好很多。

现在从 Abstract 到 Introduction 的逻辑链条很清楚：

$$
\text{agent cannot conveniently pause}
\rightarrow
\text{offline replay / interleaved replay both有代价}
\rightarrow
\text{local sleep provides biological motivation}
$$

然后进入核心 inversion：

$$
\boxed{
\text{protect the current computation}
\;\text{rather than}\;
\text{protect past parameters}
}
$$

再通过 sparse support 得到当前 batch 的 silent degrees of freedom。Abstract 现在也已经主动区分了 **exact for silent-presynaptic/suppressed units** 与 **near-exact elsewhere**，这比上一版“everything exact”的措辞稳健得多。

Introduction 对 contribution 的拆分也明显改善：isolation、rotation、trigger、adaptive decay/depth，且明确承认实验是 small-scale mechanism study，而不是 benchmark chasing。

这一点对 workshop 尤其重要，因为 CL4FMAgents 官方自己说 scope 是从 theory/algorithms 到 large-scale systems，**并不要求一定跑 LLM/FMs**。([NeurIPS 2026 Workshop][1])

## 但有一个叙事上的小风险：Section 5 有点像“第二篇 paper”

前四个 section 的故事非常统一：

> silent degrees → isolated replay → refractory rotation → timing/control.

到了 Section 5 突然变成：

> fixed decay is bad → self-referenced decay controller → depth → skip connections.

这个内容本身其实不错，甚至是现在稿子里很有趣的新结果之一：同一个 \(\rho=0.25\) controller 在不同 dataset 上自动形成不同数量级的 decay，并且深层网络明显比固定 \(\lambda\) 稳。

问题只是它目前看起来像一个并列的第四大贡献，而不是前面机制自然延伸出的东西。

我建议把 Section 5 的 framing 从：

> **One constant instead of per-dataset decay tuning, and depth**

稍微改成：

> **Scaling the mechanism: self-referenced decay and depth**

然后明确说：

> isolation/rotation 解决的是 **where and when to consolidate**；
> adaptive decay 解决的是同一个 design principle 在尺度变化下的 **how strongly to update**。

这样你的统一原则就变成：

$$
\boxed{
\text{all control signals are referenced to the learner's own current state}
}
$$

novelty 是 fast/slow ratio，homeostasis 是 own usage，decay 是 learning-drive/weight ratio。

这会让 Section 5 不再像额外塞进来的实验。

---

# 2. Workshop framing 合适，但不要过度“Agent-wash”

你开头：

> “An agent that keeps learning in deployment cannot pause to consolidate...”

对于这个 workshop 很自然。

但建议把 **cannot pause** 改成类似：

> “may not be able to afford dedicated consolidation downtime”

因为现实中的 agent 并非逻辑上“不能”暂停。一个挑剔 reviewer 很容易说服务器维护、asynchronous replica、background learner 都可以做 offline consolidation。

另外 Introduction 里：

> “The brain consolidates while it infers, in circuits the current computation is not using.”

我建议也弱一点。

Driessen 2026 确实强力支持的是：在 awake mice 中人为诱导局部 ON/OFF activity 可以降低 local sleep pressure、改变 synaptic markers，并恢复 memory consolidation；而相同总体 firing-rate reduction 的 tonic inhibition 不行。([Nature][2])

但它**没有直接证明**：

> biological consolidation is specifically occurring in circuits unused by the concurrent computation.

因此更安全的句子是：

> “Recent causal evidence shows that some core functions normally associated with sleep can be discharged locally during wakefulness.”

然后下一句再说：

> “We take this as an algorithmic motivation to ask whether unused computational degrees of freedom can similarly host consolidation.”

这样 biology → algorithm 的 inferential leap 就标得非常清楚。

---

# 3. 一个我认为必须马上修的实验数字问题：**1.2× replay cost 对不上 headline configuration**

这是这一版里我最担心的地方。

正文 headline configuration 是：

> one isolated replay micro-batch beside **every waking batch**，得到
> \(92.7\pm0.3\%\)。

Table 1 又明确说 full system 是：

> **one replay micro-batch of 16 per waking batch**。

但是 Appendix C 的 cost accounting 是：

$$
R_{\text{night}}
=
480\times256
\approx1.23\times 10^5
$$

以及

$$
R_{\text{local}}
=
18,760\times8
\approx1.50\times10^5,
$$

然后由此得到：

$$
R_{\text{local}}\approx1.2R_{\text{night}}.
$$



问题在于，**92.7 的 default 是 batch 16，不是 batch 8**。

因此真正对应 headline configuration 的 replay sample cost 是：

$$
18,760\times16=300,160.
$$

与 night 比：

$$
\frac{300,160}{480\times256}
\approx2.44.
$$

所以：

$$
\boxed{
92.7\%\text{ 的配置是约 }2.4\times\text{ night samples，而不是 }1.2\times.
}
$$

而 batch 8 的结果，根据 Figure 2 / 正文，是大约：

$$
91.8\pm0.3\%.
$$

正文自己写了 batch size 从 256 降到 8 时：

$$
92.3\rightarrow91.8.
$$



### 这会影响三处 headline

Abstract 现在说：

> 92.7 ± 0.3% ... at 1.2× its replay samples. 

Results 也说：

> 92.7 ... at 1.2× its replay samples. 

Discussion 又重复：

> above the strongest offline-rehearsal schedule at 1.2× its replay samples. 

**这三个必须统一修。**

最简单有两种写法：

> **92.7 ± 0.3% at 2.4× the night’s replay samples; at matched ≈1.2× replay cost, an 8-sample micro-batch still reaches 91.8 ± 0.3%.**

其实这个结果依然很好。

甚至我觉得故事更科学：

$$
\text{accuracy–cost frontier}
$$

比把一个 record point 强行绑定 1.2× 更有意思。

如果你可以在 deadline 前找到一个 cadence/batch 组合达到接近 92.7 且成本约 1.2×，当然最好；但**不要保留现在这个数字组合**。

---

# 4. 数学部分：上一版最大的 Eq. (5) transpose 已经修好了

这是值得肯定的。

现在你明确写：

$$
B^\ell\in
\mathbb R^{n_{\ell+1}\times n_\ell},
$$

backward signal 用：

$$
(B^\ell)^\top\epsilon^{\ell+1},
$$

因此：

$$
\Delta B^{\ell-1}
\propto
\epsilon^\ell(a^{\ell-1})^\top
$$

维度为：

$$
n_\ell\times n_{\ell-1},
$$

和 \(B^{\ell-1}\) 一致。

所以我上一版提到的 B transpose/dimensionality 问题已经解决了。

---

# 5. 但 Proposition 1 仍然有一个非常明确的逻辑矛盾

现在正文先写：

> “The readout row stays plastic ... the guarantees below concern the hidden computation.”

紧接着 Proposition 1 又写：

> “Then every hidden activity on \(x\), **hence the network output, is unchanged** for any magnitude of \(\Delta\).”



这两句话仍然不能同时成立。

Appendix A 自己其实已经承认：

> “The readout row is excluded from both propositions because it is deliberately kept plastic.” 

而且你的 empirical exactness measurement 更直接：

* hidden code changed：0.36%
* waking prediction changed：0.27%
* **readout isolated 后 prediction change 只有 0.001%**



这几乎已经证明：

$$
\text{plastic readout}
$$

正是 output invariance 不成立的主要来源。

### Proposition 1 应该直接改成

$$
\boxed{
\text{all hidden activities on }X\text{ are unchanged}
}
$$

然后补一句：

> If the readout is held fixed, or if its replay update is itself orthogonal/invisible to the current top-layer activity, then the network output is unchanged as well.

这样 theorem 和 implementation 就完全一致。

Abstract / Introduction 里也建议把：

> “the present computation is held invariant”

改成：

> “the hidden computation is held invariant on the exact channel, while the default plastic-readout system is empirically near-invariant end-to-end.”

这并不会削弱 paper。

实际上：

$$
0.27\%\rightarrow0.001\%
$$

这个 measurement 非常有价值，因为你清楚地展示了**理论 guarantee 在哪里停止**。

---

# 6. Proposition 2 比上一版好很多，但 batch notation 还不完全严谨

你现在定义：

$$
A^\ell(x)=
\{i:a_i^\ell(x)\neq0
\text{ for some sample in }x\}.
$$

这意味着这里的 \(x\) 实际已经是一个 batch。

但证明里又像处理单个 sample 一样写：

$$
a_j^{\ell-1}(x),\quad
z_i^\ell,\quad
\delta_i,
$$

以及一个 scalar 的 \(k\)-th largest threshold。

严格写的话建议改成：

$$
X=\{x_b\}_{b=1}^{m},
$$

$$
A^\ell(X)
=
\left\{
i:
\exists b,\,
a^\ell_{i,b}\neq0
\right\}.
$$

Proposition 1 的 pre-silent condition 就是：

$$
j\notin A^{\ell-1}(X)
\quad\Longleftrightarrow\quad
a^{\ell-1}_{j,b}=0,\ \forall b.
$$

Proposition 2 则定义：

$$
\delta_{i,b}
=
g^\ell
\sum_j
\Delta^\ell_{ij}
a^{\ell-1}_{j,b}
+
\Delta b_i^\ell
$$

并要求：

$$
\boxed{
\sigma(z^\ell_{i,b}+\delta_{i,b})
<
\tau^\ell_{k,b},
\qquad
\forall i,\forall b
}
$$

其中 \(\tau^\ell_{k,b}\) 是 sample \(b\) 对应的 k-WTA threshold。

这样 proposition 才和真正的 batch mask 完全一致。

顺便明确使用 strict inequality 是为了避免 k-th place tie；或者说明 deterministic tie-breaking。

---

# 7. \(g^\ell\) 还是没有定义，而 proof 仍然假定它线性

Eq. (1) 写：

$$
z^\ell=g^\ell(W^\ell a^{\ell-1})+b^\ell,
$$

但 Appendix 直接写：

$$
\hat z_i^\ell-z_i^\ell
=
g^\ell\sum_j
\Delta^\ell_{ij}a_j^{\ell-1}.
$$

 

这只有在 \(g^\ell\) 是 scalar / fixed linear gain 时才成立。

建议不要留任何悬念，直接写：

$$
g^\ell>0
$$

is a fixed scalar layer gain，

甚至干脆写成：

$$
z^\ell
=
g^\ell W^\ell a^{\ell-1}+b^\ell.
$$

这是一个两分钟能修、但 reviewer 很容易挑出来的 notation hole。

---

# 8. Bias、decay 和 momentum 的 exactness 条件还应该正式写出来

你现在 Proposition 2 已经把 \(\Delta b_i^\ell\) 纳入了，这比上一版好。

但 Eq. (3) 定义的是 **synapse mask**，正文没有告诉 reviewer：

> replay 时 bias 到底怎么 mask？

如果 bias 对 awake unit 也更新，那么 pre-silent protection 并不能阻止：

$$
z_i^\ell\rightarrow z_i^\ell+\Delta b_i^\ell.
$$

最简单的方案：

> replay-time biases are frozen;

或者：

$$
M_{b,i}^\ell
=
1[i\notin A^\ell(X)]
$$

并把 margin condition 同样应用于 bias。

### Weight decay 也一样

你的 local update 中包含：

$$
-\lambda W^\ell.
$$



所以如果“mask”只是 mask replay data gradient，而 decay 仍然作用于 awake–awake synapses：

$$
W_{ij}
\leftarrow
(1-\eta\lambda)W_{ij},
$$

那么 exactness 立即失效。

正文最好明确一句：

> **The mask applies to the entire replay-induced parameter delta, including momentum and weight decay, not only to the data gradient.**

甚至直接给 heavy-ball replay equation，会非常干净：

$$
v_t
=
M_t\odot
\left(
\mu v_{t-1}
+
G_t-\lambda W_t
\right),
$$

$$
W_{t+1}=W_t+\eta v_t.
$$

如果实际 implementation 是“mask 外 velocity frozen rather than zero”，就按代码真实实现写。

现在稿子只说：

> velocity advances only inside the mask. 

对 reviewer 来说还不够精确。

而且我全文没有找到 **heavy-ball momentum coefficient \(\mu\)**。你后来声称 velocity 提供大约 “10× averaging”：

> “the velocity's ∼10× averaging of the 16-sample replay micro-batches...” 

如果你实际用的是：

$$
\mu=0.9,
$$

那这个 ~10× timescale 很合理，但一定把 \(\mu\) 写进 Appendix C/protocol。

---

# 9. “Exactly isolated” 现在基本收住了，但主文仍有几句口径太强

我比较喜欢你现在这一段：

> “The default system does not enforce the margin, so we measured it...”
> hidden code 0.36%, prediction 0.27%, margin violation \(\sim10^{-4}\). 

这非常像一个成熟 paper 的写法：

$$
\text{prove what is provable}
+
\text{measure the boundary elsewhere}.
$$

但 Results 里仍写：

> “Inside the mask each update leaves the present computation untouched...”



这和你自己的 0.36% / 0.27% measurement 不一致。

建议换成：

> “Inside the exact portion of the mask, each update is provably hidden-state neutral; violations in the remaining margin-conditioned channel are rare in measurement.”

这样完全没有漏洞。

---

# 10. Adaptive decay 是很好的新增结果，但公式现在有一个潜在 circularity

你定义：

$$
\lambda_\ell
=
\min
\left(
\lambda_{\max},
\rho
\frac{
\operatorname{EMA}|\Delta W^\ell|
}{
|W^\ell|
}
\right).
$$

然后正文说：

> \(|\Delta W^\ell|\) is the mean magnitude of the update the optimiser **actually applied**.



但前面的 update 本身又包含：

$$
-\lambda W.
$$

因此文字上看起来是：

$$
\lambda_t
\rightarrow
\Delta W_t
\rightarrow
\lambda_t,
$$

有自指问题。

我猜代码真正做的是使用 **decay-free / pre-decay waking learning drive**。

如果是这样，请正式定义：

$$
u_t^\ell
=
\text{data-driven, post-optimizer, pre-decay waking increment},
$$

然后：

$$
\lambda_t^\ell
=
\min
\left(
\lambda_{\max},
\rho
\frac{
\operatorname{EMA}(|u_t^\ell|)
}{
\operatorname{mean}|W_t^\ell|+\epsilon
}
\right).
$$

这样整个 controller 一下就清楚了。

另外最好加入 denominator 的 \(\epsilon\) floor，至少数学定义上避免：

$$
|W|=0.
$$

---

# 11. “One constant” 这个 claim 也建议稍微降一点

你说：

> “One dimensionless constant replaces per-dataset decay tuning.”

但公式实际上至少出现：

$$
\rho=0.25,\qquad
\lambda_{\max}=0.02,
$$

而 EMA 本身还必然有一个 smoothing coefficient。

所以严格意义上不是“整个 controller 只有一个 constant”。

真正漂亮、而且准确的 claim 是：

> **One dataset-independent ratio target replaces per-dataset decay tuning.**

然后说明：

> \(\rho\) is the only control target; \(\lambda_{\max}\), EMA coefficient and numerical floor are globally fixed safeguards and are never tuned per dataset.

这就不会被 reviewer 用一句：

> “But Eq. 4 visibly contains multiple constants.”

打回来。

---

# 12. Depth 结果很强，但 causal wording 要收一点

现在你的 argument 是：

> skip connections rescue fixed-decay deep system → collapse is located in error-path attenuation.

数据确实很有意思：adaptive controller 让 4/5 层不再直接死掉，local skips 进一步把 5-layer 带回约：

$$
91.5/96.2,
$$

接近 2-layer record。

但 forward skip connection 同时会改变：

* representation；
* forward conditioning；
* effective depth；
* activity distribution；
* error path。

所以严格来说：

> “locating the collapse in error-path attenuation”

还是有点 causal overclaim。

建议变成：

> “consistent with error-path attenuation being a principal contributor.”

如果你已经有 layerwise \(\|\epsilon^\ell\|\) 或 update norm 随 depth 的 plot，那就可以强 claim；没有的话就保持 “consistent with”。

---

# 13. 实验设计：这一版其实已经相当扎实

这部分是我觉得提升最大的地方。

现在不只是给一个 headline accuracy，而是有：

$$
\text{batch-size sweep}
$$

直接验证 micro-batch hypothesis；

$$
\text{rotation / isolation / replay ablations}
$$

验证 component necessity；

$$
\text{mirror direction}
$$

验证 conceptual inversion；

$$
\text{soft vs hard rotation}
$$

验证 ON/OFF design；

$$
\text{Adam masked/leak / heavy-ball / SGD}
$$

验证 optimiser-state mechanism；

$$
\text{clocked / pressure / surprise timing}
$$

验证 triggering；

再加：

* raw CIFAR-10；
* feature CIFAR-10；
* CIFAR-100 stress test；
* depth scaling；
* LR sensitivity；
* small buffer；
* negative results；
* static i.i.d. axis；
* held-out replication。

Table 1 尤其强，因为它不是“堆 baseline”，而是在回答：

> **哪个机制实际上在起作用？**

例如 full 92.7、no rotation 88.8、no isolation 87.0、no replay 19.6、mirror 77.9，已经形成一个相当完整的 causal story。

对于 workshop paper，我认为这部分已经超过“够用”。

---

# 14. CIFAR-10 现在也比上一版更有说服力

raw CIFAR-10：

$$
29.5\pm0.8
>
26.2\pm0.6\ {\rm night}
>
25.1\pm1.5\ {\rm BP+ER}.
$$

feature CIFAR：

$$
41.5\pm1.1
>
35.0\pm0.5\ {\rm night},
$$

但：

$$
BP+ER=43.2\pm1.0.
$$



这个结果现在的解释是健康的：你没有再把故事包装成“local biological learner beats BP universally”，而是拆 substrate，发现 dense activation/accounting 可以解释一部分 gap。

这反而让 paper 更可信。

---

# 15. 我最担心的实验方法学问题反而是你自己已经坦白的：test-set tuning

Appendix J 写得非常直接：

> “Every configuration choice made during development — some four hundred sequential-axis configurations — was made on the official test sets.”

然后你另外拿 training set 的 10% 做一个从未训练过的 held-out split，重新验证 headline ordering：

$$
\rho_{\rm Spearman}=0.92,
$$

且主要 ordering 保持。

这个 transparency 我非常赞同。

但 reviewer 仍然可能说：

> benchmark test accuracy 经过数百次 config search，headline 92.7 存在 optimistic selection bias。

而且 held-out 上：

$$
92.7\rightarrow91.6
$$

default rotation，

night 是：

$$
90.7\pm2.2.
$$



也就是说 independent held-out 依然支持 ordering，但**effect size 已经不像 official test 上的 2 points 那么漂亮**。

我不会删除这个 limitation；恰恰相反，要保留。

如果 deadline 前还能做一件统计上的事，我最推荐的不是再跑一个复杂 ablation，而是：

> **冻结现在所有 hyperparameters，然后对 untouched held-out split 多跑一些 seeds，最好 6–10 seeds。**

如果有算力，再做 3 个 fixed stratified splits。

这对于 reviewer confidence 的提升，会比再加一种 mechanism 大。

---

# 16. Headline comparison 最好报告 paired difference / CI

目前：

$$
92.7\pm0.3
$$

vs.

$$
90.7\pm0.4
$$

看起来很好。

但两边 architecture 还不完全一样：night headline 用它自己的 preferred narrow substrate，而你也同时报告 same-substrate night 90.9±3.2。这种设计其实很公平——你给 baseline 选了它更喜欢的 substrate——但 reviewer 会问：

> are these seeds paired? how were hyperparameters selected?

最好附一个：

$$
\Delta A
=
A_{\rm local}-A_{\rm night}
$$

的 bootstrap / paired CI。

同时 Appendix C 把 baseline sweep 说清楚：

* night 搜了哪些 LR / widths / replay batches；
* BP+ER 搜了哪些 LR；
* local 搜了哪些；
* selection criterion 是 sequential 还是 joint sequential/static。

这样可以彻底消除 “local method got more tuning budget” 的疑虑。

---

# 17. Surprise-trigger 那段目前说得稍微过头

现在实验是：

> surprise trigger 在该设置下因为 error 从未 crossing threshold，导致 **0 replay events**，accuracy 19.0。

这说明：

$$
\boxed{
\text{the tested surprise trigger failed badly}
}
$$

非常充分。

但 contribution 里直接写：

> “Replay is triggered by homeostatic pressure and **never by surprise**.”



以及：

> “The waking trigger **must** be homeostatic.”

如果没有 extensive surprise-threshold policy sweep，这两个都是 universal claim。

建议改成：

> “Homeostatic triggering consistently outperforms the tested surprise-based trigger.”

或者：

> “Surprise is a poor trigger in our waking-replay regime.”

会稳很多。

---

# 18. Energy 部分已经比上一版修得很好

上一版最大的 citation error 是把 0.9 pJ/event 归给 Merolla。

现在已经改成：

> idealised **45-nm FP32 arithmetic-energy proxy [Horowitz, 2014]**
> 0.9 pJ accumulate vs 4.6 pJ MAC，
> 并明确说不包括 data movement 和 membrane updates。

这就严谨很多了。

我只建议再换一个词：

不要写：

> 0.9 pJ per **synaptic event**

而直接写：

> 0.9 pJ per **FP32 accumulate (AC)**

因为 Horowitz 给的是 arithmetic operation energy，不是某块 neuromorphic hardware 的真实 synaptic event energy。

你 Discussion 已经主动说：

> “the energy figure is an arithmetic proxy.”



所以整体已经没什么问题。

---

# 19. Reference：上一轮指出的几个硬错误已经修掉了

### Lässig 2023 —— 已修

上一版把 Sorbaro 写成 Sacramento；现在已经正确：

> Lässig, Aceituno, **Sorbaro**, Grewe. 

### Whittington & Bogacz —— 已不再是 orphan reference

现在正文真的引用了它来解释 local predictive-coding-style error propagation。

### Horowitz —— 已修

现在 source 和 proxy 定义基本匹配。

---

# 20. 但 reference 还有三个值得修的地方

## Sorrenti 应更新为正式 journal version

稿件仍然写：

> Sorrenti et al. (2024), arXiv:2401.08623. 

现在已经有正式版本：

> **Wake-Sleep Consolidated Learning**,
> IEEE Transactions on Neural Networks and Learning Systems,
> **36(7):12668–12679**,
> DOI **10.1109/TNNLS.2024.3458440**. ([PubMed][3])

按 issue publication，我会写 2025，然后同步把正文 `[Sorrenti et al., 2024]` 改掉。

---

## SESLR 仍然缺失，而且现在更应该补

非常接近标题关键词的一篇：

> **Online Continual Learning via Spiking Neural Networks with Sleep Enhanced Latent Replay**,
> Lin et al., arXiv:2507.02901.

它确实提出 sleep-enhanced latent replay，而且是 online CL + SNN。关键是它的 sleep phase 仍然是：

> model **exclusively trains on replay samples** during the sleep phase.

所以它并不会破坏你的 novelty，反而非常适合拿来建立差异：

$$
\text{SESLR: online learning + exclusive sleep replay}
$$

vs.

$$
\text{ours: replay concurrent with the live computation}.
$$

([arXiv][4])

我会认为这是**submission 前应该补的近邻工作**，否则 reviewer 搜 “online continual sleep replay” 很可能自己碰到它。

---

## Tononi & Cirelli 的标题被截断了

现在 bibliography 写：

> “Sleep and the price of plasticity.” 

正式标题是：

> **Sleep and the price of plasticity: from synaptic and cellular homeostasis to memory consolidation and integration.**

([PubMed][5])

不算严重错误，但既然在做 final polishing，建议补完整。

---

# 21. 还有一个 citation-to-claim 小问题：burst learning 只引 Whittington 不够

你写：

> “a bounded, event-gated burst signal in the spirit of **predictive-coding and burst-multiplexing learners** [Whittington and Bogacz, 2017].”



Whittington & Bogacz 主要是 predictive coding / local Hebbian approximation to BP，并不是 burst-dependent credit-assignment 的主要来源。

可以在这里同时加：

> Payeur, Guerguiev, Zenke, Richards & Naud (2021),
> **Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits**, Nature Neuroscience 24, 1010–1019.

这篇正好直接讲 burst-dependent plasticity 和 hierarchical credit assignment。([Nature][6])

这样“predictive-coding and burst-multiplexing”两边都有 citation。

---

# 22. 版面和可读性

我也实际看了 PDF 页面，而不只是 parsed text。

整体是干净的 NeurIPS 风格，Figure 1 的 mechanism 图比上一版清楚很多；Figure 2 现在把 isolation 与 timing 放在一起也很合理；Figure 3 的 depth/static/controller decay 三联图是 Section 5 最有用的一张图。

正文页 6–8 已经比较密，但仍可读。我的直觉是：

> **不要再往正文塞实验了。**

如果还加东西，优先放 Appendix。

正文现在最值得腾空间做的不是增加结果，而是把 theorem assumptions 写得更严谨。

---

# 23. 如果我是 CL4FMAgents reviewer，我大概会这样写

**Strengths** 会是：

> A conceptually novel inversion of parameter-isolation continual learning: instead of protecting past knowledge from current updates, replay updates are constrained to degrees of freedom invisible to the concurrent computation. The use of refractory rotation to actively create such capacity is intuitive and biologically motivated. The paper provides unusually extensive mechanism-focused ablations and is candid about the boundary of its exactness guarantees.

**Weaknesses** 会是：

> The study remains limited to small MLPs and image benchmarks; the connection to foundation-model or embodied-agent settings is conceptual rather than demonstrated. Some theoretical claims need sharper qualification because the default readout remains plastic and the post-asleep channel is only conditionally exact. Hyperparameter development used official test sets. The replay-cost accounting should be clarified.

这其实已经是一个**相当像 workshop Accept 的 review profile**。

尤其 CFP 明确欢迎 theory、algorithms、negative/reproducibility results，并把 knowledge consolidation/online adaptation 写进 topics；所以 MLP scale 本身并不是 out-of-scope。([NeurIPS 2026 Workshop][1])

---

# 24. 提交前我会按这个顺序处理

1. **立刻修 92.7 / batch16 / 1.2× replay cost 的矛盾。**这是最优先，不能带着进 review。
2. **修 Proposition 1：hidden computation exact，不要在 plastic readout 下声称 network output exact。**同步修 Abstract、Intro、Results、Discussion 中 “present computation invariant” 的口径。
3. 把 theorem 全部显式 batch 化，并补 \(g^\ell\)、bias mask、decay mask、momentum mask/tie conditions。
4. 把 heavy-ball 的 \(\mu\) 和完整 masked update equation 写进 protocol。
5. 重定义 adaptive controller 的 drive 为 **pre-decay data-driven update**；补 EMA coefficient、\(\epsilon\) floor，并把 “one constant” 改为 “one shared ratio target”。
6. Section 5 改成 **scaling stress test / self-referenced control** 的叙事，避免像第二篇 paper。
7. 把 “must/never/provably” 这类 universal wording 再扫一遍，特别是 surprise trigger、absolute threshold 和 depth causal explanation。
8. Reference 补 SESLR；Sorrenti 换 journal version；Tononi 补全标题；burst learning 加 Payeur et al. 2021。
9. 如果还有计算时间，优先增加 **untouched held-out 的 seeds**，而不是再加 ablation。
10. 如果还有更多时间且想冲 oral，可以考虑一个非常轻量的 **frozen foundation encoder + sparse continual adapter** 实验，例如冻结视觉 foundation representation，只训练你的 sparse local head。它不需要成为性能主结果，只要证明 “silent-degree mechanism survives on FM features”，就会显著加强与 CL4FMAgents 的直接连接。

---

## 我的最终判断

**我会投，而且觉得这个 venue 选得不错。**

相比你给我的上一稿，这一稿已经解决了最危险的一部分问题：你不再把所有 isolation 都包装成 unconditional exact，优化器的故事更完整，实验也从“几组 accuracy”发展成了真正的 mechanism triangulation。论文现在最有辨识度的核心已经相当明确：

$$
\boxed{
\text{Inference exposes temporary unused degrees of freedom;}
}
$$

$$
\boxed{
\text{refractory rotation actively enlarges them;}
}
$$

$$
\boxed{
\text{replay uses them for concurrent consolidation.}
}
$$

这就是 **Inference as Consolidation** 这个标题现在能够站得住的原因。

目前让我不建议“直接原样上传”的主要原因其实只有两个：**2.4×/1.2× cost accounting** 和 **plastic readout 下 Proposition 1 仍写 network-output invariance**。这两个修完以后，我对这篇作为 CL4FMAgents regular paper 的评价会从“有明显 reviewer 把柄”提升到“理论边界和实验边界都讲得比较诚实、机制又有辨识度”的状态。

顺带一提，官方目前显示 deadline 已延到 **2026-09-07 11:59 PM AoE**。([NeurIPS 2026 Workshop][1]) 所以从优先级上，我不会再扩展主故事，而会把剩余时间全部用于**消灭内部不一致 + theorem formalization + reference cleanup**。

[1]: https://neurips26-cl4fmagents.github.io/?utm_source=chatgpt.com "Continual Learning in the Era of Foundation Models and Embodied Agents — NeurIPS 2026 Workshop"
[2]: https://www.nature.com/articles/s41593-026-02318-9?utm_source=chatgpt.com "Induction of cortical on/off periods in awake mice fulfills sleep functions | Nature Neuroscience"
[3]: https://pubmed.ncbi.nlm.nih.gov/39325610/?utm_source=chatgpt.com "Wake-Sleep Consolidated Learning."
[4]: https://arxiv.org/abs/2507.02901?utm_source=chatgpt.com "Online Continual Learning via Spiking Neural Networks with Sleep Enhanced Latent Replay"
[5]: https://pubmed.ncbi.nlm.nih.gov/24411729/?utm_source=chatgpt.com "Sleep and the price of plasticity: from synaptic and cellular homeostasis to memory consolidation and integration - PubMed"
[6]: https://www.nature.com/articles/s41593-021-00857-x?utm_source=chatgpt.com "Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits | Nature Neuroscience"
