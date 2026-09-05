我读了 [cl4fmagents(3).pdf](sandbox:/workspace/scratch/60209d29f51f/upload/cl4fmagents%283%29.pdf) 第 3–4 页的 Method、附录 A 的证明，以及第 5 节的衰减控制器与深度扩展。下面我把自己放在一个**有神经网络和 continual learning 背景、但第一次接触这篇手稿的审稿人**的位置，按你的叙述顺序往下理解。

我的初步判断是：**核心数学思路可以理解，两个命题的证明主线也成立；阅读障碍主要来自几种“静默”的混用、没有完全写出的执行时序，以及数学保证与设计选择之间的界限。**我会明确指出哪些地方是公式本身告诉我的，哪些地方需要我替作者补充假设。

**首先读 Setting，我会把研究问题理解为：网络能否一边处理当前输入，一边通过 replay 维持旧知识，同时尽量不扰动当前正在使用的表示。**

这里有两组不同的问题：

* \(A_{\mathrm{seq}}\)、forgetting 衡量持续学习表现。
* \(A_{\mathrm{iid}}\)、replay samples \(R\)、energy proxy \(E\) 衡量这种机制付出的代价。

因此，我不会预期接下来看到一个“证明不会遗忘”的定理。我会预期看到的是：

> 作者先证明某些 replay 更新不会改变当前输入的计算，再通过实验说明这种更新方式有助于持续学习。

这个区分很关键。你的数学保证主要是**当前输入上的局部不干扰性**，持续学习表现则需要实验支持。

另外，Setting 中“没有 task identity”明确写的是测试阶段；它与“没有监督标签”是两回事。后面的 \(y\) 说明这仍然是 supervised continual learning。

---

**读到 Eq. (1)，我首先把它拆成“计算输入、强制休眠、稀疏竞争”三个操作。**

你的公式是：

$$
z^\ell=W^\ell a^{\ell-1}+b^\ell,
\qquad
a^\ell=
\Phi_k\!\left(
\sigma(z^\ell)\odot(1-s^\ell)
\right).
$$

按照列向量记法：

$$
a^{\ell-1}\in\mathbb R^{n_{\ell-1}},
\quad
W^\ell\in\mathbb R^{n_\ell\times n_{\ell-1}},
\quad
z^\ell,b^\ell,a^\ell\in\mathbb R^{n_\ell}.
$$

我会逐步理解为：

1. \(z^\ell\) 是这一层每个单元收到的净输入。
2. \(\sigma\) 将负输入整流为零。
3. \(s_i^\ell=1\) 的单元被强制置零。
4. \(\Phi_k\) 在剩余响应中保留最大的 \(k\) 个值。

这里一个容易忽略、但非常重要的细节是：**suppression 发生在 k-WTA 竞争之前。**

例如：

$$
z=(5,4,3,1),\qquad k=2.
$$

不休眠时：

$$
a=(5,4,0,0).
$$

如果让第一个单元休眠：

$$
s=(1,0,0,0),
$$

那么：

$$
\sigma(z)\odot(1-s)=(0,4,3,1),
$$

最终：

$$
a=(0,4,3,0).
$$

这意味着原来的第三名会补上来。它并不是先选出赢家，再把某个赢家删除。这正是后文“换一组单元承担当前计算”的基础。

读到这里，我也会建立一个后文反复需要的区分：

| 单元为什么输出为零               | 对权重更新是否天然稳健               |
| ----------------------- | ------------------------- |
| \(z_i\leq 0\)，被 ReLU 截断 | 更新可能让它变成正值                |
| 响应为正，但没有进入 top-\(k\)    | 更新可能让它成为赢家                |
| \(s_i=1\)，被强制休眠         | 固定 \(s_i\) 时，任意输入变化都仍然乘以零 |

**前两种是“目前没有激活”，第三种是“目前不允许激活”。**你后面的两个命题与推论，实际上就是围绕这个区别展开的。

这里有一个会让我暂停的符号问题：Eq. (1) 写成 \(\ell=1,\ldots,L\)，似乎包括所有层；但 Eq. (2) 的 \(\varepsilon^L=y-z^L\) 又让 \(L\) 看起来是输出层。这样读者就需要猜：输出层是否也经过 suppression 和 k-WTA？

建议明确写出隐藏层与 readout 的边界。例如用 \(H\) 表示隐藏层数，再单独定义：

$$
r=W^{\mathrm{out}}a^H+b^{\mathrm{out}},
\qquad
\varepsilon^{\mathrm{out}}=y-r.
$$

如果实际实现采用其他约定，也应直接说明。这会同时消除后面“hidden computation”和“readout exception”的歧义。

---

**读到 Eq. (2)，我会把它理解成一种局部误差信号的构造规则，而不是自动视为标准反向传播。**

你的公式是：

$$
\varepsilon^L=y-z^L,
$$

$$
\varepsilon^\ell
=
\sigma'(z^\ell)
\odot
(B^\ell)^\top
\left[
\kappa\tanh(\varepsilon^{\ell+1}/\kappa)
\odot
\mathbf 1(a^{\ell+1}>0)
\right].
$$

为了读清楚，我会先定义一个中间量：

$$
q^{\ell+1}
=
\kappa\tanh(\varepsilon^{\ell+1}/\kappa)
\odot
\mathbf 1(a^{\ell+1}>0).
$$

这样原式变成：

$$
\varepsilon^\ell
=
\sigma'(z^\ell)\odot(B^\ell)^\top q^{\ell+1}.
$$

现在每部分的作用比较清楚。

首先，输出残差：

$$
\varepsilon^L=y-z^L
$$

表示目标与当前输出之间的差。如果 \(y\) 是 one-hot target、输出采用平方误差，这对应输出端的负梯度方向。但手稿需要把 target encoding 与误差定义写出来，读者不应自行认定使用的是 cross-entropy 或平方误差。

其次：

$$
\kappa\tanh(\varepsilon/\kappa)
$$

在误差较小时近似保留原值：

$$
\kappa\tanh(\varepsilon/\kappa)\approx\varepsilon,
$$

在误差较大时限制单个传递分量的幅度：

$$
\left|\kappa\tanh(\varepsilon/\kappa)\right|\leq\kappa.
$$

随后：

$$
\mathbf 1(a^{\ell+1}>0)
$$

表示只有上层实际激活的单元发送这个误差信号。没有参与当前前向表示的上层单元，不发送该分量。

最后，\((B^\ell)^\top\) 把信号传回当前层，\(\sigma'(z^\ell)\) 再根据当前层的整流状态进行门控。

所以我会把这段理解为：

> 激活的上层单元发送幅度受限的误差信号；当前层通过独立学习的反馈连接接收这些信号，并形成自己的局部更新方向。

有两处表述值得更精确。

第一，**受到 \(\kappa\) 限制的是单个发送分量，不一定是聚合后的 \(\varepsilon^\ell_i\)。**对于 ReLU，有：

$$
|\varepsilon_i^\ell|
\leq
\kappa\sum_j|B^\ell_{ji}|.
$$

因此，若没有额外限制反馈权重，就不能直接说每层最终误差都被 \(\kappa\) 界定。

第二，这里使用独立的 \(B^\ell\)，并且没有完整写出 suppression 与 k-WTA 的导数。因此，我会把它视为作者定义的学习规则；它不需要等于标准 backpropagation，但应明确自己的定位。

你接着给出：

$$
\Delta W^\ell
\propto
\varepsilon^\ell(a^{\ell-1})^\top-\lambda W^\ell.
$$

在单个连接上就是：

$$
\Delta W^\ell_{ij}
\propto
\varepsilon_i^\ell a_j^{\ell-1}-\lambda W^\ell_{ij}.
$$

“local”的含义由此明确：这个连接的更新使用接收端的误差信号、发送端的活动，以及自身权重。

对于 forward–feedback alignment，我会这样补出其中的直觉：如果形状对应的 \(W^\ell\) 与 \(B^{\ell-1}\) 获得相同的实际增量 \(Q\)，并采用相同衰减，那么在暂不考虑额外约束投影时：

$$
W^+=W+Q-\lambda W,
\qquad
B^+=B+Q-\lambda B,
$$

于是：

$$
W^+-B^+=(1-\lambda)(W-B).
$$

共同的数据增量抵消，二者的差异受到衰减。这解释了 shared decay 为什么与 alignment 有关。不过，实际还有动量、mask 和符号约束，所以“相同实际增量”的实现条件最好交代清楚。

---

**读到 awake set，我会意识到：你保护的对象是整个 waking batch，而不是其中一个样本。**

定义是：

$$
\mathcal A^\ell(X)
=
\{i:\exists b,\ a^\ell_{i,b}\neq0\}.
$$

其中的 \(\exists b\) 非常重要。只要单元在 batch 中的任意一个样本上激活，就被视为 awake。

例如两个样本的活动为：

$$
a_{\cdot,1}=(1,0,2,0),
\qquad
a_{\cdot,2}=(0,3,2,0).
$$

那么：

$$
\mathcal A(X)=\{1,2,3\}.
$$

只有第四个单元对整个 batch 都静默。

由此得到后面证明需要的性质：

$$
j\notin\mathcal A^{\ell-1}(X)
\quad\Longrightarrow\quad
a^{\ell-1}_{j,b}=0
\quad\forall b.
$$

这比“某个样本上为零”强得多。

作为读者，我也会因此注意：**单样本 10% 的活动率，不等于整个 batch 有 90% 的单元可以视为 silent。**batch 中不同样本的激活集合取并集后，awake set 可能大很多。因此，batch size 与实际 code overlap 都会影响可隔离的空间。

---

**读到 Eq. (3)，我会把“或”拆成两个性质不同的更新通道。**

你的 mask 是：

$$
M^\ell_{ij}
=
\mathbf1
\left[
j\notin\mathcal A^{\ell-1}(X)
\ \lor\
i\notin\mathcal A^\ell(X)
\right].
$$

其中 \(j\) 是发送端，\(i\) 是接收端。

| 发送端 \(j\) | 接收端 \(i\) | 是否允许 replay 更新 |
| --------- | --------- | -------------- |
| awake     | awake     | 不允许            |
| silent    | awake     | 允许             |
| awake     | asleep    | 允许             |
| silent    | asleep    | 允许             |

读到这里，我不会立即接受“所有允许更新都不可见”。我会分开理解：

* **发送端 silent：**连接乘上的输入就是零，改变它不会改变当前净输入。
* **接收端 asleep：**连接更新可能改变该单元的净输入；只有它更新后仍然不输出，才不会影响当前计算。

这正是 Proposition 1 和 Proposition 2 需要分开的原因。

因此，你的核心 mask 包含：

> 一个无条件成立的 pre-silent 通道，以及一个需要进一步判断的 post-asleep 通道。

后者又因为“自然未激活”与“强制休眠”的区别，被拆成 margin 条件与 Corollary 1。

这个逻辑建议在 Eq. (3) 后直接用一句话预告。否则读者很容易把“目前没有使用”误读成“怎么更新都不会被使用”。

---

**读到 Eq. (4)，我会认为你正在解决一个实现层面的必要问题：保证最终的参数变化真正服从 mask。**

你的更新为：

$$
v^+
=
M\odot(\mu v+G)+(1-M)\odot v,
$$

$$
W^+
=
W+\eta M\odot v^+-\lambda M\odot W.
$$

拆成单个连接后更清楚。

当 \(M_{ij}=1\)：

$$
v^+_{ij}=\mu v_{ij}+G_{ij},
$$

$$
W^+_{ij}=(1-\lambda)W_{ij}+\eta v^+_{ij}.
$$

当 \(M_{ij}=0\)：

$$
v^+_{ij}=v_{ij},
\qquad
W^+_{ij}=W_{ij}.
$$

因此，被保护连接既不发生 replay 引起的参数变化，也不推进自己的动量状态。

为什么只 mask 当前梯度不够？因为即使：

$$
G_{ij}=0,
$$

历史动量仍可能满足：

$$
\mu v_{ij}\neq0.
$$

如果后面的权重更新不再 mask，它仍然会移动。weight decay 也同理。

但这里我会提出一个比较实质的审稿意见：

**冻结优化器状态，是比“本次隐藏表示不变”更强的要求。**

假设某个优化器允许内部状态继续更新，但最终参数增量仍严格满足：

$$
\Delta W=M\odot U,
$$

那么被保护位置的参数仍然完全不动。只要其余定理条件成立，本次隐藏表示的不变性仍可成立。

所以，需要区分：

* 最终参数变化越出 mask：可能直接破坏你的不变性保证。
* 内部 moments 在 mask 外更新，但最终参数变化仍被 mask：不自动推翻当前前向不变性，但改变了优化器状态隔离与未来学习行为。

因此，后文“letting the moments leak … surrenders the guarantee”需要明确说明：实现中到底是哪一种泄漏。

另有两个小定义应补上：Eq. (4) 用加号，所以 \(G\) 应是更新方向或负梯度的局部替代；而这里的 \(\lambda\) 是**每步直接使用的衰减系数**，没有再乘 \(\eta\)。这与后面的控制器解释直接相关。

---

**读到 Proposition 1，我会认为这是整篇文章最直接、最坚实的一步。**

条件是：

$$
\Delta^\ell_{ij}=0
\quad
\text{whenever }
j\in\mathcal A^{\ell-1}(X).
$$

也就是只允许修改来自 silent presynaptic units 的连接。

对当前 batch 中任意样本 \(b\)，假设前一层活动不变：

$$
\widehat z^\ell_{i,b}-z^\ell_{i,b}
=
\sum_j\Delta^\ell_{ij}a^{\ell-1}_{j,b}.
$$

每一项只有两种可能：

* 如果 \(j\) awake，按照条件，\(\Delta^\ell_{ij}=0\)。
* 如果 \(j\) silent，按照定义，\(a^{\ell-1}_{j,b}=0\)。

因此：

$$
\Delta^\ell_{ij}a^{\ell-1}_{j,b}=0
\quad\forall j,
$$

从而：

$$
\widehat z^\ell_{i,b}=z^\ell_{i,b}.
$$

净输入完全一样，在固定 suppression mask 与竞争规则下，活动当然也完全一样。

一个数值例子足够说明“任意幅度”的来源：

$$
a=(2,0,3),
\qquad
W=(1,4,2).
$$

原来的输出为：

$$
Wa=1\times2+4\times0+2\times3=8.
$$

把第二个权重从 \(4\) 改到 \(104\)：

$$
\widehat Wa
=
1\times2+104\times0+2\times3=8.
$$

更新再大也没有影响，因为它乘的是零。

随后使用逐层归纳：

$$
a^0=x\text{ 不变}
\Longrightarrow
a^1\text{ 不变}
\Longrightarrow
\cdots
\Longrightarrow
a^H\text{ 不变}.
$$

**这是有限幅度更新下的精确等式，不只是“小梯度近似不影响输出”。**

但我会把它的适用条件记在旁边：比较前后使用同一个 \(X\)、同一个 suppression state；这里没有允许任意 bias 更新；所有相关层的更新都要满足条件。

它也只保护这个 \(X\)。某个单元对其他输入可能激活，相关权重更新当然可以影响那些输入。这正是 replay 仍然可能发挥作用的空间。

---

**读到 Proposition 2，我会发现问题从“乘以零”变成了“会不会跨过竞争边界”。**

现在允许修改 asleep 单元的入连接，即使发送端 awake：

$$
\delta_{i,b}
=
\sum_j\Delta^\ell_{ij}a^{\ell-1}_{j,b}
+
\Delta b_i^\ell.
$$

这次通常不能再得到 \(\delta_{i,b}=0\)。

你的要求是：

$$
\sigma(z^\ell_{i,b}+\delta_{i,b})
<
\tau^\ell_{k,b},
$$

其中 \(\tau^\ell_{k,b}\) 是更新前进入 k-WTA 的第 \(k\) 大响应。

我会把它翻译成：

> 这个单元的内部净输入允许改变，但改变之后，它仍不能挤进当前赢家集合。

继续用：

$$
z=(5,4,3,1),\qquad k=2,
$$

此时：

$$
a=(5,4,0,0),
\qquad
\tau_k=4.
$$

如果第三个单元从 \(3\) 变为 \(3.5\)：

$$
z'=(5,4,3.5,1),
$$

它仍然不是赢家，因而：

$$
a'=(5,4,0,0).
$$

但如果变为 \(4.5\)：

$$
z'=(5,4,4.5,1),
$$

那么：

$$
a'=(5,0,4.5,0).
$$

第三个单元进入，第二个单元退出，隐藏表示改变。

所以，**post-asleep 更新的安全性取决于它距离 winner boundary 有多远。**

对于有严格正间隔的普通未激活单元，可以定义：

$$
m_{i,b}
=
\tau^\ell_{k,b}-\sigma(z^\ell_{i,b})>0.
$$

由于 ReLU 是 1-Lipschitz，一个更保守、但直观的充分条件是：

$$
|\delta_{i,b}|<m_{i,b}.
$$

这不是说实际算法必须用这个界，而是帮助读者理解：更新幅度不是凭空受限，它受到当前竞争间隔约束。

证明中另一个需要读者自己补出的环节是：**原来的赢家为什么不会自己发生变化？**

因为对 awake 接收端，你只允许 pre-silent 更新；由 Proposition 1，它们的净输入不变。于是：

1. 原来的赢家数值不变。
2. 所有被更新的普通 loser 都没有越过原来的门槛。
3. 原来的赢家集合与输出值都保持不变。

这三句话加进去，附录 A 的证明会容易跟很多。

此外，严格小于号是充分条件，并没有覆盖所有安全情况。例如 \(\tau_k=0\) 时，更新后仍然输出零的单元也是安全的，但不满足 \(0<0\)。建议单独说明零阈值情形，或者明确这个 margin 条件只讨论具有正竞争阈值的情况。

---

**读到 Corollary 1，我终于理解 rotation 为什么能加强理论保证。**

如果：

$$
s_i^\ell=1,
$$

那么无论更新把净输入推到哪里：

$$
\sigma(z_i^\ell+\delta_i)(1-s_i^\ell)=0.
$$

它甚至不会以一个正响应参加竞争。

因此：

* 普通 loser 必须满足“更新后仍然输掉竞争”。
* 强制休眠单元已经被禁止参加竞争，不需要 margin 条件。

这一步不是在说“休眠单元的内部状态完全没变”。恰恰相反，它的入权重、bias、净输入都可以变化；只是这些变化在当前 suppression state 下无法显露为活动。

到这里，三类保证可以精确对应起来：

| 更新通道                | 当前隐藏表示不变的原因      |
| ------------------- | ---------------- |
| pre-silent          | 权重变化乘上零输入        |
| post-suppressed     | 净输入变化最终乘上零 gate  |
| post-inactive、未强制休眠 | 更新后仍未跨过 k-WTA 门槛 |

这张对应表很适合放在两个命题附近，因为它直接解释了为什么有两条 proposition 和一条 corollary。

---

**读到 exactness measurement，我会把理论结论与系统实际运行的结论分开。**

手稿明确说默认系统不强制检查 margin，并报告：

* top hidden code 在 \(0.36\%\) 的 waking samples 上发生变化；
* prediction 在 \(0.27\%\) 上发生变化；
* readout 也隔离时，prediction 变化为 \(0.001\%\)。

我会理解为：

> 理论刻画了哪些更新严格不可见；完整算法还使用了一个未强制满足充分条件的通道，因此通过实测报告整体偏离程度。

这些比例也不能互相替代。hidden code 改变不一定导致类别改变；hidden code 不变也不保证类别不变，因为 readout 仍然可塑。

假设顶层隐藏表示 \(a^H\) 完全不变，readout 更新仍会带来：

$$
\widehat r-r
=
\Delta W^{\mathrm{out}}a^H
+
\Delta b^{\mathrm{out}}.
$$

所以，“隐藏计算不变”与“预测不变”是两个层次。

同样，我不会把这些命题进一步理解为：

* 未来输入不受影响；
* 当前样本以后永远不受影响；
* 误差信号与未来训练轨迹不变；
* 不会遗忘旧任务。

尤其你也更新反馈权重 \(B\)：即使本次前向活动不变，后续局部误差信号仍可能变化。

---

**读到 refractory rotation，我会把它理解为主动制造下一步可安全更新的单元。**

规则是：

$$
s_i^\ell(t+1)
=
\mathbf1[i\in\mathcal A^\ell(X_t)].
$$

即当前 batch 只要用过某个单元，下一批就让它休息。

在 rotation 始终开启的情况下，由定义可得：

$$
\mathcal A^\ell(X_t)
\cap
\mathcal A^\ell(X_{t+1})
=
\varnothing.
$$

因为上一批的 awake units，下一批被强制输出为零。

于是，当前承担表示的单元在下一步获得了一个受 Corollary 1 保护的写入窗口。这让 rotation 与 replay isolation 之间的联系非常直接。

但读到这里，我会提出一个必须补充的执行问题：

**这个 suppression mask 作用于 waking forward，还是也作用于 replay forward？**

如果同一个 \(s^\ell(t)\) 也用于 replay，那么强制休眠单元在 replay 的前向活动中同样为零。你的局部规则也许仍允许某些入连接通过误差信号更新，但“这些单元如何在 replay 中被重新激活并承载记忆”就不清楚了。

结合你对 \(\rho^\ell\) 的描述，我最自然的推测是：

> 单元对当前 waking computation 休眠，但在 replay computation 中可以参与活动。

如果这就是实际设计，建议明确写出 waking 与 replay 使用各自的活动变量及 suppression 约定，而不是让读者猜。

按照“replay-activated units 中，有多少对 waking batch asleep”的文字定义，我会将 \(\rho^\ell\) 理解为：

$$
\rho^\ell
=
\frac{
|\widetilde{\mathcal A}^{\ell}(X_{\mathrm{replay}})
\setminus
\mathcal A^\ell(X_{\mathrm{wake}})|
}{
|\widetilde{\mathcal A}^{\ell}(X_{\mathrm{replay}})|
},
$$

这里假设分母非零，并用波浪号区分 replay 活动。

这样 \(0.53\) 的含义就是“约 53% 的 replay 活跃单元对 waking batch 处于 asleep 状态”，而不是“53% 的权重可更新”，更不是“53% 的更新都自动满足无条件精确保证”。

---

**读到 internal control，我会把“何时 replay”和“何时 rotation”作为两套控制器理解。**

先看 homeostatic pressure：

$$
S_i\leftarrow S_i+\bar a_i(X_t).
$$

如果 \(\bar a_i(X_t)\) 表示 batch 平均活动，我会写成：

$$
\bar a_i(X_t)
=
\frac1m\sum_{b=1}^{m}a_{i,b}.
$$

它累计的是单元被使用的强度。长期活动越强，压力越大。

对于当前 asleep 集合 \(\mathcal D_t\)，触发条件可理解为：

$$
\frac{1}{|\mathcal D_t|}
\sum_{i\in\mathcal D_t}S_i
\geq\theta.
$$

于是发起 \(n_b\) 个 replay micro-batches。

这里的逻辑是：

> 某些单元此前被频繁使用，现在恰好进入可 consolidation 的休眠窗口；当这些单元的积累压力足够大，就集中进行 replay。

因此，pressure 不是误差，也不是 novelty。它是使用历史的积分量。

不过，“consolidation discharges the pressure”还不足以复现算法。我会想知道它具体是归零、减去常数，还是乘一个衰减系数；也需要定义哪些单元被认为“received consolidation”，以及跨层如何汇总 asleep pressure。

再看 relative novelty。把你的文字展开为：

$$
\bar e_f(t)
=
(1-\alpha_f)\bar e_f(t-1)+\alpha_f e_t,
$$

$$
\bar e_s(t)
=
(1-\alpha_s)\bar e_s(t-1)+\alpha_s e_t.
$$

其中 \(\alpha_f=0.1\)、\(\alpha_s=0.002\)，前者较快响应最近变化，后者保存较长历史。

条件：

$$
\bar e_f\geq\beta\bar e_s
$$

表示近期误差明显高于长期基准。

另一个条件：

$$
\bar e_f\geq\gamma e_0
$$

表示即使没有突增，目前误差仍然相对初始基准较大，任务尚未充分掌握。

两个条件取“或”，分别覆盖“最近变难了”和“一直还没学好”。

这里的尺度不变性可以具体说明：如果所有误差及其基准都统一乘以 \(c>0\)，两个不等式的真假不会改变。但它并不意味着任意更换 loss、非线性变换 error 或跨任务改变误差结构后，门控行为都会保持一致。

最后，我会特别记住你写出的默认配置：**默认 rotation 总是开启；稀疏预算下研究 pressure-triggered bursts，而 headline 配置是每个 waking batch 都 replay。**这说明这些控制器是可选调度机制，不宜让 Figure 1 给人一种所有 headline 结果都同时依赖它们的印象。

---

**把这些部分接起来时，我最需要的是一个与定理严格对应的训练时序。**

附录 Figure 4 展示了 waking forward、wake update、awake sets 与 replay，但还有一个影响定理适用性的细节：

> awake set 是针对 replay 更新之前的哪一份参数计算的？

假设实际执行顺序是：

1. 用 \(W_t\) 做 waking forward，得到 \(\mathcal A(X;W_t)\)。
2. 做一次 unmasked waking update，得到 \(W_t^{\mathrm{wake}}\)。
3. 用第一步的 awake set 构造 mask，对新参数做 replay。

那么，某个单元在 \(W_t\) 下静默，不代表它在 \(W_t^{\mathrm{wake}}\) 下仍然静默。Proposition 1 中“它乘上的活动为零”就不能直接套用。

因此，手稿需要明确采用哪一种对应关系，例如：

* 在与 awake set 对应的参数状态上先执行 replay；
* 或在 waking update 后重新计算用于隔离的活动；
* 或给出其他足以保持相关 support 的条件。

这不是断定实现存在错误，而是说：**当前文字还没有把定理中的 \(W,a,\mathcal A\) 与程序中的更新时间点完全绑定。**

对于连续多个 replay micro-batches 的 burst，也有类似问题。如果始终复用最开始的 mask，而某一步普通 asleep 单元违反 margin、改变了活动，那么后面几步继续使用旧 support 的理论依据也需要说明。

一个简短的 Algorithm box，标注参数快照、活动缓存、replay suppression 和 mask 重算时机，会比继续增加机制描述更有效。

---

**接着读第 5 节 Eq. (5)，我会把它理解成控制“衰减量相对于学习增量的大小”。**

公式为：

$$
\lambda_\ell
=
\min\left(
\lambda_{\max},
\frac{
\rho\,\operatorname{EMA}[u^\ell]
}{
\overline{|W^\ell|}+\epsilon
}
\right).
$$

其中 \(u^\ell\) 是 waking 更新中，经过优化器、尚未加衰减的数据驱动参数增量的平均绝对值。

假设实际数据增量为 \(D^\ell_t\)，则我会理解为：

$$
u^\ell_t
=
\frac1{N_\ell}
\sum_{i,j}|D^\ell_{ij,t}|.
$$

按照 Eq. (4) 的约定，在对应的 waking 更新中，这个增量可包含学习率与动量的作用；它不是裸梯度的平均幅度。

为什么要除以平均权重大小？

因为每步 decay 为：

$$
D^\ell_{\mathrm{decay}}=-\lambda_\ell W^\ell.
$$

其平均绝对幅度是：

$$
\overline{|D^\ell_{\mathrm{decay}}|}
=
\lambda_\ell\overline{|W^\ell|}.
$$

忽略上限截断与很小的 \(\epsilon\)，代入控制器可得：

$$
\overline{|D^\ell_{\mathrm{decay}}|}
\approx
\rho\,\operatorname{EMA}[u^\ell].
$$

这就是 Eq. (5) 最直观的解释：

> 让衰减更新的平均幅度，维持在近期数据驱动更新平均幅度的某个比例附近。

例如：

$$
\overline{|W|}=0.1,
\quad
\operatorname{EMA}[u]=4\times10^{-4},
\quad
\rho=0.25,
$$

那么：

$$
\lambda=10^{-3},
$$

平均 decay 幅度为：

$$
10^{-3}\times0.1=10^{-4},
$$

恰好是数据增量幅度的四分之一。

如果数据驱动更新减弱十倍，控制器也会把 decay 降低十倍，而固定 \(\lambda\) 不会响应这种变化。

你强调只测量 **pre-decay** 增量，也很合理。否则 decay 自己造成的参数变化可能被算作“学习驱动力”，再反过来支持更大的 decay。

不过，我会给这条公式划定三个边界：

* 它控制的是平均绝对幅度的比例，不是每个连接的比例。
* 数据增量可能与权重同向，也可能反向，因此它不保证权重范数增加或避免所有 collapse。
* replay 使用 mask，且一个 waking batch 可能触发多次 replay，所以这不是“整个 batch 累计 decay 必定等于 waking 学习量的 25%”。

因此，“the decay removes a fixed fraction of what learning currently adds”略显宽泛；写成**匹配衰减与数据驱动更新的平均幅度比例**会更精确。

另外，类别数增加或深度增加导致有效学习信号减弱，是你的实验与机制分析支持的解释，并不是 Eq. (5) 自身证明的普遍规律。

---

**最后读 skip synapses，我会把它理解成对计算图的扩展，而不是新的隔离原理。**

按照你给出的文字，深层隐藏单元可以写成：

$$
z^\ell
=
W^\ell a^{\ell-1}
+
S^\ell a^{\ell-2}
+
b^\ell.
$$

对应 skip 权重的局部数据更新为：

$$
\Delta S^\ell
\propto
\varepsilon^\ell(a^{\ell-2})^\top.
$$

隔离 mask 应相应使用实际发送层：

$$
M^{\ell,\mathrm{skip}}_{ij}
=
\mathbf1
\left[
j\notin\mathcal A^{\ell-2}(X)
\ \lor\
i\notin\mathcal A^\ell(X)
\right].
$$

证明仍然沿用原来的思想：如果 skip 的输入单元静默，更新乘上零；如果接收端强制休眠，变化仍被 gate 隐藏；普通 inactive 接收端则继续需要 margin。

但 skip 如何参与反馈误差传播，也应该给出对应关系。否则“restore the thick error path”对读者而言主要还是机制描述，尚未完整对应到 Eq. (2) 的计算。

读完后，若我是审稿人，我会优先要求作者补充以下内容：

| 优先补充的内容                             | 它解决的阅读障碍                  |
| ----------------------------------- | ------------------------- |
| 明确隐藏层、输出层及 loss 的定义                 | Eq. (1)–(2) 的层编号和输出处理不够明确 |
| 区分 waking 与 replay 的 suppression    | 休眠单元如何参与 replay 学习尚需读者猜测  |
| 标明 awake set 对应的参数时间点               | 决定两个命题如何应用于实际训练循环         |
| 区分参数隔离与优化器状态隔离                      | 避免把较强设计要求写成当前定理的必要条件      |
| 给出 pressure discharge 与 error 的具体定义 | 控制器目前还不能仅凭文字完整复现          |

就降低理解成本而言，我最希望看到的是：**一个贯穿 Proposition 1、Proposition 2、Corollary 1 的小型数值例子，加上一段标明状态与更新顺序的伪代码。**前者让读者理解“为什么不可见”，后者让读者确认“程序实际执行的是否就是证明里的那一步”。
