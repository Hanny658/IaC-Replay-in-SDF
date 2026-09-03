### 高优先级问题

1. CIFAR-10 特征存在测试集泄漏

[run_seq.py:829](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/run_seq.py:829>) 中 `feats()` 每次都重新估计阈值；训练集和测试集分别调用，因此测试阈值来自前 500 个测试样本。

这会影响全部 `dataset="cifarf"` 结果。应只在训练集估计 `thr`，然后固定用于 train/test。

2. Spiking 实验使用测试集校准阈值

[p10_spike.py:28](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/scripts/p10_spike.py:28>)、[p15_spike.py:36](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/scripts/p15_spike.py:36>)、[p16_spike.py:35](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/scripts/p16_spike.py:35>) 都用 `Xte[:2000]` 校准 firing thresholds。

应改为 `Xtr`，否则 spiking accuracy/energy 结果带有测试集调参。主 MNIST runner 在这点上是正确的。

3. 大量配置直接在官方测试集上反复选择

当前 `runs.csv` 已包含 435 个 sequential 配置，训练后每个配置都直接评估 `Xte`，例如 [run_seq.py:1205](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/run_seq.py:1205>)。

如果后续 phase/config 是根据这些 test accuracy 继续设计的，最终最好结果会有严重的 selection bias。建议从训练集划分 validation，所有 phase 和超参数只看 validation；最终方案冻结后，官方 test 只运行一次。

4. “exact isolation”与实际默认实现不完全一致

默认 `readout="free"`，replay 会更新全部输出层权重和 bias：[run_seq.py:1596](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/run_seq.py:1596>)。因此当前输入的最终输出并非严格不变。

另外，post-asleep 权重允许更新，但代码没有检查该神经元更新后是否仍低于 k-WTA margin：[run_seq.py:1599](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/run_seq.py:1599>)。所以目前能严格保证的是 pre-silent 部分，不是默认配置的端到端输出。需要实现 margin-preserving projection/isolated readout，或者收窄论文中的 exactness 表述。

### 中优先级问题

- `run_bio.py` checkpoint 文件名不含 seed，不同 seed 会直接覆盖：[run_bio.py:92](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/run_bio.py:92>)。
- MNIST/sequence checkpoint 只包含 config 和 seed，不包含 epochs、batch、NREM 参数；`--resume` 可能把旧实验误认为当前实验：[run_seq.py:1710](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/run_seq.py:1710>)。
- `results/bio/seq/runs.csv` 当前有 1378 行，但目录有 1438 个 checkpoint，汇总文件已经落后。
- heavy-tail columns 在完整训练集上预先选择，再传入 inner CV；因此验证 fold 参与了预处理规则选择：[evaluate.py:80](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/evaluate.py:80>)。应在每个 fold 的训练部分重新检测。
- masked waking update只要传入 `syn_mask`，就会被 `local_update()` 当作 replay：[cortex.py:541](<C:/Users/hanny/Desktop/NTU MSAI/AI6124 - Neuro Evo & Fuzzy/MLP-Cortex/src/models/cortex.py:541>)。这会让 mirror ablation 的 waking update 不更新 burst baseline、KP controller 和 goodness。建议增加显式 `is_replay` 参数。

### 验证状态

- 所有 Python 文件通过 `py_compile`。
- `pytest` 显示 `0 tests`。
- 当前 Python 环境缺少 `numpy/torch/pandas/sklearn`，仓库也没有 `requirements.txt` 或锁文件，因此暂时无法进行数值回归测试。
- Git 工作区干净。
