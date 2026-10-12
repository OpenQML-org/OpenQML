# Python API 与数值约定

安装方式见项目 README；所有下列入口均可从 `openqml` 顶层取得。

## 线路和态

| 接口 | 行为 |
| --- | --- |
| `Circuit(n_qubits)` | 创建 1–64 比特的可变线路；门调用返回自身 |
| `State(n_qubits, backend='auto', seed=None, budget=None)` | 创建零态；对 State 的门调用原地修改自身 |
| `State.from_vector(amplitudes)` | 从归一化向量复制创建纯态 |
| `circuit.run(seed=None, backend='auto', budget=None, initial_state=None)` | 创建新态执行；传入态会先复制 |
| `state.clone()` / `circuit.copy()` | 独立复制，随机数发生器状态也随态复制 |
| `circuit.compose(block, mapping=None)` | 顺序附加线路；mapping 的第 i 项对应 block 的 qi |
| `circuit.inverse()` | 无条件酉线路的逆；噪声、测量、反馈不支持反演 |
| `circuit.repeat(n)` | 返回重复线路；重复测量键会被拒绝 |
| `circuit.operations` / `circuit.resources()` | 独立操作快照 / 操作数、深度和测量等统计 |

`run(seed=...)` 对同样的线路与初态产生同样的测量分支。`sample(seed=...)` 使用独立发生器，
不修改态也不改变态内部发生器；未提供采样 seed 时，使用态自带发生器。
对测量后态执行 `sample` 是对**该分支**采样。它不是反复重跑整个动态线路。
若要统计完整反馈算法的分支，应使用多个 seed 重复调用 `circuit.run()`。

**比特顺序：** q0 是整数索引的最低有效位，字符串右侧是 q0。
矩阵 wire 列表的第一项是局部最低有效位。`state.sample(..., selected_wires=[2,0])`
返回字符串右侧对应 q2，左侧对应 q0。

### 门

单比特：`h(q)`、`x(q)`、`y(q)`、`z(q)`、`s(q)`、`sdg(q)`、`t(q)`、`tdg(q)`、
`rx(theta,q)`、`ry(theta,q)`、`rz(theta,q)`、`p(theta,q)`。

多比特：`cx(control,target)`（别名 `cnot`）、`cy`、`cz`、`cp(theta,control,target)`、
`swap(a,b)`、`ccx(a,b,target)`、`mcx(controls,target)`。
`qft(selected_wires=None, inverse=False, swaps=True)` 使用正号 DFT：
`U[j,k] = exp(2πijk/N)/sqrt(N)`；`iqft` 是其逆。

`unitary(matrix, targets, controls=(), label='unitary')` 检查矩阵形状、有限值和 `U†U=I`。
最多 4 个目标，controls 是与目标不同的任意数量比特，均为正控制。
`pauli_rotation(theta, 'XYZ...')` 实现 `exp(-i theta P/2)`。
`global_phase(theta)` 实现 `exp(i theta) I`。

State 的直接门接口包括 H/X/Y/Z、RX/RY/RZ、CX/CZ、unitary。
其他线路块可以用 `block.run(initial_state=state)` 应用到实时态。

### 读数

| 接口 | 返回值与限制 |
| --- | --- |
| `state.statevector()` | 纯态完整复数数组的副本，混态拒绝；默认至多 18 比特 |
| `state.sparse_amplitudes()` | 纯态非零振幅字典，键为全局基底整数 |
| `state.probabilities(selected_wires=None)` | 对其他比特求和后的概率数组，默认至多 18 个选中比特 |
| `state.sample(shots=1024, seed=None, selected_wires=None)` | 无塌缩计数字典；稀疏态不展开完整 2^n 概率数组 |
| `state.expectation(pauli_or_Hamiltonian)` | 精确实数期望值 |
| `state.reduced_density(selected_wires=None)` | 约化密度矩阵，至多 max_density_qubits 个选中比特 |
| `state.bloch_vector(q)` | `[<X>,<Y>,<Z>]` |
| `state.entropy(selected_wires=None)` | 约化态的冯诺依曼熵，单位 bit |
| `state.backend` / `state.work` | 实际后端 / 已消耗的估算门与读数工作量 |

### 测量与噪声

`state.measure(q, basis='Z')` 按 Born 规则抽取一个 bit 并塌缩到对应基底。
`circuit.measure(q, key=None, basis='Z')` 在执行时记录到 `state.classical[key]`。
基底支持 X/Y/Z。线路中 `h/x/y/z/rx/ry/rz/cx` 支持 `condition=('key', bit)`；
键必须引用此前的测量。键在一条线路内必须唯一。

`channel(kraus, targets)` 要求 `sum(K†K)=I`，在完整寄存器的小密度矩阵上精确执行。
即使某次 p=0，本版本的信道接口仍使用密度矩阵路径。

| 信道 | 定义 |
| --- | --- |
| `amplitude_damping(p,q)` | `K0=diag(1,sqrt(1-p))`，`K1=sqrt(p)|0><1|` |
| `dephasing(p,q)` | 相位翻转 `(1-p)rho + p Z rho Z`；p 是翻转概率 |
| `depolarizing(p,q)` | 以概率 p 将该比特替换为 I/2；Pauli 权重为 p/4 |
| `reset(q)` | `|0><0|`、`|0><1|` 两个 Kraus 算符；无条件复位，纠缠伙伴可变成混态 |

### SimulationBudget

`SimulationBudget` 是不可变配置，可以传入 State 或 Circuit.run：

```python
import openqml as q

budget = q.SimulationBudget(max_sparse_states=4096, max_operations=2000, max_work=5_000_000)
state = q.ghz(30).run(budget=budget)
```

默认 `max_qubits=40`、`max_dense_qubits=18`、`max_sparse_states=262144`、
`max_density_qubits=8`、`max_memory_mb=256`、`max_work=20000000`、`max_operations=10000`。
全部取正整数；max_qubits 最多 40，max_density_qubits 最多 10。
增加密度限制仍受工作和分配预算约束。密度读取与操作也消耗工作预算。

auto 根据比特数选择 dense 或 sparse；它不实现任意线路的张量分解。
稀疏预算按中间支持数保守检查，某个最终会抵消的中间展开也可能触发限制。
它不是振幅截断。内存预算估算中间数组与稀疏项，不能保证 NumPy、Python 和 BLAS 的
整个进程常驻内存都低于给定数值；工作预算也不是秒数或超时保证。

## Hamiltonian 和优化

`Hamiltonian({'ZZ': -1, 'XI': -0.5})` 或 `Hamiltonian([('ZZ', -1), ('XI', -0.5)])`。
所有字符串长度相同，系数有限且为实数；重复项会合并。
`H.matrix(max_qubits=10)` 和 `H.ground_energy(max_qubits=10)` 仅用于小寄存器的精确参考。
`ising_hamiltonian(n, coupling=1, field=1, periodic=False)` 构造横场 Ising 链。

`gradient(fn, parameters, epsilon=1e-6)`：完整标量目标的中心差分。
`parameter_shift(fn, parameters)`：逐个角度 ±π/2 的期望值差分；仅满足单次未缩放
Pauli 旋转假设时准确。该工具无法检查用户函数内部的参数复用。

`minimize(fn, initial, gradient_fn=None, steps=100, learning_rate=0.05, method='adam', tolerance=1e-8)`。
不提供 gradient_fn 时用中心差分。GD 有 Armijo 回溯，Adam 不要求每步损失单调。
返回 OptimizeResult：最优的 `parameters/value`、实际迭代 `history`、
`n_steps`、`n_evaluations`、`converged`。converged 只表示梯度范数达阈值；
n_evaluations 不包含用户 gradient_fn 内部的目标调用。

`vqe(H, ansatz, initial, ...)` 用用户线路的精确期望作为目标。
`qaoa_maxcut(n, edges, gammas, betas)` 支持无权或非负权边；成本是
`sum w(1-Z_i Z_j)/2`，成本层 `exp(-i gamma C)`，混合层 `exp(-i beta sum X)`。

## 模型

全部模型：`fit(X,y)` 返回自身；`predict(X)`、`score(X,y)`、`get_params()`。
X 的推荐形状是 `(samples, features)`，一维数组解释为**多个单特征样本**，
单个多特征样本必须写成 `[[f1,f2]]`。

`QuantumKernel(layers=2, n_qubits=None)` 的直接输入是弧度，生成每层 RY 编码、
固定 RZ、线性 CX 链的保真度核。
`statevectors(X)` 返回样本在行的振幅矩阵；`pairwise(X,Y=None)` 返回核矩阵。
最多 8 个比特，4 百万个批量振幅或核元素。

`QuantumKernelClassifier(layers=2, regularization=0.01, n_qubits=None)`
将训练数据按每列最大绝对值缩放至 `[-pi/2,pi/2]` 后拟合 one-hot 核岭目标。
支持多类别数字或字符串标签；`predict_proba` 为 softmax 分数，未经校准。

`VariationalQuantumClassifier(n_qubits=None,layers=2,epochs=60,learning_rate=0.08,seed=1)`
和同构造参数的 `VariationalQuantumRegressor`：先 H 准备，然后每层可训练 RZ 相位偏移、
RZ 数据编码、可训练 RY、闭合 CX 环，读出 q0 的 Z 期望。
分类器将训练输入缩放至 `[-pi/2,pi/2]`，回归器至 `[-pi,pi]`；
测试数据沿用训练缩放，可能超出该范围。
回归器的仿射输出头也参与训练。

每个量子角只出现于一个未缩放 Pauli 旋转；训练对期望值逐角移位，
再用 BCE/MSE 链式法则传播，经典输出头使用解析导数。
seed 控制初始参数；量子期望是精确计算，没有有限 shots 噪声。

训练后：`parameters_`（核模型为 `coefficients_`）、`n_parameters_`、
`n_features_in_`、`feature_scale_`、`n_circuit_evaluations_`；
变分模型另有 `history_`、`optimization_result_`。
线路次数统计计算过的样本数，含训练和后续预测，不等同于真机提交次数。

经典基线：LogisticClassifier 为二分类线性逻辑回归；RidgeRegressor 为线性岭回归；
FourierRegressor 用输入的原始弧度与每列 `1..harmonics` 的正余弦特征。
分类 score 是 accuracy，回归 score 是 R²；常量目标采用有限 0/1 分数。
它们没有做默认参数数量匹配，也不代表完整的强经典模型集合。

## 数据和评测

`list_datasets()` / `load_dataset(name, n_samples=192, seed=1, noise=0.08, frequency=3, n_features=4)`。
Dataset 提供 `X/y/name/task_type/description/metadata/fingerprint` 和
`split(test_size=0.25,seed=1)`，返回四项 `X_train,X_test,y_train,y_test`。
分类按标签分层，每类测试数量独立四舍五入，保留至少一个训练/测试样本。
所有生成器为本地合成数据，哈希包含数组 dtype、形状和字节内容。

`get_task(name,n_samples=96,seed=1,test_size=0.25,**options)` 包装数据与划分策略。
这里 seed 是**生成数据的 seed**，run_benchmark 的 seed 是**划分 seed**。
`make_task(X,y,task_type='classification',name='custom',test_size=0.25)` 支持自有数据。
`get_suite('qml-cls-1')` / `get_suite('qml-reg-1')` 返回可迭代的数据集 id 集合。

`get_model(id,**parameters)` / `list_models()` 提供模型目录。
`run_benchmark(model_or_id,task_or_dataset_id,seed=1)` 训练一次并测 held-out 指标。
若传入模型对象，其初始化 seed 由模型配置决定，不会被划分 seed 隐式覆盖。
`compare_models(task='moons',models=None,seeds=(1,2,3),epochs=40)` 每次创建独立模型；
自定义 models 是 `{显示名: 零参数工厂}`，工厂负责自身随机种子。

分类记录 accuracy / balanced_accuracy；回归记录 MSE / MAE / R²。
比较结果按模型和指标报告中位数，不自动宣布胜负，也不做显著性检验。
`to_dict()` / `save(path)` 序列化设置、指标、数据哈希、seed、版本和资源。
结果文件不保存训练模型或原始数据，不作为模型持久化文件。

## 交换与错误

`to_qasm(circuit)` / `from_qasm(source)`：OpenQASM **2.0** 的明确子集。
单 qreg，最多单 creg；标准单比特门、RX/RY/RZ、u1/p、CX/CY/CZ、cu1/cp、CCX、SWAP、
单比特 Z 测量。角表达式仅支持数字、pi、加减乘除、括号。
不支持 QASM 3、自定义 gate 定义、多个寄存器、广播、barrier、reset、反馈、噪声、
全局相位。测量键导入时按 `creg名称+索引` 命名。语义外语法明确拒绝。

`to_qiskit(circuit)` 延迟导入 Qiskit；支持包括噪声、X/Y/Z 测量和反馈的转换。
这里只转换线路，不连接 IBM/IonQ，也不运行硬件任务。

`OpenQMLError` 是库错误基类；`NotFittedError`、`ResourceLimitError`、
`UnsupportedOperationError` 区分常见问题。格式和数据校验使用 ValueError。
没有吞掉错误并返回假结果的兜底路径。
