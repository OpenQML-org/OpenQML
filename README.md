# OpenQML

**一个可以直接 `import openqml` 的 Python 量子编程与机器学习库。**

参考 [openqml.org](https://openqml.org) 的量子编程、模型、数据集和实验流程，
提供独立的 NumPy 实现。用户可以在普通电脑上编写线路、读取量子态、训练模型，
并用固定数据划分比较量子与经典方法。Python 3.10+；核心依赖只有 NumPy。

当前版本：**0.1.0 / Alpha**。这是 Python SDK 的首个实现，
不是网站 JavaScript 引擎的包装，也不声称复现网站上的全部模型或评测协议。

## 安装，然后直接引用

下载源码压缩包，解压并进入包含 `pyproject.toml` 的目录，执行（Windows / macOS / Linux）：

```bash
python -m pip install .
```

如果下载包内含 `dist/openqml-0.1.0-py3-none-any.whl`，也可直接安装 wheel，无需安装 Git：

```bash
python -m pip install dist/openqml-0.1.0-py3-none-any.whl
```

这个命令会安装包，随后在**其他目录**的 Python 程序中也能运行：

```python
import openqml

state = openqml.Circuit(2).h(0).cx(0, 1).run()
print(state.sample(1024, seed=7))  # 只出现 00 和 11
print(state.expectation("ZZ"))  # 接近 1
print(state.reduced_density([0]))  # 单比特约化态 I/2
```

当前尚未发布到 PyPI；不要将 `pip install openqml` 当作本仓库已发布的承诺。
导入本库不会联网，不需要 API key，不会自动安装或导入 Qiskit、Torch、PennyLane。

## 量子编程

```python
import numpy as np
import openqml as q

circuit = q.Circuit(3)
circuit.h(0).ry(np.pi / 3, 1).cx(0, 2).rz(0.2, 2)
state = circuit.run(seed=42)

print(state.probabilities())  # 数组索引是计算基底整数
print(state.expectation("XIZ"))  # q2=X, q1=I, q0=Z
print(state.bloch_vector(1))
print(state.entropy([0]))  # 以 bit 为单位
print(circuit.resources())  # 宽度、操作数、线路深度等

# 自定义块、映射到目标比特、求逆。
block = q.Circuit(2).h(0).cx(0, 1)
larger = q.Circuit(4).compose(block, [3, 1])
restored = larger.compose(q.Circuit(4).compose(block.inverse(), [3, 1])).run()
assert np.isclose(restored.probabilities()[0], 1)
```

支持 H/X/Y/Z、S/T 及逆门、RX/RY/RZ、相位门、CX/CY/CZ/CP、SWAP、CCX/MCX、
QFT/IQFT、Pauli 旋转、最多四个目标比特的自定义酉门、线路组合与反演。
旋转方法使用 **`ry(angle, wire)`** 的参数顺序。

支持测量反馈和精确混态噪声：

```python
import openqml as q

# 测量会塌缩；run 的 seed 控制实际分支。
circuit = q.Circuit(2).h(0).cx(0, 1).measure(0, "m")
circuit.x(1, condition=("m", 1))
state = circuit.run(seed=3)
print(state.classical)  # {'m': 0} 或 {'m': 1}

# sample 只抽样、不塌缩；reset 是精确的无条件信道。
noisy = q.bell().amplitude_damping(0.1, 0).reset(1).run()
print(noisy.reduced_density())
```

`Circuit.run()` 每次从新的零态开始；`initial_state=` 会复制传入态。
`State` 也可直接调用门与 `measure`，在普通 Python 循环/分支中实现自己的算法。

## 训练量子模型

```python
import openqml as q

data = q.load_dataset("moons", n_samples=192, seed=7)
X_train, X_test, y_train, y_test = data.split(seed=7)

model = q.QuantumKernelClassifier(layers=2, regularization=0.01)
model.fit(X_train, y_train)
print(model.predict(X_test[:5]))
print("Held-out accuracy:", model.score(X_test, y_test))

vqc = q.VariationalQuantumClassifier(layers=2, epochs=60, seed=7)
vqc.fit(X_train, y_train)
print("Initial / best training loss:", vqc.history_[0], min(vqc.history_))
```

| 接口 | 实现与用途 |
| --- | --- |
| `QuantumKernel` | 数据重上传线路的精确保真度核、交叉核矩阵 |
| `QuantumKernelClassifier` | 核岭分类，支持多类别；不是 SVM |
| `VariationalQuantumClassifier` | 二分类、参数化线路、BCE 与逐门参数移位梯度 |
| `VariationalQuantumRegressor` | 数据重上传、量子期望值、可训练仿射输出头 |
| `LogisticClassifier` | 二分类逻辑回归经典基线 |
| `RidgeRegressor` | 线性岭回归经典基线 |
| `FourierRegressor` | 周期目标的傅里叶特征岭回归基线 |

模型提供 `fit`、`predict`、`score`、`get_params`；分类模型提供 `predict_proba`。
量子模型最多使用 8 个比特，输入特征数不能超过比特数。
特征缩放只从训练数据学习；核岭的 softmax 分数未经概率校准。
准确率与泛化能力取决于数据、编码、线路与优化，量子模型并不保证胜过经典基线。

## 数据集、任务、Suite 和可复现实验

```python
import openqml as q

print(q.list_datasets())
print(q.list_models())
suite = q.get_suite("qml-cls-1")  # xor、circles、moons

task = q.get_task("moons", n_samples=96, seed=1)
result = q.run_benchmark("quantum-kernel", task, seed=7)
print(result.metrics)
result.save("benchmark-results.json")

# 每个 seed 下，各模型使用同一份数据和同一个 train/test split。
comparison = q.compare_models(task, seeds=(1, 2, 3), epochs=40)
print(comparison.summary())  # 各 seed 指标的中位数
comparison.save("benchmark-results-comparison.json")
```

内置离线生成器：`xor`、`circles`、`moons`、`parity`、`sine`、`step`、`linear`。
另有 `qml-reg-1` 回归套件。自己的数据可使用 `q.make_task(X, y, task_type=...)`。
不会下载 MNIST、分子数据或网站远程数据；这些生成器是透明的小规模验证任务。

结果保存模型配置、数据 SHA-256、数据生成配置、划分 seed、依赖版本、训练/预测耗时、
参数数量与估算的线路计算次数。协议名是 `openqml-sdk-bench-v1`；
它与网站 `openqml-duel-1` 分开，默认模型也**没有做参数数量匹配**。
固定划分和 seed 的数值结果可复现，运行耗时和平台信息可能变化。

## VQE、QAOA 和梯度

```python
import openqml as q

H = q.Hamiltonian({"ZZ": -1, "XI": -0.5, "IX": -0.5})


def ansatz(p):
    return q.Circuit(2).ry(p[0], 0).ry(p[1], 1).cx(0, 1)


fit = q.vqe(H, ansatz, [0.4, 0.3], steps=100, learning_rate=0.08)
print(fit.value, H.ground_energy())

qaoa = q.qaoa_maxcut(3, [(0, 1), (1, 2)], gammas=[0.3], betas=[0.2])
print(qaoa.run().sample(1024, seed=7))
```

也可使用 `ghz`、`bell`、`grover`、`qft` 和 `hardware_efficient_ansatz` 生成线路。
`gradient` 是完整目标函数的中心有限差分；`minimize` 支持 Adam 和回溯梯度下降。
`parameter_shift` 的 ±π/2 公式仅对**每个参数独立出现一次、未缩放 Pauli 旋转的期望值**直接适用。
共享参数、受控旋转与非线性损失请使用链式法则或 `gradient`。

## 线路交换与 Qiskit

```python
import openqml as q

circuit = q.bell()
source = circuit.to_qasm()  # 明确的 OpenQASM 2 子集
restored = q.Circuit.from_qasm(source)
```

OpenQASM 2 子集不支持的语法会明确报错，不会静默丢门；源代码不经 `eval` / `exec`。
若需要导出到 Qiskit，在源码目录安装 `python -m pip install ".[qiskit]"`，
或者另行安装 Qiskit，随后调用 `circuit.to_qiskit()`。
该适配器支持酉门、噪声和测量反馈，但不会连接账号或提交真机任务。

## 模拟资源边界

| 功能 | 默认限制 |
| --- | --- |
| 线路描述 | 64 比特 |
| 精确纯态模拟 | 最多 40 比特；由占据基底数与工作预算进一步限制 |
| 完整状态向量 / 概率数组 | 最多 18 个请求比特 |
| 稀疏占据基底 | 262,144 个；另有内存预算 |
| 精确噪声 / reset | 整个寄存器最多 8 比特 |
| 约化密度矩阵读数 | 最多 8 个请求比特 |
| 自定义酉门与信道 | 最多 4 个目标比特 |
| 默认分配预算 | 256 MiB；按保守中间数组估算，不是操作系统进程内存硬限额 |
| 默认线路/工作预算 | 10,000 个操作 / 2,000 万估算标量工作量 |

**40 比特 GHZ 等稀疏态可以运行；任意高度纠缠的 40 比特线路不保证能运行。**
本版没有张量网络或稳定子专用后端，不会丢弃小振幅近似“跑完”。
预算不足时抛出 `ResourceLimitError`。大态使用边缘概率或少量比特读数：

```python
import openqml as q

state = q.ghz(40).run()
print(state.sample(1024, seed=7))
print(state.probabilities([0, 39]))  # 仅分配长度 4 的数组
```

需要调整时传入 `SimulationBudget`；其作用与局限见 [API 文档](docs/api.md)。

## 比特顺序

`q0` 是最低有效位。数组索引和采样字符串遵循 `q[n-1]...q0`，
Pauli 字符串也将 q0 放在最右侧。矩阵的 `targets=[2, 0]` 表示 wire 2 是最低有效**局部**位。
`probabilities([2, 0])` 的局部索引规则相同。
网站部分显示使用 q0 在左侧；本 SDK 的字符串不能直接按页面显示顺序比较。

## 示例与开发

```bash
python -m openqml demo --qubits 3 --seed 7
python -m openqml datasets
python -m openqml benchmark --task moons --seeds 1 2 3 --epochs 40 --output benchmark-results.json
python examples/quantum_programming.py
python examples/quantum_learning.py
python examples/reproducible_benchmark.py
```

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m build
```

CI 已配置 Python 3.10–3.13、Windows/Linux、NumPy 1.x/2.x、核心数值测试、安装包和可选 Qiskit 适配器，
将在仓库推送后运行。已完成的本地验证见 [验证记录](VALIDATION.md)。
测试包含独立矩阵参考、QFT/DFT、噪声物理性、传态测量分支、解析梯度、训练损失链式法则、
可复现划分以及资源限制。

- [API 说明](docs/api.md)
- [English quickstart](docs/quickstart-en.md)
- [贡献指南](CONTRIBUTING.md)
- [MIT License](LICENSE)
