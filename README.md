# OpenQML

OpenQML 提供可嵌入 PyTorch、JAX 和 Flax 的可微量子层，以及连接 Qiskit、PennyLane、Cirq 和 OpenQASM 的统一中间表示（IR），支持程序导入、优化、执行与导出，并对不兼容或无法无损转换的功能给出明确诊断。

## 核心能力

- **混合学习**：量子层、残差适配器与量子卷积支持批量输入，与经典网络共同训练；提供自动微分及流式参数位移梯度。
- **跨框架编程**：保留参数表达式、量子线映射、测量与分支相位，支持原生程序、JSON IR 和 OpenQASM 互转。
- **执行与测量**：提供状态矢量、密度矩阵、噪声通道、经典反馈及有限 shots 估计，可接入 Aer、MPS 和调用方提供的执行器。
- **线路优化**：折叠已知条件、合并稳定分支、简化相邻量子门，并返回优化统计和被消除的参数；支持资源限额与转换诊断。

## 安装

Python 3.10 及以上版本，基础依赖为 NumPy。在源码目录执行：

```bash
python -m pip install .
python -m pip install '.[torch]'       # PyTorch
python -m pip install '.[jax,flax]'    # JAX / Flax / Optax
python -m pip install '.[all,dev]'     # All integrations and development tools
```

框架依赖按需安装。当前版本为 `0.3.0a7`（Alpha），源码包附 wheel 与 sdist。

## 混合训练

量子模块可直接加入常规模型，由同一个优化器更新经典参数和量子权重。

```python
import torch
from openqml.interfaces.torch import QuantumLayer

model = torch.nn.Sequential(
    torch.nn.Linear(8, 4),
    QuantumLayer.hardware_efficient(4, depth=2),
    torch.nn.Linear(4, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
inputs = torch.randn(16, 8)

optimizer.zero_grad()
loss = model(inputs).square().mean()
loss.backward()
optimizer.step()
```

JAX 提供显式权重的函数接口；Flax 将量子权重纳入参数树。已有 PennyLane QNode 可通过 `NativePennyLaneLayer` 保留设备与微分图。

## 程序转换与优化

```python
from openqml import Circuit, ClassicalCondition, convert, optimize

circuit = (
    Circuit(2)
    .h(0)
    .measure(0, 0)
    .if_else(ClassicalCondition((0,), 1), Circuit(2).ry(0.2, 1))
    .if_else(ClassicalCondition((0,), 1), Circuit(2).ry(0.3, 1))
)
report = optimize(circuit)
print(report.before, report.after)

result = convert(circuit, target="openqasm3", optimize=True)
print(result.value)
print(result.diagnostics)
```

`QuantumLayer.from_program(...)` 将原生线路或 QASM 接入训练；`QuantumProgram.capture(...)` 保留原生执行入口。

量子层支持 `optimize=True`，保留原有输入顺序和权重形状，被消除的参数返回零梯度。`optimization_options` 设置优化资源限额。

`layer.to_spec()` 导出统一的 `LayerSpec`，通过 `to_json()` 保存，再由各框架的 `QuantumLayer.from_spec(...)` 重建。

`layer.checkpoint(...)` 将源线路、参数归属、可观测量和训练权重打包为可跨框架加载的 `LayerCheckpoint`；JAX / Flax 需显式传入权重向量。`from_checkpoint(...)` 在 PyTorch 返回模块，在 JAX 返回 `(layer, weights)`，在 Flax 返回 `(module, variables)`。加载时校验完整配置和精度，优化器状态另用框架原生检查点保存；float64 的 JAX / Flax 路径需启用 JAX x64。

## 支持范围

| 接口 | 主要用途 |
| --- | --- |
| PyTorch / JAX / Flax | 批量混合训练、输入与权重微分、统一优化 |
| Qiskit / PennyLane / Cirq | 原生程序接入、受支持子集的 IR 转换与执行 |
| OpenQASM 2 / 3 | 标准门与 QASM 3 条件控制；有限静态循环经 SDK 展开 |
| NumPy / Aer / MPS / provider | 本地模拟、有限 shots 与可扩展执行后端 |

默认张量路径支持高阶微分；外部数值桥目前限一阶。Cirq 支持无测量分支体的条件转换；PennyLane 原子控制、一般运行时经典计算、脉冲与时序仍有边界。

优化以逻辑 IR 语义为准，保留显式通道及测量写入。独立 IR 优化可能缩小参数集合，量子层接口会保留原有参数布局；依赖物理门序列的原生噪声模型需重新验证。GPU、Windows 与远程作业生命周期尚未完成验证。

## 开发与许可

```bash
python -m pytest -q
ruff check .
python -m build
python -m twine check dist/*
```

测试、基准脚本和示例分别位于 `tests/`、`benchmarks/`、`examples/`。采用 Apache-2.0 许可证。
