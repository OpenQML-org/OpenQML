# OpenQML

[PyPI](https://pypi.org/project/openqml/) · [CI](https://github.com/OpenQML-org/OpenQML/actions/workflows/ci.yml) · [文档](https://github.com/OpenQML-org/OpenQML/tree/main/docs)

## 量子-经典混合AI：

用 PyTorch、JAX 或 Flax 构建和训练混合神经网络。提供统一的 `fit` / `predict` 接口，也可直接使用原生网络层、训练器和张量网络执行后端。

## 量子-经典混合量化：

支持收益分布建模、风险估计、约束资产配置、交易成本核算和衍生品定价。可微金融计算与混合网络共用原生张量；量子估计提供精度、噪声和调用预算控制。

## 量子编程跨框架互操作：

通过统一电路表示连接 Qiskit、Cirq、PennyLane 和 OpenQASM。转换、编译和执行使用同一套 API，具体支持范围见[文档](https://github.com/OpenQML-org/OpenQML/tree/main/docs)。

## 安装

需要 Python 3.10 或更高版本：

```bash
python -m pip install -U openqml
```

核心依赖为 NumPy、packaging 和 psutil。训练框架、量子 SDK 和执行后端按需加载。

## 快速开始

从收益观测计算资产权重：

```python
import openqml as oq

returns = [[0.01, -0.02], [-0.01, 0.03], [0.02, 0.01]]
portfolio = oq.finance.allocate(returns, caps=0.7, fees=0.001)
print(portfolio.weights)
```

训练模型前，安装 `openqml[torch]`、`openqml[jax]` 或 `openqml[flax]`。以下使用 PyTorch：

```bash
python -m pip install -U "openqml[torch]"
```

```python
import openqml as oq

x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 8
y = [0, 1, 1, 0] * 8
model = oq.Model(task="classification", hidden_features=8, depth=2)
model.fit(x, y)
print(model.predict(x[:2]))
```

设置 `framework="jax"` 或 `framework="flax"` 切换训练框架。`backend` 选择电路执行后端，`device` 选择原生张量设备。量子比特数由计算资源和后端能力约束，没有固定的库级上限。

量子查询优势依赖高效的相干 Oracle、噪声水平和完整执行成本，不代表已获得实机加速或投资超额收益。

[API](https://github.com/OpenQML-org/OpenQML/blob/main/docs/api.md) · [混合网络](https://github.com/OpenQML-org/OpenQML/blob/main/docs/models.md) · [量化](https://github.com/OpenQML-org/OpenQML/blob/main/docs/finance.md) · [训练](https://github.com/OpenQML-org/OpenQML/blob/main/docs/hybrid-training.md) · [张量网络](https://github.com/OpenQML-org/OpenQML/blob/main/docs/tensor-networks.md) · [运行时](https://github.com/OpenQML-org/OpenQML/blob/main/docs/runtime.md)

## 许可证

[Apache-2.0](LICENSE)
