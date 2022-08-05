[![pipeline](https://github.com/michaelosthege/aesara-federated/workflows/test/badge.svg)](https://github.com/michaelosthege/aesara-federated/actions)

# `aesara-federated`
This package implements federated computing with [Aesara](https://github.com/aesara-devs/aesara).

Using `aesara-federated`, differentiable cost functions can be computed on federated nodes.
Inputs and outputs are transmitted in binary via a bidirectional gRPC stream.

A client side `LogpGradOp` is provided to conveniently embed federated compute operations in Aesara graphs such as a [PyMC](https://github.com/pymc-devs/pymc) model.

The example code implements a simple Bayesian linear regression to data that is "private" to the federated compute process.

Run each command in its own terminal:

```bash
python demo_node.py
```

```bash
python demo_model.py
```

## Installation
```bash
conda env create -f environment.yml
```

## Contributing
Additional dependencies are needed to compile the [protobufs](./protobufs/):

```bash
conda install -c conda-forge protobuf
pip install --pre betterproto[compiler]
```

```bash
python protobufs/generate.py
```

Set up `pre-commit` for automated code style enforcement:

```bash
pip install pre-commit
pre-commit install
```
