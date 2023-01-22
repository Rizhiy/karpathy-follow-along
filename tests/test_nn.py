from __future__ import annotations

import pytest

from micrograd.nn import MLP, MSE, Layer, Neuron
from micrograd.value import Value


def test_neuron():
    in_size = 4
    neuron = Neuron(in_size)
    inputs = [Value(i) for i in range(in_size)]
    out = neuron(inputs)
    assert isinstance(out, Value)
    assert -1 <= out <= 1


def test_layer():
    in_size = 4
    out_size = 3
    layer = Layer(in_size, out_size)
    inputs = [Value(i) for i in range(in_size)]
    out = layer(inputs)
    assert isinstance(out, list)
    assert len(out) == out_size
    for o in out:
        assert isinstance(o, Value)
        assert -1 <= o <= 1


def test_mlp():
    in_size = 3
    out_sizes = [4, 4, 1]
    mlp = MLP(in_size, out_sizes)
    inputs = [Value(i) for i in range(in_size)]
    out = mlp(inputs)
    assert isinstance(out, list)
    assert len(out) == out_sizes[-1]
    for o in out:
        assert isinstance(o, Value)
        assert -1 <= o <= 1


def test_mlp_parameters():
    in_size = 3
    out_sizes = [4, 4, 1]
    mlp = MLP(in_size, out_sizes)
    params = mlp.parameters()
    assert isinstance(params, list)
    all_sizes = [in_size] + out_sizes
    total_params = sum((i + 1) * o for i, o in zip(all_sizes[:-1], all_sizes[1:]))
    assert len(params) == total_params
    for p in params:
        assert isinstance(p, Value)


def test_mse():
    preds = [Value(0.7), Value(0.5)]
    targets = [Value(1), Value(0)]
    mse = MSE()
    loss = mse(preds, targets)
    assert isinstance(loss, Value)
    assert loss == pytest.approx((0.3**2 + 0.5**2) / 2)
