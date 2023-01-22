from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable

from .value import Value


class Module(ABC):
    @abstractmethod
    def parameters(self) -> list[Value]:
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters():
            p._grad = 0


class Neuron(Module):
    def __init__(self, num_weights: int, activation: Callable[[Value], Value] = lambda v: v.tanh()) -> None:
        self._weights = [Value(random.uniform(-1, 1)) for _ in range(num_weights)]
        self._bias = Value(random.uniform(-1, 1))
        self._activation = activation

    def __call__(self, inputs: list[Value]) -> Value:
        return self._activation(sum((w * x for w, x in zip(self._weights, inputs)), self._bias))

    def parameters(self) -> list[Value]:
        return self._weights + [self._bias]


class Layer(Module):
    def __init__(self, num_in: int, num_out: int, activation: Callable[[Value], Value] = lambda v: v.tanh()) -> None:
        self._neurons = [Neuron(num_in, activation) for _ in range(num_out)]

    def __call__(self, inputs: list[Value]) -> list[Value]:
        return [n(inputs) for n in self._neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self._neurons for p in n.parameters()]


class MLP(Module):
    def __init__(
        self, num_in: int, num_outs: list[int], activation: Callable[[Value], Value] = lambda v: v.tanh()
    ) -> None:
        self._layers = [Layer(num_in, num_outs[0], activation)]
        for i in range(1, len(num_outs)):
            self._layers.append(Layer(num_outs[i - 1], num_outs[i]))

    def __call__(self, inputs: list[Value]) -> list[Value]:
        for layer in self._layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self) -> list[Value]:
        return [p for layer in self._layers for p in layer.parameters()]


class MSE:
    def __call__(self, preds: list[Value], targets: list[Value]) -> Value:
        assert len(preds) == len(targets)
        return sum((pred - target) ** 2 for pred, target in zip(preds, targets)) / len(preds)
