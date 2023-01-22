from __future__ import annotations

from micrograd.value import Value


class SGD:
    def __init__(self, params: list[Value], lr=0.01):
        self._params = params
        self._lr = lr

    def step(self) -> None:
        for p in self._params:
            p._value -= self._lr * p.grad

    def zero_grad(self) -> None:
        for p in self._params:
            p._grad = 0
