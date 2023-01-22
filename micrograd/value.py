from __future__ import annotations

import math
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING


class Operation(Enum):
    ADD = auto()
    MUL = auto()
    TANH = auto()
    EXP = auto()
    POW = auto()


if TYPE_CHECKING:
    AcceptableTypes = float | int | Value


class Value:
    def __init__(self, value: float, _children: tuple[Value, ...] = (), _op: Operation = None, label="") -> None:
        self._value = value
        self._children = set(_children)
        self._op = _op
        self._label = label

        self._grad = 0.0
        self._backward = lambda: None

    @property
    def value(self) -> float:
        return self._value

    @property
    def grad(self) -> float:
        return self._grad

    @property
    def children(self) -> set[Value]:
        return self._children

    @property
    def operation(self) -> Operation:
        return self._op

    @property
    def label(self) -> str:
        return self._label

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"Value({self._value})"

    def __hash__(self) -> int:
        return hash(self._value)

    # Functions
    @staticmethod
    def _convert_other(other: AcceptableTypes) -> Value:
        if isinstance(other, Value):
            return other
        return Value(other)

    def __eq__(self, other: AcceptableTypes) -> bool:
        other = self._convert_other(other)
        return isinstance(other, Value) and self._value == other._value

    def __add__(self, other: Value) -> Value:
        other = self._convert_other(other)
        out = Value(self._value + other._value, (self, other), Operation.ADD)

        def _backward():
            self._grad += out.grad
            other._grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: AcceptableTypes) -> Value:
        return self + other

    def __mul__(self, other: Value) -> Value:
        other = self._convert_other(other)
        out = Value(self._value * other._value, (self, other), Operation.MUL)

        def _backward():
            self._grad += out.grad * other.value
            other._grad += out.grad * self.value

        out._backward = _backward
        return out

    def __rmul__(self, other: AcceptableTypes) -> Value:
        return self * other

    def __pow__(self, power: float) -> Value:
        assert isinstance(power, int | float), "Only supporting int and float for now"
        out = Value(self.value**power, (self,), Operation.POW)

        def _backward():
            self._grad += out.grad * power * self.value ** (power - 1)

        out._backward = _backward
        return out

    def __truediv__(self, other: AcceptableTypes) -> Value:
        return self * self._convert_other(other) ** -1

    def __rtruediv__(self, other: AcceptableTypes) -> Value:
        return self._convert_other(other) / self

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value) -> Value:
        return self + -other

    def __rsub__(self, other: AcceptableTypes) -> Value:
        return self._convert_other(other) - self

    def exp(self) -> Value:
        out = Value(math.exp(self.value), (self,), Operation.EXP)

        def _backward():
            self._grad += out.grad * out.value

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        out = Value(math.tanh(self.value), (self,), Operation.TANH)

        def _backward():
            self._grad += out.grad * (1 - out.value**2)

        out._backward = _backward
        return out

    def backward(self, child=False) -> None:
        self._grad = 1.0
        all_children = deque([self])
        while all_children:
            child = all_children.popleft()
            child._backward()
            all_children.extend(child.children)
