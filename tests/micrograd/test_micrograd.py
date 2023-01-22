from __future__ import annotations

import math

import pytest

from micrograd.value import Value


class TestValue:
    def test_eq(self):
        a = Value(2.0)
        b = Value(2.0)
        c = Value(3.0)
        assert a == b
        assert a != c

    def test_add(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c == Value(5.0)
        d = c + 1
        assert d == Value(6.0)
        e = 1 + d
        assert e == Value(7.0)

    def test_mult(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c == Value(6.0)
        d = c * 2.0
        assert d == Value(12.0)
        e = 2.0 * d
        assert e == Value(24.0)

    def test_pow(self):
        a = Value(3.0)
        b = a**5
        assert b == Value(3**5)
        c = a**4.0
        assert c == Value(3**4)

    def test_div(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a / b
        assert c == Value(2.0 / 3.0)
        d = c / 2.0
        assert d == Value(1.0 / 3.0)
        f = 1 / d
        assert f == Value(3.0)

    def test_sub(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a - b
        assert c == Value(-1.0)
        d = c - 1
        assert d == Value(-2.0)
        e = 1 - d
        assert e == Value(3.0)

    def test_exp(self):
        a = Value(2.0)
        b = a.exp()
        assert b == Value(math.exp(2.0))

    def test_tanh(self):
        a = Value(2.0)
        b = a.tanh()
        assert b == Value(math.tanh(2.0))

    def test_backward(self):
        x1 = Value(2.0)
        x2 = Value(0.0)
        w1 = Value(-3.0)
        w2 = Value(1.0)
        b = Value(math.atanh(0.5**0.5) + 6)
        n = x1 * w1 + x2 * w2 + b
        o = n.tanh()

        assert o == pytest.approx(0.7071, rel=1e-5)
        o.backward()
        assert x1.grad == pytest.approx(-1.5, rel=1e-5)
        assert x2.grad == pytest.approx(0.5, rel=1e-5)
        assert w1.grad == pytest.approx(1.0, rel=1e-5)
        assert w2.grad == pytest.approx(0.0, rel=1e-5)
        assert b.grad == pytest.approx(0.5, rel=1e-5)
