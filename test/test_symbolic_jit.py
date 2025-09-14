import unittest

from test.helpers import assert_jit_cache_len
from tinygrad import Variable, Tensor, TinyJit
import numpy as np

class TestSymbolicJit(unittest.TestCase):
  def test_matmul(self):
    def f(a, b): return (a@b).realize()
    jf = TinyJit(f)
    a = Tensor.rand(3, 10)
    b = Tensor.rand(10, 5)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(a[:, :vi], b[:vi, :]).numpy()
      expected = f(a[:, :i], b[:i, :]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_mixed_with_no_symbol_kernel(self):
    def f(a, b):
      s = (a@b).realize()
      s = (s+s).realize() # this one does not have symbols in input
      return s
    jf = TinyJit(f)
    a = Tensor.rand(3, 10)
    b = Tensor.rand(10, 5)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(a[:, :vi], b[:vi, :]).numpy()
      expected = f(a[:, :i], b[:i, :]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 2)

  def test_attention(self):
    def f(q, k, v): return Tensor.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).realize()
    jf = TinyJit(f)
    q = Tensor.rand(2, 1, 4, 8)
    k = Tensor.rand(2, 10, 4, 8)
    v = Tensor.rand(2, 10, 4, 8)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(q, k[:, :vi], v[:, :vi]).reshape(2, 4, 1, 8).numpy()
      expected = f(q, k[:, :i], v[:, :i]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 5)

  def test_jit_symbolic_shape_mismatch(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    a = Tensor.rand(3, 10)
    b = Tensor.rand(3, 10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      add(a[:, :vi], b[:, :vi])
    vi2 = Variable("i", 1, 10).bind(7)
    a = Tensor.rand(3, 7)[:, :vi2]
    bad = Tensor.rand(4, 7)[:, :vi2]
    with self.assertRaises(AssertionError):
      add(a, bad)

  def test_shrink(self):
    # shrink is a movement, so we pair it with a simple function to test the JIT interaction
    def f(a): return (a+1).realize()
    jf = TinyJit(f)
    a = Tensor.rand(7, 11)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = a.shrink(((3,5),(vi,vi+2)))
      symbolic = jf(symbolic).numpy()
      expected = f(a.shrink(((3,5),(i,i+2)))).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_slice(self):
    # slice is a movement, so we pair it with a simple function to test the JIT interaction
    def f(a): return (a+1).realize()
    jf = TinyJit(f)
    a = Tensor.rand(7, 11)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = a[3:5, vi:vi+2]
      symbolic = jf(symbolic).numpy()
      expected = f(a[3:5, i:i+2]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_ones_sum(self):
    def f(a): return a.sum().realize()
    jf = TinyJit(f)
    t = Tensor.ones(10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(t[:vi]).item()
      expected = f(t[:i]).item()
      np.testing.assert_equal(symbolic, expected)

if __name__ == '__main__':
  unittest.main()
