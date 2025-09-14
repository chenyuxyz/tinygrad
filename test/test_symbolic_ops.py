import unittest
from tinygrad import Tensor, Variable, GlobalCounters
from tinygrad.shape.shapetracker import View
from tinygrad.uop.ops import sym_infer
from examples.gpt2 import Attention
import numpy as np

class TestSymbolicOps(unittest.TestCase):
  def test_matmul(self):
    def f(a, b): return (a@b).realize()
    a = Tensor.rand(3, 10)
    b = Tensor.rand(10, 5)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = f(a[:, :vi], b[:vi, :]).numpy()
      expected = f(a[:, :i], b[:i, :]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_attention(self, dropout_p=0.0, imin=1, imax=5, use_symbolic=True):
    def f(q, k, v): return Tensor.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p).realize()
    q = Tensor.rand(2, 1, 4, 8)
    k = Tensor.rand(2, 10, 4, 8)
    v = Tensor.rand(2, 10, 4, 8)
    for i in range(imin, imax):
      vi = Variable("i", 1, 10).bind(i) if use_symbolic else i
      Tensor.realize(q, k, v)
      GlobalCounters.reset()
      symbolic = f(q, k[:, :vi, :, :], v[:, :vi, :, :]).reshape(2, 4, 1, 8).numpy()
      expected = f(q, k[:, :i, :, :], v[:, :i, :, :]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_attention_cmp_symbolic(self):
    # symbolic isn't seeing if i == i, so it's not putting them on the same axis
    self.test_attention(imin=4, imax=5, use_symbolic=False)
    self.test_attention(imin=4, imax=5, use_symbolic=True)

  # until this works, symbolic single kernel softmax won't
  @unittest.expectedFailure
  def test_attention_simple_view(self):
    i = Variable("i", 2, 10)
    v1 = View.create((2,4,1,i,i), ((i*4),i,0,0,1))
    v2 = View.create((2,4,1,i,i,i), (((i*i)*4),(i*i),0,0,i,1))
    self.assertIsNotNone(v1+v2)

  def test_attention_training(self):
    with Tensor.train():
      self.test_attention(dropout_p=0.0)
      with self.assertRaises(ValueError):
        # symbolic shape dropout is not supported
        self.test_attention(dropout_p=0.5)

  def test_attention_pos_0_sz_0(self):
    Attention(128, 8)(Tensor.ones(1, 0, 128), Variable("start_pos", 0, 128).bind(0), None)

  def test_attention_pos_0_sz_1(self):
    Attention(128, 8)(Tensor.ones(1, 1, 128), Variable("start_pos", 0, 128).bind(0), None)

  def test_attention_pos_0_sz_2(self):
    Attention(128, 8)(Tensor.ones(1, 2, 128), Variable("start_pos", 0, 128).bind(0), None)

  def test_shrink(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(7, 11)
      symbolic = a.shrink(((3,5),(vi,vi+2)))
      symbolic = symbolic.numpy()
      expected = a.shrink(((3,5),(i,i+2))).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_slice(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(7, 11)
      symbolic = a[3:5, vi:vi+2]
      symbolic = symbolic.numpy()
      expected = a[3:5, i:i+2].numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_ones_sum(self):
    t = Tensor.ones(10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = t[:vi].sum().item()
      expected = t[:i].sum().item()
      np.testing.assert_equal(symbolic, expected)

  @unittest.expectedFailure
  def test_conv2d_ceildiv_edge_case(self):
    v = Variable('v', 11, 50_000)
    val = 39601
    x = Tensor.randn(1, 22, 50_000)[:, :, :v.bind(val)]
    weight = Tensor.randn(256, 22, 12)

    result = x.conv2d(weight=weight, groups=1, stride=6, dilation=1, padding=(3, 3))
    var_val = {v: val}
    shape = tuple(sym_infer(s, var_val) for s in result.shape)
    self.assertEqual(shape, (1, 256, 6600))  # TODO: fails if ceildiv is incorrect
    # TODO: test output is correct

if __name__ == '__main__':
  unittest.main()
