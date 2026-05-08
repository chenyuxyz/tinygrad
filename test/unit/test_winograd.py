import unittest, sys
import numpy as np
from tinygrad import Tensor, GlobalCounters, Context, nn, dtypes
from tinygrad.helpers import WINO

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinogradClose(unittest.TestCase):
  def test_close(self):
    inp = Tensor.rand(1, 16, 16, 16)
    conv = nn.Conv2d(16, 16, 3)
    conv(inp).realize() # warmup
    GlobalCounters.reset()
    print("non winograd")
    with Context(WINO=0):
      cmp = conv(inp).realize() # warmup
    GlobalCounters.reset()
    print("winograd")
    with Context(WINO=1):
      test = conv(inp).realize()
    np.testing.assert_allclose(cmp.numpy(), test.numpy(), atol=1e-5)

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinograd(unittest.TestCase):
  def setUp(self):
    self.old = WINO.value
    WINO.value = 1
  def tearDown(self):
    WINO.value = self.old

  def test_padded_conv2d(self):
    # tests padding order in winograd
    x,w = Tensor.rand(1,3,11,28).realize(), Tensor.rand(4,3,3,3).realize()
    with Context(WINO=0): expected = Tensor.conv2d(x,w,padding=1).realize()
    with Context(WINO=1): result = Tensor.conv2d(x,w,padding=1).realize()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

  def test_handwritten_conv_triggers_wino(self):
    # 3^n conv built directly from _pool + multiply + sum (i.e. not via Tensor.conv2d).
    # Master gates wino inside Tensor.conv2d, so any of these handwritten variants would NOT
    # fire there. The rewrite rule version fires on the produced UOp graph regardless of how
    # it was built — exercising the generality of the affine-detection approach.
    def manual_conv(x, w, swap_mul=False, downstream=lambda r: r):
      bs, cin, cout, HW = x.shape[0], x.shape[1], w.shape[0], w.shape[2:]
      pooled = x._pool(HW, 1, 1)
      oyx = pooled.shape[2:-len(HW)]
      pooled = pooled.reshape(bs, 1, cin, 1, *oyx, *HW).expand(bs, 1, cin, cout, *oyx, *HW)\
        .permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])
      reshaped_w = w.reshape(1, 1, cout, *[1]*len(oyx), cin, *HW)
      mul = (reshaped_w * pooled) if swap_mul else (pooled * reshaped_w)
      return downstream(mul.sum([-1-i for i in range(1+len(oyx))], keepdim=True).reshape(bs, cout, *oyx))
    cases = [
      ("1D",                  Tensor.rand(1,4,9).realize(),       Tensor.rand(4,4,3).realize(),       {}),
      ("2D",                  Tensor.rand(1,4,9,9).realize(),     Tensor.rand(4,4,3,3).realize(),     {}),
      ("3D",                  Tensor.rand(1,4,9,9,9).realize(),   Tensor.rand(4,4,3,3,3).realize(),   {}),
      ("swapped MUL operands",Tensor.rand(1,4,9,9).realize(),     Tensor.rand(4,4,3,3).realize(),     {"swap_mul": True}),
      ("downstream relu",     Tensor.rand(1,4,9,9).realize(),     Tensor.rand(4,4,3,3).realize(),     {"downstream": lambda r: r.relu()}),
    ]
    for name, x, w, kw in cases:
      with self.subTest(name=name):
        with Context(WINO=0): base = len(manual_conv(x, w, **kw).schedule_linear().src)
        with Context(WINO=1): wino = len(manual_conv(x, w, **kw).schedule_linear().src)
        self.assertGreater(wino, base, f"{name}: wino did not fire (base={base}, wino={wino})")

  def test_mixed_dtype_accumulate_triggers_wino(self):
    x = Tensor.rand(1,4,9,9).cast(dtypes.half).realize()
    w = Tensor.rand(4,4,3,3).cast(dtypes.half).realize()
    with Context(WINO=0):
      expected = Tensor.conv2d(x, w, padding=1, dtype=dtypes.float32).realize()
      base = len(Tensor.conv2d(x, w, padding=1, dtype=dtypes.float32).schedule_linear().src)
    with Context(WINO=1):
      result = Tensor.conv2d(x, w, padding=1, dtype=dtypes.float32).realize()
      wino = len(Tensor.conv2d(x, w, padding=1, dtype=dtypes.float32).schedule_linear().src)
    self.assertGreater(wino, base)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-2, rtol=1e-2)

  def test_5x5_conv_triggers_wino(self):
    x = Tensor.rand(1,4,16,16).realize()
    w = Tensor.rand(8,4,5,5).realize()
    with Context(WINO=0):
      expected = Tensor.conv2d(x, w, padding=2).realize()
      base = len(Tensor.conv2d(x, w, padding=2).schedule_linear().src)
    with Context(WINO=1):
      result = Tensor.conv2d(x, w, padding=2).realize()
      wino = len(Tensor.conv2d(x, w, padding=2).schedule_linear().src)
    self.assertGreater(wino, base)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4, rtol=1e-4)

  def test_dilation2_triggers_wino(self):
    x = Tensor.rand(1,4,17,17).realize()
    w = Tensor.rand(4,4,3,3).realize()
    with Context(WINO=0):
      expected = Tensor.conv2d(x, w, padding=2, dilation=2).realize()
      base = len(Tensor.conv2d(x, w, padding=2, dilation=2).schedule_linear().src)
    with Context(WINO=1):
      result = Tensor.conv2d(x, w, padding=2, dilation=2).realize()
      wino = len(Tensor.conv2d(x, w, padding=2, dilation=2).schedule_linear().src)
    self.assertGreater(wino, base)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4, rtol=1e-4)

  def test_conv_transpose2d_triggers_wino(self):
    x = Tensor.rand(1,4,9,9).realize()
    w = Tensor.rand(4,4,3,3).realize()
    with Context(WINO=0):
      expected = Tensor.conv_transpose2d(x, w, stride=1, padding=0).realize()
      base = len(Tensor.conv_transpose2d(x, w, stride=1, padding=0).schedule_linear().src)
    with Context(WINO=1):
      result = Tensor.conv_transpose2d(x, w, stride=1, padding=0).realize()
      wino = len(Tensor.conv_transpose2d(x, w, stride=1, padding=0).schedule_linear().src)
    self.assertGreater(wino, base)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4, rtol=1e-4)

  def test_conv_transpose2d_5x5_and_bias(self):
    # 5x5 transposed conv (F(2x2,5x5)) plus bias
    x = Tensor.rand(1,4,13,13).realize()
    w = Tensor.rand(4,8,5,5).realize()
    b = Tensor.rand(8).realize()
    with Context(WINO=0):
      expected = Tensor.conv_transpose2d(x, w, b, stride=1).realize()
      base = len(Tensor.conv_transpose2d(x, w, b, stride=1).schedule_linear().src)
    with Context(WINO=1):
      result = Tensor.conv_transpose2d(x, w, b, stride=1).realize()
      wino = len(Tensor.conv_transpose2d(x, w, b, stride=1).schedule_linear().src)
    self.assertGreater(wino, base)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4, rtol=1e-4)

  def test_dilation2_with_bias(self):
    x = Tensor.rand(1,4,17,17).realize()
    w = Tensor.rand(8,4,3,3).realize()
    b = Tensor.rand(8).realize()
    with Context(WINO=0): expected = Tensor.conv2d(x, w, b, padding=2, dilation=2).realize()
    with Context(WINO=1): result = Tensor.conv2d(x, w, b, padding=2, dilation=2).realize()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4, rtol=1e-4)

  def test_stride2_does_not_misfire(self):
    # Wino is provably not a net win for stride > 1: 36 muls/4 outputs (stride=2) vs 9 muls/output direct.
    # Verify the matcher correctly declines stride=2 conv (kernel count stays at non-wino baseline).
    x, w = Tensor.empty(1,4,9,9).realize(), Tensor.empty(4,4,3,3).realize()
    with Context(WINO=0): base = len(Tensor.conv2d(x, w, stride=2).schedule_linear().src)
    with Context(WINO=1): wino = len(Tensor.conv2d(x, w, stride=2).schedule_linear().src)
    self.assertEqual(wino, base, "wino should not fire for stride=2 (provable loss vs direct)")

if __name__ == '__main__':
  unittest.main(verbosity=2)
