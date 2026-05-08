import unittest, sys
from tinygrad import Tensor, GlobalCounters, dtypes, Context
from tinygrad.helpers import CI, Profiling, WINO

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinograd(unittest.TestCase):
  def setUp(self):
    self.old = WINO.value
    WINO.value = 1
  def tearDown(self):
    WINO.value = self.old

  def test_profile(self):
    x,w = Tensor.rand(1,4,9,9).realize(), Tensor.rand(4,4,3,3).realize()
    with Profiling(enabled=not CI, sort='time'):
      Tensor.conv2d(x,w).realize()

  def test_forward_kernels(self):
    x,w = Tensor.rand(1,4,9,9).realize(), Tensor.rand(4,4,3,3).realize()
    out = Tensor.conv2d(x,w)
    self.assertEqual(len(out.schedule_linear().src), 2)

  def test_cin1_and_depthwise_trigger_wino(self):
    x1, w1 = Tensor.empty(1,1,9,9).realize(), Tensor.empty(4,1,3,3).realize()
    self.assertEqual(len(Tensor.conv2d(x1, w1, padding=1).schedule_linear().src), 2)
    xd, wd = Tensor.empty(1,4,9,9).realize(), Tensor.empty(4,1,3,3).realize()
    self.assertEqual(len(Tensor.conv2d(xd, wd, padding=1, groups=4).schedule_linear().src), 2)

  def test_backward_kernels(self):
    # NOTE: out.mean() collapses the conv to a constant scalar, so its backward graph has no real
    # forward-conv structure left to rewrite. Use a real loss so the forward conv survives in the
    # backward graph and pm_wino fires on it without needing a dedicated gradient hook.
    x,w = Tensor.empty(1,4,9,9,requires_grad=True).realize(), Tensor.empty(4,4,3,3,requires_grad=True).realize()
    y = Tensor.empty(1,4,9,9).realize()
    out = Tensor.conv2d(x,w, padding=1)
    ((out - y)**2).sum().backward()
    backward_schedule = x.grad.schedule_linear(w.grad)
    self.assertEqual(len(backward_schedule.src), 4)

  def test_counters(self):
    IC, OC, X, Y = 4,4,9,9
    x,w = Tensor.rand(1,IC,Y,X).realize(), Tensor.rand(OC,IC,3,3).realize()
    GlobalCounters.reset()
    with Context(WINO=1):
      Tensor.conv2d(x,w).realize()
    ops_wino, mem_wino = GlobalCounters.global_ops, GlobalCounters.global_mem
    GlobalCounters.reset()
    with Context(WINO=0):
      Tensor.conv2d(x,w).realize()
    ops_normal, mem_normal = GlobalCounters.global_ops, GlobalCounters.global_mem

    ops_ratio, mem_ratio = ops_wino/ops_normal, mem_wino/mem_normal
    print(f"ops: normal {ops_normal:9d} wino {ops_wino:9d} ratio {ops_ratio:.2f}")
    print(f"mem: normal {mem_normal:9d} wino {mem_wino:9d} ratio {mem_ratio:.2f}")

    # TODO: what's optimal on this?
    self.assertLess(ops_ratio, 4.3)
    self.assertLess(mem_ratio, 4)

  def test_dtype(self):
    IC, OC, X, Y = 4,4,9,9
    x,w = Tensor.empty(1,IC,Y,X), Tensor.empty(OC,IC,3,3)
    self.assertEqual(Tensor.conv2d(x,w).dtype, dtypes.default_float)

    x,w = Tensor.empty(1,IC,Y,X,dtype=dtypes.half), Tensor.empty(OC,IC,3,3,dtype=dtypes.half)
    self.assertEqual(Tensor.conv2d(x,w).dtype, dtypes.half)

if __name__ == '__main__':
  unittest.main(verbosity=2)
