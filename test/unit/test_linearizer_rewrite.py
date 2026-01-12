import unittest
from tinygrad import Tensor, Context, Device
from tinygrad.engine.realize import get_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import KernelInfo

class TestLinearizerRewrite(unittest.TestCase):
  def test_reduction(self):
    t = Tensor.ones((64,64), device="NULL").contiguous().realize()
    out = (t*2).sum(axis=1)
    with Context(SPLIT_REDUCEOP=0, DEVECTORIZE=0):
      si = out.schedule()[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.UPCAST, 0, 4))
      opts_to_apply.append(Opt(OptOps.UNROLL, 0, 4))
      ast = si.ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = get_program(ast, Device["CPU"].renderer)
      print(prg.src)

  def test_arange(self):
    out = Tensor.arange(32, device="NULL")
    with Context(SPLIT_REDUCEOP=0, DEVECTORIZE=0):
      si = out.schedule()[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.UPCAST, 0, 4))
      ast = si.ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = get_program(ast, Device["CPU"].renderer)
      print(prg.src)

  def test_kernel_info(self):
    out = Tensor.arange(4, device="NULL")
    si = out.schedule()[-1]

    ast = si.ast.replace(arg=KernelInfo(opts_to_apply=()))
    prg = get_program(ast, Device["CPU"].renderer)
    assert prg.applied_opts == (), f"expected no opts, got {prg}"

    prg = get_program(ast.replace(arg=None), Device["CPU"].renderer)
    assert prg.applied_opts != (), f"expected opts to apply, got {prg.applied_opts}"

    prg = get_program(ast.replace(arg=KernelInfo(name="custom")), Device["CPU"].renderer)
    self.assertEqual(prg.name, "custom")

    # explicitly setting name="test" should also be preserved (not auto-generated)
    prg = get_program(ast.replace(arg=KernelInfo(name="test")), Device["CPU"].renderer)
    self.assertEqual(prg.name, "test")

  def test_kernel_name_preserved(self):
    # this test verifies that explicit kernel names are preserved through codegen
    # these tests would have failed before the fix (name="test" was treated as default and auto-generated)
    out = Tensor.arange(10, device="NULL")
    si = out.schedule()[-1]

    # test 1: custom name is preserved
    prg = get_program(si.ast.replace(arg=KernelInfo(name="my_kernel")), Device["CPU"].renderer)
    self.assertEqual(prg.name, "my_kernel")
    self.assertIn("my_kernel", prg.src)

    # test 2: explicitly setting name="test" is preserved (NOT auto-generated)
    prg = get_program(si.ast.replace(arg=KernelInfo(name="test")), Device["CPU"].renderer)
    self.assertEqual(prg.name, "test")
    self.assertEqual(prg.function_name, "test")

    # test 3: no name (None) triggers auto-generation
    prg = get_program(si.ast.replace(arg=KernelInfo()), Device["CPU"].renderer)
    self.assertNotEqual(prg.name, "test")  # auto-generated names like "E_10" are used

  def test_apply_opts_preserves_name(self):
    # test that name is preserved when apply_opts is used (opts_to_apply triggers apply_opts)
    # this would have failed on master because name="test" was treated as default
    t = Tensor.ones((64,64), device="NULL").contiguous().realize()
    out = (t*2).sum(axis=1)
    with Context(SPLIT_REDUCEOP=0, DEVECTORIZE=0):
      si = out.schedule()[-1]
      opts_to_apply = [Opt(OptOps.UPCAST, 0, 4)]

      # test 1: custom name preserved with opts_to_apply
      ast = si.ast.replace(arg=KernelInfo(name="my_optimized_kernel", opts_to_apply=tuple(opts_to_apply)))
      prg = get_program(ast, Device["CPU"].renderer)
      self.assertEqual(prg.name, "my_optimized_kernel")
      self.assertIn("my_optimized_kernel", prg.src)
      self.assertEqual(prg.applied_opts, tuple(opts_to_apply))

      # test 2: name="test" preserved with opts_to_apply (would fail on master)
      ast = si.ast.replace(arg=KernelInfo(name="test", opts_to_apply=tuple(opts_to_apply)))
      prg = get_program(ast, Device["CPU"].renderer)
      self.assertEqual(prg.name, "test")
      self.assertEqual(prg.function_name, "test")

      # test 3: no name with opts_to_apply triggers auto-generation
      ast = si.ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = get_program(ast, Device["CPU"].renderer)
      self.assertNotEqual(prg.name, "test")

if __name__ == '__main__':
  unittest.main()
