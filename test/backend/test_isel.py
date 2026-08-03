import unittest
from typing import cast
from tinygrad import Device
from tinygrad.uop.ops import UOp, dtypes, graph_rewrite
from tinygrad.renderer.isa.x86 import X86Renderer, X86Ops, to_imm
from tinygrad.renderer.isa import IselContext, machine_const

# INDEX on a register value with a constant index extracts a single element (the old GEP)
def lane(y:UOp, i:int) -> UOp: return y.index(UOp.const(i, dtypes.int), dtype=y.dtype.scalar())

@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, X86Renderer), "only x86")
class TestIselX86(unittest.TestCase):
  def isel_rewrite(self, x:UOp):
    return graph_rewrite(x, cast(X86Renderer, Device[Device.DEFAULT].renderer).isel_matcher, IselContext(x), bottom_up=True)

  def _check_op(self, dt_op, expr):
    nargs = expr.__code__.co_argcount
    for dt,op in dt_op:
      with self.subTest(dtype=dt):
        v = [UOp.variable(str(i), 0, 0, dt) for i in range(nargs)]
        n = self.isel_rewrite(expr(*v))
        self.assertIs(n.arg, op)

  def test_cmove(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    b = UOp.variable("b", 0, 0, dtypes.int32)
    c = (a < b).where(a, b)
    d = (a != b).where(a, b)
    f = c + d
    n = self.isel_rewrite(f)
    self.assertTrue(n.src[0].arg is X86Ops.CMOVL and n.src[1].arg is X86Ops.CMOVNE)
    # both comparisons become the same instruction
    self.assertTrue(n.src[0].src[2] == n.src[1].src[2] and n.src[0].src[2].arg is X86Ops.CMP)

  def test_vinsertps(self):
    a = UOp.variable("a", 0, 0, dtypes.float32)
    b = UOp.variable("b", 0, 0, dtypes.float32)
    c = UOp.variable("c", 0, 0, dtypes.float32)
    d = UOp.variable("e", 0, 0, dtypes.float32)

    valid = [UOp.stack(lane(a, 0), lane(b, 1), lane(a, 2), lane(b, 3)),
             UOp.stack(lane(a, 3), lane(b, 2), lane(c, 1), d)]
    for shuf in valid: self.assertIs(self.isel_rewrite(shuf).arg, X86Ops.VINSERTPS)

  # complex address is [base + index*scale + displacement]
  def test_complex_address(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    load = UOp.param(0, dtypes.int32, (16,)).index(a + 1).load()
    n = self.isel_rewrite(load)
    # displacement is the constant in "a" scaled to the buffer element size; encoding chooses its width
    self.assertIs(n.src[2], machine_const(4))

  # a constant reaches isel as a strong CONST or as a cast weak one, both select the same machine const
  def test_to_imm_forms(self):
    for dt in (dtypes.int8, dtypes.uint8, dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64):
      for v in (0, 1, dt.min, dt.max): self.assertIs(to_imm(UOp.const(v, dt)), to_imm(UOp.const(v).cast(dt)), f"{dt} {v}")

  # a machine const states the number the hardware sees, and only fits an immediate if it fits 4 bytes
  def test_to_imm_value(self):
    self.assertIs(to_imm(UOp.const(-1, dtypes.uint32)), machine_const(2**32-1))
    for dt,v in ((dtypes.int64, 2**31), (dtypes.uint64, 2**32), (dtypes.weakint, 2**31)): self.assertIsNone(to_imm(UOp.const(v, dt)))

if __name__ == "__main__":
  unittest.main()
