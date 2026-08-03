import unittest
from tinygrad import Device
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.renderer.isa.x86 import X86Ops, X86Renderer, RBP, RDI, RSP, RSI, RAX, RDX, XMM, GPR, machine_const, def_reg

def ins(op, dt, src, tag=None): return UOp(Ops.INS, arg=op, dtype=dt, src=src, tag=tag)

@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, X86Renderer), "only on x86")
class TestEncodingsX86(unittest.TestCase):
  # NOTE: x86 supports a single displacement as memory address and index without base memory address
  # these have no use cases so they aren't supported
  def encode(self, u:UOp): return Device[Device.DEFAULT].renderer.render([u])

  # displacement of 0 isn't emitted
  def test_base_address(self):
    load = ins(X86Ops.MOV, dtypes.int32, (def_reg(dtypes.uint64, RDI), UOp(Ops.NOOP), machine_const(0), machine_const(4)), RDI)
    # mov edi, dword ptr [rdi]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 3F"))

  # rsp/r12 require a sib byte when used as base memory address
  def test_rsp_base_address(self):
    load = ins(X86Ops.MOV, dtypes.int32, (def_reg(dtypes.uint64, RSP), UOp(Ops.NOOP), machine_const(0), machine_const(4)), RSP)
    # mov esp, dword ptr [rsp]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 24 24"))

  # rbp/r13 require a displacement when used as base memory address
  def test_rbp_base_address(self):
    load = ins(X86Ops.MOV, dtypes.int32, (def_reg(dtypes.uint64, RBP), UOp(Ops.NOOP), machine_const(0), machine_const(4)), RBP)
    # mov ebp, dword ptr [rbp + 0]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 6D 00"))

  # test [base + index*scale]
  def test_base_index_address(self):
    load = ins(X86Ops.MOV, dtypes.int32, (def_reg(dtypes.uint64, RAX), def_reg(dtypes.int32, RDX), machine_const(0), machine_const(4)), RAX)
    # mov eax, dword ptr [rax + rdx*4]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 04 90"))

  # rsp as index means no index
  def test_rsp_index_address(self):
    load = ins(X86Ops.MOV, dtypes.int32, (def_reg(dtypes.uint64, RAX), def_reg(dtypes.int32, RSP), machine_const(0), machine_const(4)), RAX)
    # mov eax, dword ptr [rax]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 00"))

  # however r12 is a valid index
  def test_r12_index_address(self):
    load = ins(X86Ops.MOV, dtypes.int32,
               (def_reg(dtypes.uint64, RAX), def_reg(dtypes.int32, GPR[12]), machine_const(0), machine_const(4)), RAX)
    # mov eax, dword ptr [rax + r12*4]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("42 8B 04 A0"))

  # test [base + index*scale + 8bit disp]
  def test_complex_address_8bit_disp(self):
    load = ins(X86Ops.MOV, dtypes.int32, (def_reg(dtypes.uint64, RDI), def_reg(dtypes.int32, RSI), machine_const(10), machine_const(4)), RDI)
    # mov edi, dword ptr [rdi + rsi*4 + 0xa]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B 7C B7 0A"))

  # test [base + index*scale + 32bit disp]
  def test_complex_address_32bit_disp(self):
    load = ins(X86Ops.MOV, dtypes.int32,
               (def_reg(dtypes.uint64, RDI), def_reg(dtypes.int32, RSI), machine_const(10000), machine_const(4)), RDI)
    # mov edi, dword ptr [rdi + rsi*4 + 0x2710]
    self.assertEqual(bytes.fromhex(self.encode(load)), bytes.fromhex("8B BC B7 10 27 00 00"))

  # 8bit variants of legacy instructions subtract 1 from opcode
  def test_8bit_legacy_encoding(self):
    cast = ins(X86Ops.MOVSX, dtypes.int32, (def_reg(dtypes.int8, RDX),), RAX)
    # movsx eax, dl
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("0F BE C2"))

  # accessing lower 8 bits of rsp, rbp, rsi, rdi requires rex prefix
  def test_lower_8bits_reg(self):
    cast = ins(X86Ops.MOVSX, dtypes.int32, (def_reg(dtypes.int8, RDI),), RAX)
    # movsx eax, dil
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("40 0F BE C7"))

  # test 16 bit variant of legacy instruction
  def test_16bit_legacy_encoding(self):
    cast = ins(X86Ops.MOVSX, dtypes.int16, (def_reg(dtypes.int8, RDX),), RAX)
    # movsx ax, dl
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("66 0F BE C2"))

  # test 64 bit variant of legacy instruction
  def test_64bit_legacy_encoding(self):
    cast = ins(X86Ops.MOVSX, dtypes.int64, (def_reg(dtypes.int8, RDX),), RAX)
    # movsx rax, dl
    self.assertEqual(bytes.fromhex(self.encode(cast)), bytes.fromhex("48 0F BE C2"))

  # test compact vex encoding
  def test_compact_vex_encoding(self):
    xmm0, xmm1 = def_reg(dtypes.float32, XMM[0]), def_reg(dtypes.float32, XMM[1])
    add = ins(X86Ops.VADDSS, dtypes.float32, (xmm0, xmm1), XMM[0])
    # vaddss xmm0, xmm0, xmm1
    self.assertEqual(bytes.fromhex(self.encode(add)), bytes.fromhex("C5 FA 58 C1"))

  # test long vex encoding
  def test_long_vex_encoding(self):
    xmm0, xmm8 = def_reg(dtypes.float32, XMM[0]), def_reg(dtypes.float32, XMM[8])
    add = ins(X86Ops.VADDSS, dtypes.float32, (xmm0, xmm8), XMM[0])
    # vaddss xmm0, xmm0, xmm8
    self.assertEqual(bytes.fromhex(self.encode(add)), bytes.fromhex("C4 C1 7A 58 C0"))

  # test ymm encoding
  def test_ymm_encoding(self):
    xmm0, xmm1 = def_reg(dtypes._uint256, XMM[0]), def_reg(dtypes._uint256, XMM[1])
    add = ins(X86Ops.VADDPS, dtypes._uint256, (xmm0, xmm1), XMM[0])
    # vaddps ymm0, ymm0, ymm1
    self.assertEqual(bytes.fromhex(self.encode(add)), bytes.fromhex("C5 FC 58 C1"))

  # test encoding where register is in the immediate field
  def test_reg_in_imm_field(self):
    xmm0, xmm1, xmm2 = def_reg(dtypes.float32, XMM[0]), def_reg(dtypes.float32, XMM[1]), def_reg(dtypes.float32, XMM[2])
    blend = ins(X86Ops.VBLENDVPS, dtypes.float32, (xmm0, xmm1, xmm2), XMM[0])
    # vblendvps xmm0, xmm0, xmm1, xmm2
    self.assertEqual(bytes.fromhex(self.encode(blend)), bytes.fromhex("C4 E3 79 4A C1 20"))

  # when writting to mem the uop takes the store form where dtype is void and there's no definition
  def test_write_mem(self):
    address = (def_reg(dtypes.uint64, RDI), def_reg(dtypes.int32, RSI), machine_const(10), machine_const(4))
    xmm0 = def_reg(dtypes.float32, XMM[0])
    extr = ins(X86Ops.VPEXTRD, dtypes.void, address + (xmm0, machine_const(0)))
    # vpextrd dword ptr [rdi + rsi*4 + 0xa], xmm0, 0
    self.assertEqual(bytes.fromhex(self.encode(extr)), bytes.fromhex("C4 E3 79 16 44 B7 0A 00"))

  # test two address instruction with fused load works
  def test_two_address_load(self):
    address = (def_reg(dtypes.uint64, RDI), def_reg(dtypes.int32, RSI), machine_const(10), machine_const(4))
    cmove = ins(X86Ops.CMOVE, dtypes.int32, address, RAX)
    # cmove eax, dword ptr [rdi + rsi*4 + 0xa]
    self.assertEqual(bytes.fromhex(self.encode(cmove)), bytes.fromhex("0F 44 44 B7 0A"))

  # test instruction where displacement and imm have the same value
  def test_disp_imm_same_value(self):
    address = (def_reg(dtypes.uint64, RDI), def_reg(dtypes.int8, RSI), machine_const(10), machine_const(1))
    mov = ins(X86Ops.MOVi, dtypes.void, address + (machine_const(10),))
    # mov byte ptr [rdi + rsi + 0xa], 0xa
    self.assertEqual(bytes.fromhex(self.encode(mov)), bytes.fromhex("40 C6 44 37 0A 0A"))

    address = (def_reg(dtypes.uint64, RDI), def_reg(dtypes.int32, RSI), machine_const(10), machine_const(4))
    imul = ins(X86Ops.IMULi, dtypes.int32, address + (machine_const(10),), RDI)
    # imul edi, dword ptr [rdi + rsi*4 + 0xa], 0xa
    self.assertEqual(bytes.fromhex(self.encode(imul)), bytes.fromhex("69 7C B7 0A 0A 00 00 00"))

  # a displacement takes the smallest width that holds it wherever it was minted, so a stack argument offset is a disp8 like any other
  def test_disp_width(self):
    def load(d): return ins(X86Ops.MOV, dtypes.uint64, (def_reg(dtypes.uint64, RSP), UOp(Ops.NOOP), machine_const(d), machine_const(8)), RAX)
    # mov rax, qword ptr [rsp + 8] / [rsp + 127]
    self.assertEqual(bytes.fromhex(self.encode(load(8))), bytes.fromhex("48 8B 44 24 08"))
    self.assertEqual(bytes.fromhex(self.encode(load(127))), bytes.fromhex("48 8B 44 24 7F"))
    # the rule is on the magnitude, so 128 and -128 both take 4 bytes
    self.assertEqual(bytes.fromhex(self.encode(load(128))), bytes.fromhex("48 8B 84 24 80 00 00 00"))
    self.assertEqual(bytes.fromhex(self.encode(load(-128))), bytes.fromhex("48 8B 84 24 80 FF FF FF"))

  # a machine const is a number with no width, the field it lands in writes its low bytes
  def test_imm_low_bytes(self):
    # movabs rax, 0xfffffffffffffffe, the value is wider than the register
    movabs = ins(X86Ops.MOVABS, dtypes.uint64, (machine_const(2**65-2),), RAX)
    self.assertEqual(bytes.fromhex(self.encode(movabs)), bytes.fromhex("48 B8 FE FF FF FF FF FF FF FF"))
    for dt,v,b in ((dtypes.int8, 300, "2C"), (dtypes.int8, -200, "38"), (dtypes.uint8, 300, "2C")):
      # mov al, <low byte>
      self.assertEqual(bytes.fromhex(self.encode(ins(X86Ops.MOVi, dt, (machine_const(v),), RAX))), bytes.fromhex("40 C6 C0 " + b))
    # mov ax, 0x1170
    self.assertEqual(bytes.fromhex(self.encode(ins(X86Ops.MOVi, dtypes.int16, (machine_const(70000),), RAX))), bytes.fromhex("66 40 C7 C0 70 11"))

  # a constant in an immediate slot takes the width of the slot, not of the value, in either honest form
  def test_pair_imm_width(self):
    # add rax, 5 with an imm32, the operand being 64 bit doesn't widen the immediate
    for c in (machine_const(5), UOp.const(5).cast(dtypes.int64)):
      self.assertEqual(bytes.fromhex(self.encode(ins(X86Ops.ADDi, dtypes.int64, (c,), RAX))), bytes.fromhex("48 81 C0 05 00 00 00"))

  # cmoves have the cmp as the last src even though it is not explicitly used, the cmp doesn't define a reg and is ignored in the encoding
  def test_cmove_ignore_cmp(self):
    cmove = ins(X86Ops.CMOVE, dtypes.int32, (def_reg(dtypes.int32, RAX), UOp(Ops.INS, arg=X86Ops.CMP)), RDX)
    # cmove edx, eax
    self.assertEqual(bytes.fromhex(self.encode(cmove)), bytes.fromhex("0F 44 D0"))

if __name__ == "__main__":
  unittest.main()
