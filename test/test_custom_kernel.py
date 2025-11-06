import unittest
from tinygrad import Tensor, UOp, Context
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import KernelInfo, AxisType

# **** kernels ****

def custom_arange_kernel(C:UOp) -> UOp:
  i = UOp.range(C.size, 0)
  return C[i].store(i.cast(C.dtype.base)).end(i).sink(arg=KernelInfo(name=f"custom_arange_{C.size}"))

def custom_add_one_kernel(B:UOp, A:UOp) -> UOp:
  A,B = A.flatten(), B.flatten()
  assert B.size == A.size
  i = UOp.range(A.size, 0)
  return B[i].store(A[i] + 1).end(i).sink(arg=KernelInfo(name=f"add_one_{A.size}"))

def custom_elementwise_add_kernel(C:UOp, A:UOp, B:UOp) -> UOp:
  C,A,B = C.flatten(), A.flatten(), B.flatten()
  i = UOp.range(C.size, 0)
  return C[i].store(A[i]+B[i]).end(i).sink(arg=KernelInfo(name=f"custom_add_kernel_{C.size}")).simplify()

def custom_elementwise_addmul_kernel(C:UOp, D:UOp, A:UOp, B:UOp) -> UOp:
  C,D,A,B = C.flatten(), D.flatten(), A.flatten(), B.flatten()
  assert C.size == D.size
  i = UOp.range(C.size, 0)
  store_c = C[i].store(A[i]+B[i])
  store_d = D[i].store(A[i]*B[i])
  return UOp.group(store_c, store_d).end(i).sink(arg=KernelInfo(name=f"custom_addmul_kernel_{C.size}")).simplify()

def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  assert A.shape[1] == B.shape[0]
  i, j, k = UOp.range(C.shape[0], 0), UOp.range(C.shape[1], 1), UOp.range(A.shape[1], 2, axis_type=AxisType.REDUCE)
  C = C[i, j].set(0.0)
  C = C[i, j].set(C.after(k)[i, j] + A[i, k] * B[k, j], end=k)
  prog = C.end(i, j)
  return prog.sink(arg=KernelInfo(name=f"custom_gemm_{C.shape[0]}_{C.shape[1]}_{A.shape[1]}", opts_to_apply=()))

def custom_sum(B:UOp, A:UOp) -> UOp:
  i = UOp.range(A.shape[0], 0, axis_type=AxisType.REDUCE)
  B = B[0].set(0.0)
  B = B[0].set(B.after(i)[0] + A[i], end=i)
  return B.sink(arg=KernelInfo(name=f"custom_sum_{A.shape[0]}", opts_to_apply=()))

def flip_contract_kernel(dest:UOp, src:UOp):
  i = UOp.range(dest.shape[0], 0)
  j = UOp.range(dest.shape[1], 1, AxisType.UPCAST)
  vec = src[i, j].contract(j)
  store = UOp.group(*[dest[i, k].store(vec.gep(3-k)) for k in range(4)])
  return store.end(i).sink(arg=KernelInfo(name=f"flip_contract_{dest.size}", opts_to_apply=()))

def slice_sum_kernel(dest:UOp, src:UOp):
  G = UOp.range(src.shape[0], 0)
  slice_src = src[G, :]
  reg = UOp.placeholder((1,), dest.dtype.base, 0, addrspace=AddrSpace.REG)
  reg = reg.after(G)[0].set(0)
  R = UOp.range(src.shape[1], 1, AxisType.REDUCE)
  reg = reg[0].set(reg.after(R)[0] + slice_src[R], end=R)
  ast = dest[G].set(reg[0], end=G)
  return ast.sink(arg=KernelInfo(name=f"slice_sum_{src.shape[0]}_{src.shape[1]}", opts_to_apply=()))

def flash_attention_forward(O:UOp, Q:UOp, K:UOp, V:UOp) -> UOp:
  """
  Simple Flash Attention Forward Pass - Element by Element

  Q: [seq_len, head_dim]
  K: [seq_len, head_dim]
  V: [seq_len, head_dim]
  O: [seq_len, head_dim]
  """

  N, d = Q.shape[0], Q.shape[1]  # seq_len, head_dim
  scale = 1.0 / (d ** 0.5)

  # Outer loop over query positions
  i = UOp.range(N, 0)
  # Loop over output dimensions
  d_out = UOp.range(d, 1)

  # Create registers for running max, sum, and output value
  m_reg = UOp.placeholder((1,), Q.dtype.base, 0, addrspace=AddrSpace.REG)
  l_reg = UOp.placeholder((1,), Q.dtype.base, 1, addrspace=AddrSpace.REG)
  o_reg = UOp.placeholder((1,), Q.dtype.base, 2, addrspace=AddrSpace.REG)

  # Initialize registers for this query position and output dimension
  m_reg = m_reg.after(i, d_out)[0].set(-float('inf'))
  l_reg = l_reg.after(i, d_out)[0].set(0.0)
  o_reg = o_reg.after(i, d_out)[0].set(0.0)

  # Inner loop over key positions (reduction)
  j = UOp.range(N, 2, axis_type=AxisType.REDUCE)

  # Compute dot product Q[i] * K[j] over head_dim
  k_inner = UOp.range(d, 3, axis_type=AxisType.REDUCE)
  qk_reg = UOp.placeholder((1,), Q.dtype.base, 3, addrspace=AddrSpace.REG)
  qk_reg = qk_reg[0].set(0.0)
  qk_reg = qk_reg[0].set(qk_reg.after(k_inner)[0] + Q[i, k_inner] * K[j, k_inner] * scale, end=k_inner)
  qk_val = qk_reg[0]

  # Update running max
  m_old = m_reg.after(j)[0]
  m_new = m_old.maximum(qk_val)

  # Compute attention weight: exp(qk - m_new)
  attn_weight = (qk_val - m_new).exp()

  # Update running sum with correction factor
  correction = (m_old - m_new).exp()
  l_old = l_reg.after(j)[0]
  l_new = l_old * correction + attn_weight

  # Load V[j, d_out] and update output accumulator
  v_val = V[j, d_out]
  o_old = o_reg.after(j)[0]
  o_new = o_old * correction + v_val * attn_weight

  # Store updated values back to registers
  m_reg = m_reg[0].set(m_new, end=j)
  l_reg = l_reg[0].set(l_new, end=j)
  o_reg = o_reg[0].set(o_new, end=j)

  # Final normalization
  l_final = l_reg.after(j)[0]
  o_final = o_reg.after(j)[0] / l_final

  # Store to output
  store = O[i, d_out].store(o_final)

  return store.end(d_out, i).sink(arg=KernelInfo(name=f"flash_attention_{N}_{d}", opts_to_apply=()))


# **** backward callbacks ****

def backward_gemm(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src
  grad_a = (Tensor(gradient) @ Tensor(b).T).uop
  grad_b = (Tensor(a).T @ Tensor(gradient)).uop
  return (None, grad_a, grad_b)

def backward_gemm_custom(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src
  grad_a = Tensor.empty_like(Tensor(a)).custom_kernel(Tensor(gradient), Tensor(b).T, fxn=custom_gemm)[0].uop
  grad_b = Tensor.empty_like(Tensor(b)).custom_kernel(Tensor(a).T, Tensor(gradient), fxn=custom_gemm)[0].uop
  return (None, grad_a, grad_b)

# **** tests ****

class TestCustomKernel(unittest.TestCase):
  def test_simple(self):
    a = Tensor.ones(16, 16).contiguous()
    b = Tensor.ones(16, 16).contiguous()
    c = Tensor.empty(16, 16)

    c = Tensor.custom_kernel(c,a,b, fxn=custom_elementwise_add_kernel)[0]

    out = c.flatten().tolist()
    assert all(x == 2 for x in out), "all 2"

  def test_multioutput(self):
    a = Tensor.full((16, 16), 3.).contiguous()
    b = Tensor.full((16, 16), 3.).contiguous()
    c = Tensor.empty(16, 16)
    d = Tensor.empty(16, 16)

    c,d = Tensor.custom_kernel(c,d,a,b, fxn=custom_elementwise_addmul_kernel)[:2]
    Tensor.realize(c,d)

    assert all(x == 6 for x in c.flatten().tolist()), "all 6"
    assert all(x == 9 for x in d.flatten().tolist()), "all 9"

  def test_arange(self):
    ref = Tensor.arange(100)
    tst = Tensor.empty_like(ref)
    tst = tst.custom_kernel(fxn=custom_arange_kernel)[0]
    self.assertTrue((ref == tst).all().item())

  def test_flip_contract(self):
    a = Tensor.randn(10,4)
    b = Tensor.empty_like(a)
    b = b.custom_kernel(a, fxn=flip_contract_kernel)[0]
    self.assertTrue((a.flip(1) == b).all().item())

  def test_noncontig(self):
    a = Tensor.ones(16, 16).contiguous()
    tst = Tensor.empty_like(a)
    b = a+1
    b_p1 = Tensor.custom_kernel(tst, b, fxn=custom_add_one_kernel)[0]
    self.assertTrue((b_p1 == 3).all().item())

  def test_sum(self):
    # TODO: this only works for float, and silently fails with int
    a = Tensor([1.0, 2, 3, 4, 5])
    tst = Tensor.empty(1)
    b = Tensor.custom_kernel(tst, a, fxn=custom_sum)[0]
    self.assertEqual(b.item(), 15)

  def test_slice_sum(self):
    A = Tensor.randn(16, 16).contiguous()
    B = Tensor.empty(16)
    B = Tensor.custom_kernel(B, A, fxn=slice_sum_kernel)[0]
    self.assertTrue(B.allclose(A.sum(1)))

  def test_gemm(self):
    N = 16
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)
    c = Tensor.empty(N, N)

    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm)[0]
    err = (tst - (a@b)).square().max()
    self.assertLess(err.item(), 1e-6)

  def test_gemm_backward_custom(self): self.test_gemm_backward(True)
  # NOTE: grad_fxn doesn't work with pyrender
  @Context(SPEC=1)
  def test_gemm_backward(self, custom_backward_gemm=False):
    N = 4
    a_rand = Tensor.randn(N, 8)
    b_rand = Tensor.randn(8, N)
    Tensor.realize(a_rand, b_rand)

    a, b = Tensor(a_rand.numpy(), requires_grad=True), Tensor(b_rand.numpy(), requires_grad=True)
    c = Tensor.empty(N, N)
    tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm, grad_fxn=backward_gemm_custom if custom_backward_gemm else backward_gemm)[0]
    tst.sum().backward()
    grad_a, grad_b = a.grad, b.grad
    Tensor.realize(tst, grad_a, grad_b)

    a, b = Tensor(a_rand.numpy(), requires_grad=True), Tensor(b_rand.numpy(), requires_grad=True)
    ref = (a@b)
    ref.sum().backward()
    real_grad_a, real_grad_b = a.grad, b.grad
    Tensor.realize(ref, real_grad_a, real_grad_b)

    err = (tst - ref).square().max()
    self.assertLess(err.item(), 1e-6)

    err = (grad_a - real_grad_a).square().max()
    self.assertLess(err.item(), 1e-6)

    err = (grad_b - real_grad_b).square().max()
    self.assertLess(err.item(), 1e-6)

def test_flash_attention():
  """Test the flash attention kernel against standard attention"""

  N, d = 32, 16  # seq_len, head_dim

  # Create input tensors
  Q = Tensor.randn(N, d).contiguous()
  K = Tensor.randn(N, d).contiguous()
  V = Tensor.randn(N, d).contiguous()
  O = Tensor.empty(N, d)

  # Run flash attention
  O_flash = Tensor.custom_kernel(O, Q, K, V, fxn=lambda o,q,k,v: flash_attention_forward(o,q,k,v))[0]

  # Reference implementation
  scores = (Q @ K.T) / (d ** 0.5)
  attn = scores.softmax(axis=-1)
  O_ref = attn @ V

  # Compare
  Tensor.realize(O_flash, O_ref)
  err = (O_flash - O_ref).square().max()
  print(f"Max error: {err.item()}")
  print(f"Test {'PASSED' if err.item() < 1e-4 else 'FAILED'}")

if __name__ == '__main__':
  unittest.main()
