from tinygrad import Tensor
import numpy as np

N = 512
a = Tensor.rand(N, N).realize()
b = Tensor.rand(N, N).realize()

def f(a, b): return a.T @ b.T

na = a.numpy()
nb = b.numpy()
n = f(na, nb)
t = f(a, b)
np.testing.assert_allclose(t.numpy(), n)

# TEST=1 LLVM=1 DEBUG=2 python test.py