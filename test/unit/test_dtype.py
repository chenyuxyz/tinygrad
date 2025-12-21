import unittest
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes, DType, ImageDType, PtrDType, to_dtype, can_safe_cast

class TestImageDType(unittest.TestCase):
  def test_image_scalar(self):
    assert dtypes.imagef((10,10)).base.scalar() == dtypes.float32
    assert dtypes.imageh((10,10)).base.scalar() == dtypes.float32
  def test_image_vec(self):
    assert dtypes.imagef((10,10)).base.vec(4) == dtypes.float32.vec(4)
    assert dtypes.imageh((10,10)).base.vec(4) == dtypes.float32.vec(4)

class TestEqStrDType(unittest.TestCase):
  def test_image_ne(self):
    if ImageDType is None: raise unittest.SkipTest("no ImageDType support")
    assert dtypes.float == dtypes.float32, "float doesn't match?"
    assert dtypes.imagef((1,2,4)) != dtypes.imageh((1,2,4)), "different image dtype doesn't match"
    assert dtypes.imageh((1,2,4)) != dtypes.imageh((1,4,2)), "different shape doesn't match"
    assert dtypes.imageh((1,2,4)) == dtypes.imageh((1,2,4)), "same shape matches"
    assert isinstance(dtypes.imageh((1,2,4)), ImageDType)
  def test_ptr_eq(self):
    assert dtypes.float32.ptr() == dtypes.float32.ptr()
    assert not (dtypes.float32.ptr() != dtypes.float32.ptr())
  def test_ptr_nbytes(self):
    assert dtypes.float16.ptr(32).nbytes() == 32 * dtypes.float16.itemsize
  def test_ptr_nbytes_unlimited(self):
    self.assertRaises(RuntimeError, lambda: dtypes.float32.ptr().nbytes())
  def test_strs(self):
    if PtrDType is None: raise unittest.SkipTest("no PtrDType support")
    self.assertEqual(str(dtypes.imagef((1,2,4))), "dtypes.imagef((1, 2, 4))")
    self.assertEqual(str(dtypes.float32.ptr(16)), "dtypes.float.ptr(16)")

class TestToDtype(unittest.TestCase):
  def test_dtype_to_dtype(self):
    dtype = dtypes.int32
    res = to_dtype(dtype)
    self.assertIsInstance(res, DType)
    self.assertEqual(res, dtypes.int32)

  def test_str_to_dtype(self):
    dtype = "int32"
    res = to_dtype(dtype)
    self.assertIsInstance(res, DType)
    self.assertEqual(res, dtypes.int32)

class TestCastConvenienceMethod(unittest.TestCase):
  def test_method(self):
    for input_dtype in (dtypes.float, dtypes.int):
      t = Tensor([1, 2], dtype=input_dtype)
      self.assertEqual(t.dtype, input_dtype)
      self.assertEqual(t.bool().dtype, dtypes.bool)
      self.assertEqual(t.short().dtype, dtypes.short)
      self.assertEqual(t.int().dtype, dtypes.int)
      self.assertEqual(t.long().dtype, dtypes.long)
      self.assertEqual(t.half().dtype, dtypes.half)
      self.assertEqual(t.bfloat16().dtype, dtypes.bfloat16)
      self.assertEqual(t.float().dtype, dtypes.float)
      self.assertEqual(t.double().dtype, dtypes.double)

class TestDtypeTolist(unittest.TestCase):
  def test_bfloat16(self):
    self.assertEqual(Tensor([-60000, 1.5, 3.1, 60000], device="PYTHON", dtype=dtypes.bfloat16).tolist(), [-59904.0, 1.5, 3.09375, 59904.0])
  def test_fp8(self):
    # 448
    self.assertEqual(Tensor([-30000, 1.5, 3.1, 30000], device="PYTHON", dtype=dtypes.fp8e4m3).tolist(), [-448.0, 1.5, 3.0, 448.0])
    # 57344
    self.assertEqual(Tensor([-30000, 1.5, 3.1, 30000], device="PYTHON", dtype=dtypes.fp8e5m2).tolist(), [-28672.0, 1.5, 3.0, 28672.0])

class TestCanSafeCastMatchesNumpy(unittest.TestCase):
  def test_can_safe_cast_matches_numpy(self):
    import numpy as np
    # Map tinygrad dtypes to numpy dtypes (only those with direct equivalents)
    dtype_map = {
      dtypes.bool: np.bool_, dtypes.int8: np.int8, dtypes.int16: np.int16, dtypes.int32: np.int32, dtypes.int64: np.int64,
      dtypes.uint8: np.uint8, dtypes.uint16: np.uint16, dtypes.uint32: np.uint32, dtypes.uint64: np.uint64,
      dtypes.float16: np.float16, dtypes.float32: np.float32, dtypes.float64: np.float64,
    }
    for dt0, np0 in dtype_map.items():
      for dt1, np1 in dtype_map.items():
        self.assertEqual(can_safe_cast(dt0, dt1), np.can_cast(np0, np1, casting='safe'), f"{dt0} -> {dt1}")

if __name__ == "__main__":
  unittest.main()
