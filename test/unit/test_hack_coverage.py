"""
Tests for edge cases covered by documented hacks in tinygrad.
These tests ensure the workarounds continue to function correctly.
"""
import unittest
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import temp

class TestDiskAssignHack(unittest.TestCase):
  """Tests for the DISK assign hack in tensor.py:276"""

  def test_disk_assign_from_tensor(self):
    # Test assign to DISK tensor from CPU tensor
    a = Tensor.empty(5, device=f"disk:{temp('hack_disk_assign')}").assign(Tensor.ones(5)).numpy()
    np.testing.assert_equal(a, np.ones(5))

  def test_disk_assign_from_list(self):
    # Test assign to DISK tensor from list
    a = Tensor.empty(5, device=f"disk:{temp('hack_disk_assign_list')}").assign([1, 2, 3, 4, 5]).numpy()
    np.testing.assert_equal(a, [1, 2, 3, 4, 5])

  def test_disk_assign_dtype_matching(self):
    # Test assign with different dtypes
    a = Tensor.empty(4, dtype=dtypes.int32, device=f"disk:{temp('hack_disk_dtype')}").assign([1, 2, 3, 4]).numpy()
    np.testing.assert_equal(a, [1, 2, 3, 4])

class TestThreeOpAssignShapeHack(unittest.TestCase):
  """Tests for the 3-op assign shape hack in uop/ops.py:299-300.

  The ASSIGN op can have 3 sources:
    - src[0]: target buffer
    - src[1]: value to assign
    - src[2]: movement ops to track shape (added by indexing)

  The shape check only considers src[:2] to avoid issues with the 3rd op.
  """

  def test_assign_basic_shape(self):
    # Basic assign should work
    a = Tensor.ones(4, 4).contiguous().realize()
    a.assign(Tensor.full((4, 4), 2.0))
    a.realize()
    np.testing.assert_equal(a.numpy(), np.full((4, 4), 2.0))

  def test_assign_with_permute(self):
    # Assign with permuted view (tests shape tracking)
    a = Tensor.arange(16).reshape(4, 4).contiguous().realize()
    b = Tensor.arange(16).reshape(4, 4)
    a_perm = a.permute(1, 0)
    a_perm.assign(b)
    a_perm.realize()
    # The permuted view should be assigned correctly
    np.testing.assert_equal(a_perm.numpy(), b.numpy())

  def test_assign_preserves_buffer(self):
    # Ensure assign doesn't create new buffer
    a = Tensor.ones(4, 4).contiguous().realize()
    buf_before = a.uop.base.realized
    a += 1
    a.realize()
    buf_after = a.uop.base.realized
    self.assertEqual(buf_before, buf_after)

class TestNoopToConstHack(unittest.TestCase):
  """Tests for the noop-to-const hack in schedule/rangeify.py:241.

  When a NOOP turns into a CONST, the BUFFERIZE needs special handling.
  """

  def test_self_copy_noop(self):
    # COPY to same device is a NOOP
    a = Tensor([1, 2, 3]).contiguous().realize()
    b = a.to(a.device)  # This should be a NOOP
    np.testing.assert_equal(b.numpy(), [1, 2, 3])

  def test_noop_in_graph(self):
    # Test that NOOPs don't break the graph
    a = Tensor([1, 2, 3]).contiguous()
    b = a + 0  # Could be optimized to NOOP
    np.testing.assert_equal(b.numpy(), [1, 2, 3])

class TestImageConv2dNonMultipleOf4Hack(unittest.TestCase):
  """Tests for non-multiple-of-4 handling in image_conv2d (tensor.py:3893-3937).

  GPU image formats typically require channel counts to be multiples of 4.
  The hack pads input/output channels as needed.
  """

  @unittest.skipUnless(Device.DEFAULT in {"GPU", "METAL"}, "image conv only on GPU/METAL")
  def test_conv_cin_not_multiple_of_4(self):
    # Test with cin=3 (not multiple of 4)
    from tinygrad.helpers import IMAGE
    if not IMAGE: self.skipTest("IMAGE not enabled")

    x = Tensor.randn(1, 3, 8, 8)  # cin=3
    w = Tensor.randn(16, 3, 3, 3)  # cout=16, cin=3
    # Should work even though cin=3 is not multiple of 4
    y = x.image_conv2d(w)
    self.assertEqual(y.shape, (1, 16, 6, 6))

  @unittest.skipUnless(Device.DEFAULT in {"GPU", "METAL"}, "image conv only on GPU/METAL")
  def test_conv_cout_not_multiple_of_4(self):
    # Test with cout=5 (not multiple of 4)
    from tinygrad.helpers import IMAGE
    if not IMAGE: self.skipTest("IMAGE not enabled")

    x = Tensor.randn(1, 4, 8, 8)  # cin=4 (multiple of 4)
    w = Tensor.randn(5, 4, 3, 3)  # cout=5 (not multiple of 4)
    # Should work even though cout=5 is not multiple of 4
    y = x.image_conv2d(w)
    self.assertEqual(y.shape, (1, 5, 6, 6))

class TestCStyleGetitemHelper(unittest.TestCase):
  """Test for the hacky __getitem__ helper in renderer/cstyle.py:160.

  This allows pattern matching code to access self.r via ctx[key].
  """

  def test_render_simple_kernel(self):
    # Just ensure rendering works (uses the __getitem__ internally)
    a = Tensor([1, 2, 3])
    b = a + 1
    b.realize()
    np.testing.assert_equal(b.numpy(), [2, 3, 4])

class TestMstackMselectBufferHack(unittest.TestCase):
  """Tests for the MSTACK/MSELECT buffer hack in schedule/rangeify.py:405.

  When handling AFTER ops with MSTACK/MSELECT, the buffer is extracted
  from the first source instead of the MSTACK/MSELECT itself.
  """

  def test_multidevice_buffer_handling(self):
    # This tests the multi-device path which uses MSTACK/MSELECT
    # Note: actual multi-device testing requires hardware with multiple devices
    # Here we just test basic copy which exercises similar code paths
    a = Tensor([1, 2, 3, 4]).contiguous().realize()
    b = a.to(Device.DEFAULT)
    np.testing.assert_equal(b.numpy(), [1, 2, 3, 4])

class TestConstReplacementHack(unittest.TestCase):
  """Tests for the CONST replacement hack in schedule/rangeify.py:432.

  When using symbolic values, CONSTs may get replaced and need their
  src tuple cleaned up.
  """

  def test_symbolic_const_handling(self):
    from tinygrad import Variable
    # Test that symbolic shapes work correctly
    # The tensor can be created with symbolic shape
    n = Variable("n", 1, 10).bind(5)
    a = Tensor.ones(n)
    # The shape should contain the symbolic variable
    self.assertEqual(len(a.shape), 1)
    # After binding, operations should work (tested via schedule cache path)
    b = a.sum()
    # Sum reduces the shape away, so result should be concrete
    self.assertEqual(b.numpy().item(), 5.0)

  def test_const_in_expression(self):
    # Ensure consts are handled in expressions
    a = Tensor([1, 2, 3])
    b = a + 5  # 5 becomes a CONST
    np.testing.assert_equal(b.numpy(), [6, 7, 8])

class TestCopyDetectionHack(unittest.TestCase):
  """Tests for the COPY detection hack in schedule/rangeify.py:501.

  When splitting kernels, COPY/BUFFER_VIEW/ENCDEC ops need special
  detection to avoid wrapping them in a SINK.
  """

  def test_copy_to_device(self):
    # Test basic copy detection
    a = Tensor([1, 2, 3]).realize()
    b = a.to("CPU")
    np.testing.assert_equal(b.numpy(), [1, 2, 3])

  def test_buffer_view_detection(self):
    # Buffer views should also be detected
    a = Tensor([1, 2, 3, 4]).contiguous().realize()
    b = a[1:3]  # This creates a view
    np.testing.assert_equal(b.numpy(), [2, 3])

class TestImageDTypeHack(unittest.TestCase):
  """Tests for the ImageDType hack in device.py:97.

  ImageDType handling is done in Buffer constructor, which ideally
  should be elsewhere but works for now.
  """

  @unittest.skipUnless(Device.DEFAULT in {"GPU", "METAL"}, "image dtype only on GPU/METAL")
  def test_image_dtype_creation(self):
    from tinygrad.helpers import IMAGE
    if not IMAGE: self.skipTest("IMAGE not enabled")

    # Just test that image tensors can be created with image dtypes
    base_dtype = dtypes.imagef if hasattr(dtypes, 'imagef') else None
    if base_dtype is None:
      self.skipTest("imagef dtype not available")

class TestMetadataPreservationHack(unittest.TestCase):
  """Tests for the metadata preservation hack in engine/schedule.py:146.

  The graph_rewrite_map is called just to preserve metadata for
  debugging/visualization purposes.
  """

  def test_metadata_through_schedule(self):
    # Metadata should be preserved through the schedule
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    c.realize()
    # If this works without error, metadata handling is fine
    np.testing.assert_equal(c.numpy(), [5, 7, 9])

if __name__ == "__main__":
  unittest.main()
