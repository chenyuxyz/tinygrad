# ruff: noqa: E501, A001, A002, A006
# A001 Variable `input` is shadowing a Python builtin
# A002 Function argument `input` is shadowing a Python builtin
# A006 Lambda argument `input` is shadowing a Python builtin
from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import Ops
from tinygrad.helpers import getenv, prod, strides_for_shape, argfix
import torch.lib
TORCH_DEBUG = getenv("TORCH_DEBUG")
import torch, pathlib, operator, functools, weakref
torch.autograd.grad_mode.set_multithreading_enabled(False)
from tinygrad.dtype import _from_torch_dtype, _to_torch_dtype

# https://pytorch.org/docs/stable/torch.compiler_ir.html

def _from_torch_device(device: torch.device): return f"{Device.DEFAULT}:{device.index or 0}"
def _to_torch_device(device: str): return torch.device("tiny", int(device.partition(":")[2] or 0))

import torch.utils.cpp_extension
mod = torch.utils.cpp_extension.load(name="custom_device_extension", sources=[str(pathlib.Path(__file__).parent / "wrapped_tensor.cpp")])
def calculate_storage_offset(x: Tensor) -> int:
  offset = 0
  for u in x.uop.toposort():
    if u.op == Ops.SHRINK:
      u_strides = strides_for_shape(u.src[0].shape)
      for i, (start, _) in enumerate(u.marg): offset += start * u_strides[i]
  return offset
def wrap(x: Tensor) -> torch.Tensor:
  x._strides = strides_for_shape(x.shape) # always recalculate
  if (not hasattr(x, '_storage_offset')) or (not x.uop.is_realized): x._storage_offset = calculate_storage_offset(x)
  return mod.wrap(x, _to_torch_dtype(x.dtype), _to_torch_device(x.device).index)
def unwrap(x:torch.Tensor) -> Tensor:
  assert isinstance(x, torch.Tensor), f"x isn't {type(x)}"
  return mod.unwrap(x)
class TinyBackend:
  def is_initialized(self): return True
  def is_available(self): return True
  def current_device(self): return 0
  def _is_in_bad_fork(self): return False
  def manual_seed_all(self, seed: int): Tensor.manual_seed(seed)
  def device_count(self): return getenv("GPUS", 1) # TODO: device count in tiny?
torch.utils.rename_privateuse1_backend("tiny")
torch._register_device_module("tiny", TinyBackend())
torch.utils.generate_methods_for_privateuse1_backend()
aten = torch.ops.aten

# track view relationships for in place operations
def canonical_base(view: Tensor): return getattr(view, "_view_base", view)
def derived_views(base: Tensor): return [t for tref in getattr(base, "_views", set()) if (t:=tref()) is not None]
def unwrap_args(args, kwargs):
  return [unwrap(x) if isinstance(x, torch.Tensor) else x for x in args], {k:unwrap(v) if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()}
def wrap_view_op(fn):
  @functools.wraps(fn)
  def _wrap(*args, **kwargs):
    args, kwargs = unwrap_args(args, kwargs)
    ret = fn(*args, **kwargs)
    base = canonical_base(args[0])
    ret._view_base = base
    base._views = getattr(base, "_views", set())
    base._views.add(weakref.ref(ret))
    ret._view_ops = _get_view_ops(args[0]) + [(fn, args[1:], kwargs)]
    return wrap(ret)
  return _wrap

view_ops = {
  "aten.view": Tensor.reshape,
  "aten._unsafe_view": Tensor.reshape,  # when are views unsafe, and do we care?
  "aten.view.dtype": lambda self,dtype: self.bitcast(_from_torch_dtype(dtype)),
  "aten.expand": Tensor.expand,
  "aten.t": Tensor.transpose,
  "aten.transpose.int": Tensor.transpose,
  "aten.squeeze.dim": Tensor.squeeze,
  "aten.unsqueeze": Tensor.unsqueeze,
  "aten.select.int": lambda self, dim, idx: self[(slice(None),) * (dim%self.ndim) + (idx,)],
  "aten.permute": Tensor.permute,
  "aten.alias": lambda self: self,
  "aten.diagonal": Tensor.diagonal,
  }

# torch 2.10 handles this natively
if tuple(map(int, torch.__version__.split('.')[:2])) < (2, 10): view_ops.update({"aten.detach": Tensor.detach})

for k,v in view_ops.items(): torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(wrap_view_op(v))

def _get_view_ops(view): return getattr(view, "_view_ops", [])

def _apply_view_ops(target, ops):
  for fn, args, kwargs in ops: target = fn(target, *args, **kwargs)
  return target

# similar to https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/InferSize.h
def _reshape_target_shape(shape:tuple[int, ...], args) -> tuple[int, ...]|None:
  if not (req := argfix(*args)): return None
  new_shape, infer_idx = [], -1
  for i, s in enumerate(req):
    if s is None: s = shape[i] if i < len(shape) else None
    if not isinstance(s, int): return None
    if s == -1:
      if infer_idx != -1: return None
      infer_idx = len(new_shape)
    new_shape.append(s)
  total = prod(shape)
  if infer_idx != -1:
    known = prod(x for x in new_shape if x != -1)
    if known == 0:
      if total != 0: return None
      new_shape[infer_idx] = 0
    else: new_shape[infer_idx] = total // known
  return tuple(new_shape) if prod(new_shape) == total else None

# TODO: can we get rid of this? only for test_flatten_reshape_add
def _try_simple_reshape_view_write(base: Tensor, view: Tensor, val: Tensor) -> bool:
  if not (ops := _get_view_ops(view)): return False
  shapes = [base.shape]
  for fn, args, _ in ops:
    if fn is Tensor.reshape:
      if not (next_shape := _reshape_target_shape(shapes[-1], args)): return False
      shapes.append(next_shape)
  if shapes[-1] != view.shape: return False
  for s in reversed(shapes[:-1]): val = val.reshape(s)
  base.assign(val)
  return True

def _view_write(base: Tensor, view: Tensor, value: Tensor) -> None:
  val = value if value.dtype == base.dtype else value.cast(base.dtype)
  if view.shape == base.shape: return base.assign(val)
  if _try_simple_reshape_view_write(base, view, val): return
  idx_base = Tensor.arange(base.numel(), device=base.device, dtype=dtypes.int32).reshape(base.shape)
  idx_view = _apply_view_ops(idx_base, _get_view_ops(view)).reshape(-1)
  flat_base = base.reshape(base.numel()).contiguous()
  flat_base[idx_view] = val.reshape(-1)
  base.assign(flat_base.reshape(base.shape))

def _apply_inplace(target: Tensor, value: Tensor) -> None:
  val = value if value.dtype == target.dtype else value.cast(target.dtype)
  base = canonical_base(target)
  views = derived_views(base)
  if not views: return target.assign(val)
  view_ops_map = {v: _get_view_ops(v) for v in views}
  if target is base or target.uop is base.uop: base.assign(val)
  else: _view_write(base, target, val)
  for v in views: v.replace(_apply_view_ops(base, view_ops_map[v]))

# *** bad functions on CPU ***

@torch.library.impl("aten::_index_put_impl_", "privateuseone")
def _index_put_impl_(self, indices, values, accumulate=False, unsafe=False):
  # TODO: move to tinygrad
  ret = aten._index_put_impl_(self.cpu(), [x.cpu() if isinstance(x, torch.Tensor) else None for x in indices], values.cpu(), accumulate, unsafe).to(self.device)
  unwrap(self).assign(unwrap(ret))
  return self

@torch.library.impl("aten::index_put", "privateuseone")
def index_put(self, indices, values, accumulate=False):
  return aten.index_put(self.cpu(), [z.cpu() if isinstance(z, torch.Tensor) else None for z in indices], values.clone().cpu(), accumulate).tiny()

@torch.library.impl("aten::isin.Tensor_Tensor_out", "privateuseone")
def isin_tensor_tensor_out(x, y, *, assume_unique=False, invert=False, out=None):
  result = (unwrap(x).unsqueeze(-1) == unwrap(y).flatten()).any(-1)
  return out.copy_(wrap(~result if invert else result))

@torch.library.impl("aten::randperm.generator_out", "privateuseone")
def randperm_generator(n, generator=None, out=None):
  return out.copy_(wrap(Tensor.randperm(n, generator=generator, device=unwrap(out).device)))

@torch.library.impl("aten::cummax", "privateuseone")
def cummax(self, dim):
  values, indices = unwrap(self).cummax(dim)
  return (wrap(values), wrap(indices.cast(dtypes.int64)))

@torch.library.impl("aten::cummin", "privateuseone")
def cummin(self, dim):
  values, indices = unwrap(self).cummin(dim)
  return (wrap(values), wrap(indices.cast(dtypes.int64)))

@torch.library.impl("aten::nonzero", "privateuseone")
def nonzero(self): return wrap(unwrap(self).nonzero())

@torch.library.impl("aten::_linalg_eigh", "privateuseone")
# TODO: move to tinygrad
def _linalg_eigh(self, UPLO: str = 'U'):
  w, v = torch.linalg.eigh(self.cpu(), UPLO=UPLO)
  return w.tiny(), v.tiny()

@torch.library.impl("aten::_linalg_det", "privateuseone")
# TODO: move to tinygrad
def _linalg_det(self: torch.Tensor):
  result = aten._linalg_det(self.cpu())
  return result[0].tiny(), result[1].tiny(), result[2].tiny()

def upsample_backward(grad_out, output_size, input_size, *args, f=None): return f(grad_out.cpu(), output_size, input_size, *args).tiny()

for i in [
  "upsample_linear1d_backward", "upsample_nearest1d_backward", "_upsample_nearest_exact1d_backward",
  "upsample_nearest2d_backward", "_upsample_nearest_exact2d_backward",
  "upsample_nearest3d_backward", "_upsample_nearest_exact3d_backward",
  "upsample_trilinear3d_backward", "upsample_bilinear2d_backward"
]:
  torch.library.impl(f"aten::{i}", "privateuseone")(functools.partial(upsample_backward, f=getattr(aten, i)))

# *** end bad functions on CPU ***

@torch.library.impl("aten::index.Tensor", "privateuseone")
def index_tensor(x, y):
  return wrap(unwrap(x)[[unwrap(_y.to(x.device)) if _y is not None else slice(None) for _y in y]])


@torch.library.impl("aten::_local_scalar_dense", "privateuseone")
def _local_scalar_dense(tensor): return unwrap(tensor).item()

@wrap_view_op
def _as_strided(tensor:Tensor, size, stride, storage_offset=0):
  base = getattr(tensor, "_as_strided_base", canonical_base(tensor)).flatten()
  if prod(size) == 1: return base[storage_offset].reshape(size)
  indices = Tensor.zeros(size, dtype=dtypes.int32, device=base.device) + storage_offset
  for dim, (sz, st) in enumerate(zip(size, stride)):
    if st != 0:
      dim_range = Tensor.arange(sz, device=base.device, dtype=dtypes.int32) * st
      shape_for_broadcast = [1] * dim + [sz] + [1] * (len(size) - dim - 1)
      indices = indices + dim_range.reshape(shape_for_broadcast)
  result = base[indices.flatten()].reshape(size)
  result._as_strided_base = base
  return result

@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(tensor:torch.Tensor, size, stride, storage_offset=None):
  storage_offset = storage_offset or tensor.storage_offset()
  return _as_strided(tensor, size, stride, storage_offset)

@torch.library.impl("aten::_reshape_alias", "privateuseone")
def _reshape_alias(tensor:torch.Tensor, size, stride):
  return _as_strided(tensor, size, stride)

@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(size, stride, dtype, layout=None, device=None, pin_memory=False):
  if TORCH_DEBUG: print(f"empty_strided {size=} {stride=} {dtype=} {layout=} {device=} {pin_memory=}")
  ret = Tensor.empty(*size, dtype=_from_torch_dtype(dtype), device=_from_torch_device(device)).contiguous()
  # TODO: should return with requested strides
  return wrap(ret)

@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(size, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
  if TORCH_DEBUG: print(f"empty.memory_format {size=} {dtype=} {layout=} {device=} {pin_memory=} {memory_format=}")
  ret = Tensor.empty(*size, dtype=_from_torch_dtype(dtype or torch.get_default_dtype()), device=_from_torch_device(device)).contiguous()
  return wrap(ret)

@torch.library.impl("aten::max_pool2d_with_indices", "privateuseone")
def max_pool2d_with_indices(self:torch.Tensor, kernel_size:tuple[int, ...], stride=None, padding=0, dilation=1, ceil_mode=False):
  # TODO: supprt stride [] in tinygrad?
  if stride is not None and len(stride) == 0: stride = None
  ret, idx = unwrap(self).max_pool2d(kernel_size, stride, dilation, padding, ceil_mode, return_indices=True)
  return (wrap(ret), wrap(idx.cast(dtypes.int64)))

@torch.library.impl("aten::max_pool2d_with_indices_backward", "privateuseone")
def max_pool2d_with_indices_backward(grad_out:torch.Tensor, self:torch.Tensor, kernel_size:tuple[int, ...], stride=None, padding=0, dilation=1, ceil_mode=False, indices=None):
  return wrap(Tensor.max_unpool2d(unwrap(grad_out), unwrap(indices), output_size=unwrap(self).shape))

@torch.library.impl("aten::max_unpool2d", "privateuseone")
def max_unpool2d(self:torch.Tensor, indices:torch.Tensor, output_size):
  return wrap(unwrap(self).max_unpool2d(unwrap(indices), output_size=output_size))

@torch.library.impl("aten::arange", "privateuseone")
def arange(end, dtype=None, device=None, pin_memory=None):
  has_float = isinstance(end, float)
  return wrap(Tensor.arange(0, end, dtype=_from_torch_dtype(dtype or (torch.get_default_dtype() if has_float else torch.int64))))

@torch.library.impl("aten::arange.start", "privateuseone")
def arange_start(start, end, dtype=None, device=None, pin_memory=None):
  has_float = any(isinstance(x, float) for x in (start, end))
  return wrap(Tensor.arange(start, end, dtype=_from_torch_dtype(dtype or (torch.get_default_dtype() if has_float else torch.int64))))

@torch.library.impl("aten::arange.start_step", "privateuseone")
def arange_start_step(start, end, step, dtype=None, device=None, pin_memory=None):
  has_float = any(isinstance(x, float) for x in (start, end, step))
  return wrap(Tensor.arange(start, end, step, dtype=_from_torch_dtype(dtype or (torch.get_default_dtype() if has_float else torch.int64))))

@torch.library.impl("aten::convolution_overrideable", "privateuseone")
def convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
  if TORCH_DEBUG >= 1:
    print(f"convolution {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  input, weight, bias = unwrap(input), unwrap(weight), unwrap(bias) if bias is not None else None
  if not transposed: return wrap(input.conv2d(weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding))
  return wrap(input.conv_transpose2d(weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding, output_padding=output_padding))

@torch.library.impl("aten::convolution_backward_overrideable", "privateuseone")
def convolution_backward_overrideable(grad_out, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask):
  if TORCH_DEBUG >= 1:
    print(f"convolution_backward {input.shape=} {weight.shape=} {stride=} {padding=} {dilation=} {transposed=} {output_padding=} {groups=}")
  grad_out, input, weight, bias = unwrap(grad_out).detach(), unwrap(input).detach(), unwrap(weight).detach(), Tensor.zeros(weight.shape[0], device=_from_torch_device(weight.device))
  if not transposed: out = Tensor.conv2d(input, weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding)
  else:
    bias = Tensor.zeros(weight.shape[1] * groups)
    out = Tensor.conv_transpose2d(input, weight, bias, groups=groups, stride=stride, dilation=dilation, padding=padding, output_padding=output_padding)
  grads = out.gradient(*[t for t,m in zip([input, weight, bias], output_mask) if m], gradient=grad_out)
  return tuple([wrap(grads.pop(0)) if m else None for m in output_mask])

@torch.library.impl("aten::slice.Tensor", "privateuseone")
@wrap_view_op
def slice_tensor(self, dim=0, start=None, end=None, step=1):
  slices = [slice(None)] * self.ndim
  slices[dim] = slice(start, end, step)
  return self[slices]

@torch.library.impl("aten::slice_backward", "privateuseone")
def slice_backward(grad_out, input_sizes, dim, start, end, step):
  grad_input = Tensor.zeros(input_sizes).contiguous()
  slices = [slice(None)] * len(input_sizes)
  slices[dim] = slice(start, end, step)
  grad_input[slices] = unwrap(grad_out)
  return wrap(grad_input)

@torch.library.impl("aten::select_backward", "privateuseone")
def select_backward(grad_out, input_sizes, dim, index):
  grad_input = Tensor.zeros(input_sizes).contiguous()
  slices = [slice(None)] * len(input_sizes)
  slices[dim] = index
  grad_input[slices] = unwrap(grad_out)
  return wrap(grad_input)

def avg_pool(self, kernel_size, stride=[], padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
  return wrap(unwrap(self).avg_pool2d(kernel_size, stride if stride != [] else None, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad))

def avg_pool_backward(grad_out, self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
  self, grad_out = unwrap(self), unwrap(grad_out)
  out = Tensor.avg_pool2d(self, kernel_size, stride if stride != [] else None, dilation=1, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)
  return wrap(out.gradient(self, gradient=grad_out)[0])

for dim in [2, 3]:
  torch.library.impl(f"aten::avg_pool{dim}d", "privateuseone")(avg_pool)
  torch.library.impl(f"aten::avg_pool{dim}d_backward", "privateuseone")(avg_pool_backward)

def pad_forward(self, padding, mode=None): return wrap(Tensor.pad(unwrap(self), padding, mode=mode))

def pad_backward(grad_out, self, padding, mode):
  self, grad_out = unwrap(self), unwrap(grad_out)
  out = Tensor.pad(self, padding, mode=mode)
  return wrap(out.gradient(self, gradient=grad_out)[0])

for dim in [1, 2, 3]:
  for pad_type, mode in [("replication", "replicate"), ("reflection", "reflect")]:
    torch.library.impl(f"aten::{pad_type}_pad{dim}d", "privateuseone")(functools.partial(pad_forward, mode=mode))
    torch.library.impl(f"aten::{pad_type}_pad{dim}d_backward", "privateuseone")(functools.partial(pad_backward, mode=mode))

def upsample(self, size, align_corners=False, mode=None): return wrap(Tensor.interpolate(unwrap(self), size, mode=mode, align_corners=align_corners))
for i,pre in enumerate(["", "bi", "tri"]):
  torch.library.impl(f"aten::upsample_{pre}linear{i+1}d", "privateuseone")(functools.partial(upsample, mode="linear"))
  torch.library.impl(f"aten::upsample_nearest{i+1}d", "privateuseone")(functools.partial(upsample, mode="nearest"))
  torch.library.impl(f"aten::_upsample_nearest_exact{i+1}d", "privateuseone")(functools.partial(upsample, mode="nearest-exact"))

@torch.library.impl("aten::scatter_add.out", "privateuseone")
def scatter_add(self, dim, index, src, out):
  self, index, src, out_unwrapped = unwrap(self), unwrap(index), unwrap(src), unwrap(out)
  if self.shape == (): _apply_inplace(out_unwrapped, src)
  else: _apply_inplace(out_unwrapped, Tensor.scatter_reduce(self, dim, index, src, reduce='sum'))
  return out

def _copy_between_devices(src, dest, cast_dtype, to_device, non_blocking=False):
  if src.is_tiny and dest.is_tiny:
    src_t, dest_t = unwrap(src), unwrap(dest)
    if dest_t.uop.is_contiguous() or dest_t.uop.is_realized: src_t = src_t.contiguous()
    _apply_inplace(dest_t, src_t.cast(cast_dtype).to(to_device))
  elif src.is_tiny and dest.is_cpu:
    dest.resize_(src.numel()).resize_(src.shape)
    dest.copy_(torch.from_numpy(unwrap(src).cast(cast_dtype).numpy()))
  elif src.is_cpu and dest.is_tiny:
    unwrap(dest).assign(Tensor(src.numpy()).cast(cast_dtype).to(to_device))
  else:
    raise NotImplementedError(f"can't copy from {src.device} -> {dest.device}")

@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(src: torch.Tensor, dest, non_blocking=False):
  cast_dtype = _from_torch_dtype(dest.dtype)
  to_device = _from_torch_device(dest.device)
  _copy_between_devices(src, dest, cast_dtype, to_device, non_blocking)
  return dest

@torch.library.impl("aten::copy_", "privateuseone")
def copy_(self, src, non_blocking=False):
  cast_dtype = _from_torch_dtype(self.dtype)
  to_device = _from_torch_device(self.device)
  _copy_between_devices(src, self, cast_dtype, to_device, non_blocking)
  return self

@torch.library.impl("aten::topk.values", "privateuseone")
def topk_values(input, k, dim=None, largest=True, sorted=True, values=None, indices=None):
  out_values, out_indices = unwrap(input).topk(k, dim if dim is not None else -1, largest, sorted)
  _apply_inplace(unwrap(values), out_values)
  _apply_inplace(unwrap(indices), out_indices.cast(dtypes.int64))
  return values, indices

@torch.library.impl("aten::sort.values_stable", "privateuseone")
def sort_values(input, dim=-1, descending=False, stable=True, values=None, indices=None):
  out_values, out_indices = unwrap(input).sort(dim, descending)
  _apply_inplace(unwrap(values), out_values)
  _apply_inplace(unwrap(indices), out_indices.cast(dtypes.int64))
  return values, indices

@torch.library.impl("aten::_linalg_svd", "privateuseone")
def _linalg_svd(self, full_matrices=False):
  U, S, Vh = unwrap(self).svd(full_matrices)
  return wrap(U), wrap(S), wrap(Vh)

# register some decompositions
from torch._decomp import get_decompositions
decomps = [
  aten.native_layer_norm_backward,
  aten.linalg_cross,
  aten.addmm,
  aten.addcmul,
  aten.addcdiv,
  aten._log_softmax_backward_data,
  aten.threshold_backward,
  aten.softplus_backward,
  aten.elu,  # elu has a scale + input_scale param
  aten.elu_backward,
  aten.softplus,
  aten.logaddexp,
  aten.threshold,
  aten.nll_loss_forward,
  aten.nll_loss_backward,
  aten.nll_loss2d_backward,
  # AttributeError: 'int' object has no attribute '_broadcasted'
  aten.sigmoid_backward,
  aten.tanh_backward,
  aten.sinc,
  aten._prelu_kernel,
  aten.softshrink,
  aten.hardshrink,
  aten.log_sigmoid_forward,
  aten.log_sigmoid_backward,
  aten.isneginf,
  aten.isposinf,
  aten.nan_to_num,
  aten.logit,
  aten.rsub,
  aten.index_select,
  aten.native_dropout, aten.native_dropout_backward,
  aten._softmax_backward_data, aten.embedding_dense_backward,
  aten.linalg_vector_norm,
  aten.binary_cross_entropy, aten.binary_cross_entropy_backward,
  aten.upsample_nearest2d.out,
  # activations
  aten.hardswish, aten.hardswish_backward,
  aten.hardtanh, aten.hardtanh_backward,
  aten.gelu, aten.gelu_backward,
  aten.logical_and,
  aten.randint,
  aten.eye,
  aten.hardsigmoid_backward,
  aten.leaky_relu_backward,
  aten.nll_loss2d_forward,
  aten.unfold_backward,
  aten.norm,
  # prims-based decompositions (prims now implemented)
  aten.abs,
  aten.neg,
  aten.reciprocal,
  aten.sqrt,
  aten.rsqrt,
  aten.exp,
  aten.exp2,
  aten.expm1,
  aten.log,
  aten.log2,
  aten.log10,
  aten.log1p,
  aten.sin,
  aten.cos,
  aten.tan,
  aten.sinh,
  aten.cosh,
  aten.tanh,
  aten.asin,
  aten.acos,
  aten.atan,
  aten.asinh,
  aten.acosh,
  aten.atanh,
  aten.ceil,
  aten.floor,
  #aten.round,  # conflicts with round.decimals_out special handling
  aten.trunc,
  aten.sign,
  aten.erf,
  aten.bitwise_not,
  aten.add,
  aten.sub,
  aten.mul,
  aten.div,
  aten.maximum,
  aten.minimum,
  aten.pow,
  aten.remainder,
  aten.fmod,
  aten.bitwise_and,
  aten.bitwise_or,
  aten.bitwise_xor,
  aten.eq,
  aten.ne,
  aten.lt,
  aten.le,
  aten.gt,
  aten.ge,
  aten.prod,
  aten.sum,
  aten.amax,
  aten.amin,
  aten.var,
  aten.var_mean,
  aten.where,
  aten.cat,
  aten.flip,
  aten.fmax,
  aten.fmin,
  # NOTE: no decomposition exists for these
  #aten.max_pool2d_with_indices,
  #aten.digamma,   # -> prims.digamma (not implemented)
  #aten.erfinv,    # -> prims.erf_inv (not implemented)
  #aten.lgamma,    # -> prims.lgamma (not implemented)
  #aten.lerp,      # -> prims.copy_strided (not implemented)
]
for k,v in get_decompositions(decomps).items():
  key = str(k._schema).split("(")[0]
  if TORCH_DEBUG >= 2: print("register decomp for", k)
  torch.library.impl(key, "privateuseone")(v)

# NOTE: we should only implement the "out" form, it should be 0 overhead
# TODO: due to issue with empty / is_realized, it is slow to use assign so we use replace
# the goal is to make as much as we can this
simple_tensor_methods = [
  # NOTE: most unary/binary/trig ops now use decompositions via prims
  # only keep ops without decompositions or with aten-only decomps
  "silu", "hardsigmoid", "sigmoid", "clamp", "mish", "leaky_relu",  # activations (aten-only decomps or no decomp)
  "round",  # need direct impl for round.decimals_out special handling
  "copysign",  # no prims decomp
  "tril", "triu",  # no decomp
  "all", "any", "argmax", "argmin", "cumsum", "cumprod",  # reductions without prims decomp
  "avg_pool2d", "linspace"]  # complex ops

tiny_backend_out = {**{f"aten.{x}.out":getattr(Tensor,x) for x in simple_tensor_methods}, **{
  # NOTE: most ops now use decompositions via prims, only keep ops without decomps
  "aten.all.all_out": Tensor.all,  # needed for torch.all() without dim
  "aten.any.all_out": Tensor.any,  # needed for torch.any() without dim
  "aten.bmm.out": operator.matmul,  # no decomp
  "aten.clamp_max.Tensor_out": lambda input,max_: input.clamp(max_=max_),  # aten-only decomp
  "aten.clamp_min.Tensor_out": lambda input,min_: input.clamp(min_=min_),  # aten-only decomp
  "aten.round.decimals_out": lambda self,decimals: (self*10**decimals).round()/10**decimals,  # special case
  "aten.bitwise_left_shift.Tensor_out": lambda x,y: x*(2**y),  # keep for now
  "aten.bitwise_right_shift.Tensor_out": lambda x,y: x//(2**y),  # keep for now
  "aten.lerp.Scalar_out": Tensor.lerp,  # no prims.copy_strided
  "aten.scatter.value_out": Tensor.scatter,  # no decomp
  "aten.scatter.src_out": Tensor.scatter,  # no decomp
}}

# we add the "out" here
def wrap_out(f):
  def _wrap_out(*args, **kwargs):
    out = kwargs.pop('out')
    assigned = f(*args, **kwargs)
    if getenv("ALLOW_DTYPE_MISMATCH", 1): assigned = assigned.cast(out.dtype)
    assert out.shape == assigned.shape, f"shape mismatch: {assigned.shape} -> {out.shape}"
    assert out.device == assigned.device, f"device mismatch: {assigned.device} -> {out.device}"
    assert out.dtype == assigned.dtype, f"dtype mismatch: {assigned.dtype} -> {out.dtype}"
    return out.assign(assigned)
  return _wrap_out

def _inplace_op(t, new_value):
  if not hasattr(t, "_view_base") and not getattr(canonical_base(t), "_views", set()): t.replace(new_value)
  else: _apply_inplace(t, new_value)
  return t

tiny_backend = {**{k:wrap_out(v) for k,v in tiny_backend_out.items()}, **{
  # NOTE: most ops now use decompositions via prims, only keep ops without decomps or special cases
  "aten.floor_divide": lambda x,y: x//y,
  "aten.floor_divide_.Tensor": lambda x,y: x//y,
  "aten.__lshift__.Scalar": lambda x,y: x*(2**y),
  "aten.__ilshift__.Scalar": lambda x,y: x*(2**y),
  "aten.__rshift__.Scalar": lambda x,y: x//(2**y),
  "aten.__irshift__.Scalar": lambda x,y: x//(2**y),
  # inplace ops
  "aten.zero_": lambda x: x.zeros_like(),
  "aten.fill_.Scalar": lambda x, y: x.full_like(y),
  "aten.fill_.Tensor": lambda self, value: Tensor.full(self.shape, value.reshape(()).item(), device=self.device, dtype=self.dtype),
  "aten.add_.Tensor": lambda self, other, alpha=1.0: self + other * alpha,
  "aten.add_.Scalar": lambda self, other, alpha=1.0: self + other * alpha,
  "aten.mul_.Tensor": lambda self, other: self * other,
  "aten.mul_.Scalar": lambda self, other: self * other,
  "aten.relu": Tensor.relu,  # aten-only decomp
  "aten.relu_": lambda x: x.relu(),
  # ops without decompositions
  "aten.mean": Tensor.mean,
  "aten.mean.dim": Tensor.mean,
  "aten.min": Tensor.min,
  "aten.max": Tensor.max,
  "aten.max.dim": lambda self, dim, keepdim=False: (self.max(dim, keepdim), self.argmax(dim, keepdim).cast(dtype=dtypes.int64)),
  "aten.mm": Tensor.matmul,
  "aten.mv": Tensor.matmul,
  "aten.dot": Tensor.dot,
  "aten.isnan": Tensor.isnan,
  "aten.std.correction": Tensor.std,  # aten-only decomp
  "aten.std_mean.correction": Tensor.std_mean,  # aten-only decomp
  "aten.scatter.value": Tensor.scatter,
  "aten.scatter.value_reduce": Tensor.scatter,
  "aten.scatter_reduce.two": Tensor.scatter_reduce,
  "aten.gather": lambda self, dim, index: self.gather(dim, index.cast(dtypes.int)),
  "aten.repeat": lambda x,*repeats: Tensor.repeat(x,*repeats).contiguous(),  # aten-only decomp
  "aten._softmax": lambda self,dim,half_to_float: self.softmax(dim),  # aten-only decomp
  "aten._log_softmax": lambda self,dim,half_to_float: self.log_softmax(dim),  # aten-only decomp
  "aten.random_": lambda self: Tensor.randint(*self.shape, low=dtypes.min(self.dtype), high=dtypes.max(self.dtype), device=self.device, dtype=self.dtype),
  "aten.random_.from": lambda self, from_, to: Tensor.randint(*self.shape, low=from_, high=to, device=self.device, dtype=self.dtype),
  "aten.uniform_": lambda self, low=0, high=1: Tensor.uniform(*self.shape, low=low, high=high, dtype=self.dtype),
  "aten.normal_": lambda self, mean=0, std=1: Tensor.normal(*self.shape, mean=mean, std=std, dtype=self.dtype),
  "aten.logical_not": Tensor.logical_not,  # aten-only decomp
  "aten.logical_or_": lambda x, y: x | y,
  "aten.multinomial": Tensor.multinomial,
  "aten.masked_fill_.Scalar": lambda self, mask, value: self.masked_fill(mask, value),
  "aten.masked_fill_.Tensor": lambda self, mask, value: self.masked_fill(mask, value),
  "aten.masked_fill.Scalar": Tensor.masked_fill,  # aten-only decomp
  "aten.masked_fill.Tensor": Tensor.masked_fill,  # aten-only decomp
  "aten.masked_select": Tensor.masked_select,
  "aten.sgn": Tensor.sign,  # aten-only decomp
  "aten.argmax": Tensor.argmax,
  "aten.argmin": Tensor.argmin,
  "aten.squeeze_.dim": lambda self, dim: self.replace(self.squeeze(dim), allow_shape_mismatch=True),
  "aten.topk": Tensor.topk,
  "aten.cumsum": lambda self, dim: self.cumsum(dim).contiguous(),  # aten-only decomp
  "aten.logsumexp": lambda self, axis, keepdim=False: self.logsumexp(axis[0], keepdim=keepdim),  # aten-only decomp
  "aten.roll": Tensor.roll,  # aten-only decomp
  "aten.logcumsumexp": Tensor.logcumsumexp,
  "aten.lerp.Tensor": Tensor.lerp,  # no prims.copy_strided
  "aten.ones_like": lambda self, dtype=None, device=None, **kwargs:
    self.ones_like(**{k: v for k, v in {"dtype": _from_torch_dtype(dtype) if dtype else None,
                                        "device": _from_torch_device(device) if device else None}.items() if v is not None}),
  "aten.unfold": Tensor.unfold,  # aten-only decomp
}}

# operations that need inplace treatment (use _inplace_op instead of wrap_fxn) AKA return original tensor
inplace_ops = {
  "aten.zero_",
  "aten.fill_.Scalar",
  "aten.fill_.Tensor",
  "aten.add_.Tensor",
  "aten.add_.Scalar",
  "aten.mul_.Tensor",
  "aten.mul_.Scalar",
  "aten.floor_divide_.Tensor",
  "aten.__ilshift__.Scalar",
  "aten.__irshift__.Scalar",
  "aten.relu_",
  "aten.random_",
  "aten.random_.from",
  "aten.uniform_",
  "aten.normal_",
  "aten.logical_or_",
  "aten.masked_fill_.Scalar",
  "aten.masked_fill_.Tensor",
}

def wrap_fxn(k,f):
  def nf(*args, **kwargs):
    if TORCH_DEBUG:
      print(k, len(args), [x.shape if isinstance(x, torch.Tensor) else x for x in args],
                          {k:v.shape if isinstance(v, torch.Tensor) else v for k,v in kwargs.items()})
    args, kwargs = unwrap_args(args, kwargs)
    out = f(*args, **kwargs)
    if isinstance(out, Tensor): return wrap(out)
    elif isinstance(out, tuple): return tuple(wrap(x) for x in out)
    else: raise RuntimeError(f"unknown output type {type(out)}")
  return nf

def wrap_inplace(k,f):
  def nf(*args, **kwargs):
    orig = args[0]
    args, kwargs = unwrap_args(args, kwargs)
    _inplace_op(args[0], f(*args, **kwargs))
    return orig
  return nf

for k,v in tiny_backend.items():
  wrapper = wrap_inplace if k in inplace_ops else wrap_fxn
  torch.library.impl(k.replace("aten.", "aten::"), "privateuseone")(wrapper(k,v))

@torch.library.impl("aten::equal", "privateuseone")
def equal(x: torch.Tensor, y: torch.Tensor): return (x==y).all().item()

if TORCH_DEBUG:
  from torch.utils._python_dispatch import TorchDispatchMode
  class DispatchLog(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      #print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
      print(f"Dispatch Log: {func}")
      return func(*args, **(kwargs or {}))
  (_dispatch_log:=DispatchLog()).__enter__() # NOTE: must be kept alive

# this implementation is needed to allow the batchnorm kernels to fuse in e.g. mnist training
# aten::native_batch_norm does more than Tensor.batchnorm
@torch.library.impl("aten::native_batch_norm", "privateuseone")
def native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps):
  input_t, weight_t, bias_t = unwrap(input), unwrap(weight) if weight is not None else None, unwrap(bias) if bias is not None else None
  running_mean_t, running_var_t = unwrap(running_mean) if running_mean is not None else None, unwrap(running_var) if running_var is not None else None
  if training:
    batch_var, batch_mean = input_t.var_mean(axis=tuple(x for x in range(input_t.ndim) if x != 1), correction=0)
    batch_invstd = batch_var.add(eps).rsqrt()
    out = input_t.batchnorm(weight_t, bias_t, batch_mean, batch_invstd)
    if running_mean_t is not None and running_var_t is not None:
      numel_ratio = input_t.numel() / (input_t.numel() - input_t.shape[1])
      running_mean_t.assign((1 - momentum) * running_mean_t + momentum * batch_mean.detach())
      running_var_t.assign((1 - momentum) * running_var_t + momentum * numel_ratio * batch_var.detach())
    return wrap(out), wrap(batch_mean), wrap(batch_invstd)
  else:
    out = input_t.batchnorm(weight_t, bias_t, running_mean_t, running_var_t.add(eps).rsqrt())
    return wrap(out), wrap(running_mean_t), wrap(running_var_t.add(eps).rsqrt())

@torch.library.impl("aten::native_batch_norm_backward", "privateuseone")
def native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask):
  grad_out_t, input_t = unwrap(grad_out), unwrap(input)
  weight_t = unwrap(weight) if weight is not None else None
  save_mean_t = unwrap(save_mean)
  save_invstd_t = unwrap(save_invstd)
  out = input_t.batchnorm(weight_t, None, save_mean_t, save_invstd_t)
  targets = [t for t, m in zip([input_t, weight_t], output_mask[:2]) if t is not None and m]
  if targets:
    grads = out.gradient(*targets, gradient=grad_out_t)
    grad_input = grads.pop(0) if output_mask[0] else None
    grad_weight = grads.pop(0) if output_mask[1] and weight_t is not None else None
  else:
    grad_input, grad_weight = None, None
  grad_bias = grad_out_t.sum(axis=tuple(x for x in range(grad_out_t.ndim) if x != 1)) if output_mask[2] else None
  return (wrap(grad_input) if grad_input is not None else None,
          wrap(grad_weight) if grad_weight is not None else None,
          wrap(grad_bias) if grad_bias is not None else None)

# _pad_circular is not CompositeImplicitAutograd (unlike reflect/replicate pad)
# we need torch.autograd.Function with explicit AutogradPrivateUse1 registration
class _PadCircular(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, padding):
    ctx.save_for_backward(input)
    ctx.padding = padding
    return pad_forward(input, padding, mode="circular")
  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return pad_backward(grad_output, input, ctx.padding, mode="circular"), None

@torch.library.impl("aten::_pad_circular", "privateuseone")
def _pad_circular(self, padding): return _PadCircular.apply(self, padding)

@torch.library.impl("aten::_pad_circular", "AutogradPrivateUse1")
def _pad_circular_autograd(self, padding): return _PadCircular.apply(self, padding)

# *** prims implementations (enables torch._refs decompositions) ***

@torch.library.impl("prims::var", "privateuseone")
def prims_var(inp, dims, correction=1, *, output_dtype=None):
  return wrap(unwrap(inp).var(axis=tuple(dims) if dims else None, correction=correction))

@torch.library.impl("prims::sum", "privateuseone")
def prims_sum(inp, dims, *, output_dtype=None):
  result = unwrap(inp).sum(axis=tuple(dims) if dims else None)
  if output_dtype: result = result.cast(_from_torch_dtype(output_dtype))
  return wrap(result)

@torch.library.impl("prims::div", "privateuseone")
def prims_div(self, other):
  return wrap(unwrap(self) / (unwrap(other) if isinstance(other, torch.Tensor) else other))

@torch.library.impl("prims::sub", "privateuseone")
def prims_sub(self, other):
  return wrap(unwrap(self) - (unwrap(other) if isinstance(other, torch.Tensor) else other))

@torch.library.impl("prims::mul", "privateuseone")
def prims_mul(self, other):
  return wrap(unwrap(self) * (unwrap(other) if isinstance(other, torch.Tensor) else other))

@torch.library.impl("prims::broadcast_in_dim", "privateuseone")
def prims_broadcast_in_dim(a, shape, broadcast_dimensions):
  t = unwrap(a)
  # Create shape with 1s, then place original dims at broadcast_dimensions positions
  expand_shape = [1] * len(shape)
  for i, dim in enumerate(broadcast_dimensions): expand_shape[dim] = t.shape[i]
  return wrap(t.reshape(expand_shape).expand(shape))

@torch.library.impl("prims::convert_element_type", "privateuseone")
def prims_convert_element_type(a, dtype):
  return wrap(unwrap(a).cast(_from_torch_dtype(dtype)))

@torch.library.impl("prims::squeeze", "privateuseone")
def prims_squeeze(a, dimensions):
  t = unwrap(a)
  for dim in sorted(dimensions, reverse=True): t = t.squeeze(dim)
  return wrap(t)

@torch.library.impl("prims::rsqrt", "privateuseone")
def prims_rsqrt(self):
  return wrap(unwrap(self).rsqrt())

# unary prims
@torch.library.impl("prims::abs", "privateuseone")
def prims_abs(self): return wrap(unwrap(self).abs())

@torch.library.impl("prims::neg", "privateuseone")
def prims_neg(self): return wrap(-unwrap(self))

@torch.library.impl("prims::sqrt", "privateuseone")
def prims_sqrt(self): return wrap(unwrap(self).sqrt())

@torch.library.impl("prims::reciprocal", "privateuseone")
def prims_reciprocal(self): return wrap(unwrap(self).reciprocal())

@torch.library.impl("prims::exp", "privateuseone")
def prims_exp(self): return wrap(unwrap(self).exp())

@torch.library.impl("prims::exp2", "privateuseone")
def prims_exp2(self): return wrap(unwrap(self).exp2())

@torch.library.impl("prims::expm1", "privateuseone")
def prims_expm1(self): return wrap(unwrap(self).exp() - 1)

@torch.library.impl("prims::log", "privateuseone")
def prims_log(self): return wrap(unwrap(self).log())

@torch.library.impl("prims::log2", "privateuseone")
def prims_log2(self): return wrap(unwrap(self).log2())

@torch.library.impl("prims::log10", "privateuseone")
def prims_log10(self): return wrap(unwrap(self).log10())

@torch.library.impl("prims::log1p", "privateuseone")
def prims_log1p(self): return wrap((unwrap(self) + 1).log())

@torch.library.impl("prims::sin", "privateuseone")
def prims_sin(self): return wrap(unwrap(self).sin())

@torch.library.impl("prims::cos", "privateuseone")
def prims_cos(self): return wrap(unwrap(self).cos())

@torch.library.impl("prims::tan", "privateuseone")
def prims_tan(self): return wrap(unwrap(self).tan())

@torch.library.impl("prims::sinh", "privateuseone")
def prims_sinh(self): return wrap(unwrap(self).sinh())

@torch.library.impl("prims::cosh", "privateuseone")
def prims_cosh(self): return wrap(unwrap(self).cosh())

@torch.library.impl("prims::tanh", "privateuseone")
def prims_tanh(self): return wrap(unwrap(self).tanh())

@torch.library.impl("prims::asin", "privateuseone")
def prims_asin(self): return wrap(unwrap(self).asin())

@torch.library.impl("prims::acos", "privateuseone")
def prims_acos(self): return wrap(unwrap(self).acos())

@torch.library.impl("prims::atan", "privateuseone")
def prims_atan(self): return wrap(unwrap(self).atan())

@torch.library.impl("prims::asinh", "privateuseone")
def prims_asinh(self): return wrap(unwrap(self).asinh())

@torch.library.impl("prims::acosh", "privateuseone")
def prims_acosh(self): return wrap(unwrap(self).acosh())

@torch.library.impl("prims::atanh", "privateuseone")
def prims_atanh(self): return wrap(unwrap(self).atanh())

@torch.library.impl("prims::ceil", "privateuseone")
def prims_ceil(self): return wrap(unwrap(self).ceil())

@torch.library.impl("prims::floor", "privateuseone")
def prims_floor(self): return wrap(unwrap(self).floor())

@torch.library.impl("prims::round", "privateuseone")
def prims_round(self): return wrap(unwrap(self).round())

@torch.library.impl("prims::trunc", "privateuseone")
def prims_trunc(self): return wrap(unwrap(self).trunc())

@torch.library.impl("prims::sign", "privateuseone")
def prims_sign(self): return wrap(unwrap(self).sign())

@torch.library.impl("prims::erf", "privateuseone")
def prims_erf(self): return wrap(unwrap(self).erf())

@torch.library.impl("prims::bitwise_not", "privateuseone")
def prims_bitwise_not(self): return wrap(unwrap(self).bitwise_not())

# binary prims - handle both tensor and scalar inputs
def _unwrap_or_scalar(x): return unwrap(x) if isinstance(x, torch.Tensor) else x

@torch.library.impl("prims::add", "privateuseone")
def prims_add(self, other): return wrap(unwrap(self) + _unwrap_or_scalar(other))

@torch.library.impl("prims::maximum", "privateuseone")
def prims_maximum(self, other): return wrap(Tensor.maximum(unwrap(self), _unwrap_or_scalar(other)))

@torch.library.impl("prims::minimum", "privateuseone")
def prims_minimum(self, other): return wrap(Tensor.minimum(unwrap(self), _unwrap_or_scalar(other)))

@torch.library.impl("prims::pow", "privateuseone")
def prims_pow(self, other): return wrap(unwrap(self).pow(_unwrap_or_scalar(other)))

@torch.library.impl("prims::remainder", "privateuseone")
def prims_remainder(self, other): return wrap(unwrap(self) % _unwrap_or_scalar(other))

@torch.library.impl("prims::fmod", "privateuseone")
def prims_fmod(self, other):
  x, y = unwrap(self), _unwrap_or_scalar(other)
  return wrap(x - x.div(y, rounding_mode="trunc") * y)

@torch.library.impl("prims::atan2", "privateuseone")
def prims_atan2(self, other): return wrap(Tensor.atan2(unwrap(self), _unwrap_or_scalar(other)))

@torch.library.impl("prims::bitwise_and", "privateuseone")
def prims_bitwise_and(self, other): return wrap(unwrap(self).bitwise_and(_unwrap_or_scalar(other)))

@torch.library.impl("prims::bitwise_or", "privateuseone")
def prims_bitwise_or(self, other): return wrap(unwrap(self).bitwise_or(_unwrap_or_scalar(other)))

@torch.library.impl("prims::bitwise_xor", "privateuseone")
def prims_bitwise_xor(self, other): return wrap(unwrap(self).bitwise_xor(_unwrap_or_scalar(other)))

@torch.library.impl("prims::shift_left", "privateuseone")
def prims_shift_left(self, other): return wrap(unwrap(self) * (2 ** _unwrap_or_scalar(other)))

@torch.library.impl("prims::shift_right_arithmetic", "privateuseone")
def prims_shift_right_arithmetic(self, other): return wrap(unwrap(self) // (2 ** _unwrap_or_scalar(other)))

# comparison prims
@torch.library.impl("prims::eq", "privateuseone")
def prims_eq(self, other): return wrap(unwrap(self).eq(_unwrap_or_scalar(other)))

@torch.library.impl("prims::ne", "privateuseone")
def prims_ne(self, other): return wrap(unwrap(self).ne(_unwrap_or_scalar(other)))

@torch.library.impl("prims::lt", "privateuseone")
def prims_lt(self, other): return wrap(unwrap(self) < _unwrap_or_scalar(other))

@torch.library.impl("prims::le", "privateuseone")
def prims_le(self, other): return wrap(unwrap(self) <= _unwrap_or_scalar(other))

@torch.library.impl("prims::gt", "privateuseone")
def prims_gt(self, other): return wrap(unwrap(self) > _unwrap_or_scalar(other))

@torch.library.impl("prims::ge", "privateuseone")
def prims_ge(self, other): return wrap(unwrap(self) >= _unwrap_or_scalar(other))

# reduction prims
@torch.library.impl("prims::prod", "privateuseone")
def prims_prod(inp, dims, *, output_dtype=None):
  result = unwrap(inp).prod(axis=tuple(dims) if dims else None)
  if output_dtype: result = result.cast(_from_torch_dtype(output_dtype))
  return wrap(result)

@torch.library.impl("prims::amax", "privateuseone")
def prims_amax(inp, dims, *, output_dtype=None):
  return wrap(unwrap(inp).max(axis=tuple(dims) if dims else None))

@torch.library.impl("prims::amin", "privateuseone")
def prims_amin(inp, dims, *, output_dtype=None):
  return wrap(unwrap(inp).min(axis=tuple(dims) if dims else None))

# other essential prims
@torch.library.impl("prims::where", "privateuseone")
def prims_where(pred, a, b): return wrap(Tensor.where(unwrap(pred), _unwrap_or_scalar(a), _unwrap_or_scalar(b)))

@torch.library.impl("prims::cat", "privateuseone")
def prims_cat(tensors, dim): return wrap(Tensor.cat(*[unwrap(t) for t in tensors], dim=dim))

@torch.library.impl("prims::clone", "privateuseone")
def prims_clone(self, *, memory_format=None): return wrap(unwrap(self).contiguous())

@torch.library.impl("prims::view_of", "privateuseone")
def prims_view_of(a): return wrap(unwrap(a))

@torch.library.impl("prims::transpose", "privateuseone")
def prims_transpose(a, permutation): return wrap(unwrap(a).permute(permutation))

@torch.library.impl("prims::rev", "privateuseone")
def prims_rev(a, dims): return wrap(unwrap(a).flip(dims))

@torch.library.impl("prims::fmax", "privateuseone")
def prims_fmax(self, other):
  x, y = unwrap(self), unwrap(other)
  return wrap(Tensor.where(x.isnan() & ~y.isnan(), y, Tensor.where(~x.isnan() & y.isnan(), x, Tensor.maximum(x, y))))

@torch.library.impl("prims::fmin", "privateuseone")
def prims_fmin(self, other):
  x, y = unwrap(self), unwrap(other)
  return wrap(Tensor.where(x.isnan() & ~y.isnan(), y, Tensor.where(~x.isnan() & y.isnan(), x, Tensor.minimum(x, y))))

@torch.library.impl("prims::iota", "privateuseone")
def prims_iota(length, *, start, step, dtype, device, requires_grad):
  return wrap(Tensor.arange(start, start + length * step, step, dtype=_from_torch_dtype(dtype), device=_from_torch_device(device)))
