"""Winograd convolution as a UOp graph rewrite (http://arxiv.org/abs/1509.09308).

Detection: walk back through movement ops with `apply_movement_op`, then verify the resulting
coordinate expressions describe a 3^n or 5^n convolution via affine analysis on RANGE UOps
(no fragile tree-shape matching).

Supports kernel sizes 3 (F(4,3)) and 5 (F(2,5)) — both share the same alpha=6 `Bt`. Catches:
forward conv (with optional bias), grouped/depthwise/`cin==1`, mixed-dtype accumulate (top-level
CAST in reduce body), dilation=2 (rewritten as a sparse 5x5), stride-1 transposed conv (axes-
swapped + flipped weight), and the backward of any non-degenerate loss (the forward conv pattern
survives in `compute_gradient`'s output and gets caught at schedule time).

Provably out of scope: stride > 1 (FLOP ratio `(m+K-1)^d / (s^d K^d)` < 1 only when `s < (m+K-1)/K`,
which fails for `s >= 2` with both F(4,3) and F(2,5)); kernel sizes other than 3 or 5 (need
different transform matrices); raw `dw` filter-gradient (the small object is the *output*, not
the input — needs a different Winograd algorithm).
"""
from __future__ import annotations
import itertools
from typing import TYPE_CHECKING
from tinygrad.dtype import DType, DTypeLike, Invalid
from tinygrad.helpers import WINO, flatten, flat_to_grouped, prod, resolve_pool_pads
from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.uop.ops import GroupOp, Ops, PatternMatcher, UOp, UPat, graph_rewrite
from tinygrad.uop.symbolic import propagate_invalid

if TYPE_CHECKING:
  from tinygrad.tensor import Tensor
  from tinygrad.uop.ops import sint

# *** winograd math (reused tensor-level transform) ***

winograd_G  = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
winograd_G_5 = [[1/4, 0, 0, 0, 0], [-1/6, -1/6, -1/6, -1/6, -1/6], [-1/6, 1/6, -1/6, 1/6, -1/6],
                [1/24, 1/12, 1/6, 1/3, 2/3], [1/24, -1/12, 1/6, -1/3, 2/3], [0, 0, 0, 0, 1]]
winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]]
winograd_At_5 = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 1]]
WINO_MATRICES = {3:(winograd_G, winograd_Bt, winograd_At), 5:(winograd_G_5, winograd_Bt, winograd_At_5)}

def _matcols(mat, dims:int, shp:tuple[sint, ...], device:str|tuple[str, ...], dtype:DType) -> list[list[Tensor]]:
  from tinygrad.tensor import Tensor
  return [[Tensor.cat(*[Tensor.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), device=device, dtype=dtype) for m in mat], dim=dim)
           for k in range(len(mat[0]))] for dim in range(dims)]

def _apply_winograd_matrix(mat, t:Tensor, dims:int) -> Tensor:
  from tinygrad.tensor import Tensor
  t_ = t.reshape(t.shape[:dims] + (1,)*dims + t.shape[dims:]).expand(t.shape[:dims] + (len(mat),)*dims + t.shape[dims:])
  cols = _matcols(mat, dims, t_.shape[dims:], t_.device, t_.dtype)
  ret = sum(prod(col[idx] for col, idx in zip(cols, mat_is)) * t_[mat_is] for mat_is in itertools.product(range(len(mat[0])), repeat=dims))
  assert isinstance(ret, Tensor), "sum didn't return a Tensor"
  return ret

def wino_conv(x_uop:UOp, weight_uop:UOp, bias_uop:UOp|None, groups:int, padding, dtype:DTypeLike|None, kernel:int) -> UOp:
  from tinygrad.tensor import Tensor
  x, weight = Tensor(x_uop), Tensor(weight_uop)
  bias = Tensor(bias_uop) if bias_uop is not None else None
  (bs, _), (cout, cin), HW = x.shape[:2], weight.shape[:2], weight.shape[2:]
  G, Bt, At = WINO_MATRICES[kernel]
  padding_ = resolve_pool_pads(padding, len(HW))
  rcout, oyx = cout // groups, x.pad(padding_)._pool(HW, 1, 1).shape[2:-len(HW)]
  HWI, HWO = (len(Bt),) * len(HW), (len(At),) * len(HW)
  pads = [(pB, pA + (-(s + pB + pA - (kernel-1)) % len(At))) for (pB, pA), s in zip(flat_to_grouped(padding_), x.shape[-len(HW):])]
  d = x.pad(flatten(reversed(pads)))._pool(HWI, HWO)
  d = d.permute(*range(len(d.shape)-len(HW), len(d.shape)), *range(len(d.shape)-len(HW)))
  tyx = d.shape[-len(HWI):]
  g = weight.permute(*range(len(weight.shape)-len(HW), len(weight.shape)), *range(len(weight.shape)-len(HW)))
  gfactors = _apply_winograd_matrix(G,  g, len(HW)).reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
  dfactors = _apply_winograd_matrix(Bt, d, len(HW)).reshape(*HWI, bs, groups, 1, cin, *tyx)
  # contiguous() is load-bearing: without it `cin==1` fuses the whole pipeline back into a single kernel.
  prod_factors = (gfactors * dfactors).sum(axis=-1-len(HW), dtype=dtype).contiguous()
  ret = _apply_winograd_matrix(At, prod_factors, len(HW))
  ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW), 0]]])
  ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink_to(bs, cout, *oyx)
  return (ret if bias is None else ret.add(bias.reshape(1, -1, *[1]*len(HW)))).contiguous().contiguous_backward().uop

# *** principled detection: affine analysis + walking through movement ops ***

# An affine expression over RANGE UOps: ({range: coefficient}, intercept)
def _strip_invalid(x:UOp) -> UOp:
  x = graph_rewrite(x, propagate_invalid, name="wino affine")
  return x.src[1] if x.op is Ops.WHERE and x.src[2].op is Ops.CONST and x.src[2].arg is Invalid else x

def _affine(x:UOp) -> tuple[dict[UOp, int], int]|None:
  coeffs: dict[UOp, int] = {}
  intercept = 0
  for term in _strip_invalid(x).split_uop(Ops.ADD):
    term = _strip_invalid(term)
    if term.op is Ops.CONST and isinstance(term.arg, int):
      intercept += term.arg
      continue
    if term.op is Ops.RANGE:
      coeffs[term] = coeffs.get(term, 0) + 1
      continue
    if term.op is Ops.MUL:
      if all(s.op is Ops.CONST and isinstance(s.arg, int) for s in term.src):
        intercept += term.src[0].arg * term.src[1].arg
        continue
      for rng, c in (term.src, term.src[::-1]):
        if rng.op is Ops.RANGE and c.op is Ops.CONST and isinstance(c.arg, int):
          coeffs[rng] = coeffs.get(rng, 0) + c.arg
          break
      else: return None
      continue
    return None
  return {k:v for k,v in coeffs.items() if v}, intercept

def _axis_coeff(c:UOp, want:UOp|None=None) -> tuple[UOp, int, int]|None:
  if (a:=_affine(c)) is None or len(a[0]) != 1: return None
  ax, coeff = next(iter(a[0].items()))
  return (ax, coeff, a[1]) if (want is None or ax is want) else None

def _is_axis(c:UOp, want:UOp|None=None) -> UOp|None:
  """If `c` is exactly one RANGE (coeff 1, intercept 0), return it (optionally checking equality with `want`)."""
  return axc[0] if (axc:=_axis_coeff(c, want)) is not None and axc[1:] == (1, 0) else None

def _walk(x:UOp, coords:tuple[UOp, ...], dims:int) -> tuple[UOp, tuple[UOp, ...]]|None:
  """Walk back through movement ops, propagating coords. Return the deepest node of shape `dims`."""
  best: tuple[UOp, tuple[UOp, ...]]|None = None
  while True:
    if len(x.shape) == dims: best = (x, coords)
    if x.op not in GroupOp.Movement: return best
    coords, x = apply_movement_op(x.op, x.src[0].shape, x.marg, coords), x.src[0]

def _dilate_weight2(weight_uop:UOp, dims:int) -> UOp:
  from tinygrad.tensor import Tensor
  t = Tensor(weight_uop)
  for axis in range(t.ndim-dims, t.ndim):
    shp = list(t.shape); shp.insert(axis+1, 1)
    t = t.reshape(tuple(shp))
    t = t.pad(tuple((0, 1) if i == axis+1 else None for i in range(t.ndim)))
    shp = list(t.shape); shp[axis] *= shp.pop(axis+1)
    t = t.reshape(tuple(shp))
    t = t.shrink(tuple((0, t.shape[i]-1) if i == axis else None for i in range(t.ndim)))
  return t.uop

def _match_weight(side:UOp, coords:tuple[UOp, ...], dims:int, reduced:set[UOp]):
  """Weight: shape (cout, cin, *(kernel,)*dims). Coords: (cout_axis, cin_axis, *kernel_axes)."""
  if (w:=_walk(side, coords, dims+2)) is None or len(set(ks:=w[0].shape[-dims:])) != 1 or ks[0] not in WINO_MATRICES: return None
  weight_uop, wc = w
  kaxes = tuple(_is_axis(c) for c in wc[-dims:])
  if any(k is None or k not in reduced for k in kaxes) or len(set(kaxes)) != dims: return None
  if (cin_ax:=_is_axis(wc[1])) is not None:
    if cin_ax not in reduced or cin_ax in kaxes: return None
  elif _affine(wc[1]) != ({}, 0): return None
  return weight_uop, cin_ax, kaxes, ks[0]

def _match_tconv_weight(side:UOp, coords:tuple[UOp, ...], dims:int, reduced:set[UOp]):
  """Transposed-conv weight: shape (cin, cout, *(kernel,)*dims) with flipped kernel coords."""
  from tinygrad.tensor import Tensor
  if (w:=_walk(side, coords, dims+2)) is None or len(set(ks:=w[0].shape[-dims:])) != 1 or ks[0] not in WINO_MATRICES: return None
  weight_uop, wc = w
  if (cin_ax:=_is_axis(wc[0])) is None or cin_ax not in reduced: return None
  if (cout_ax:=_is_axis(wc[1])) is None or cout_ax in reduced or cout_ax is cin_ax: return None
  if any((axc:=_axis_coeff(c)) is None or axc[0] not in reduced or axc[1:] != (-1, ks[0]-1) for c in wc[-dims:]): return None
  return Tensor(weight_uop).permute(1, 0, *range(2, dims+2)).flip(*range(2, dims+2)).uop, cin_ax, ks[0]

def _match_act(side:UOp, coords:tuple[UOp, ...], dims:int, cin_ax:UOp|None, kaxes:tuple[UOp, ...], reduced:set[UOp], kernel:int):
  """Activation: shape (bs, cin_total, *spatial). Coords: (bs_axis, cin_combined, *(stride*oy + dilation*ky pairs))."""
  if (w:=_walk(side, coords, dims+2)) is None: return None
  x_uop, ac = w
  ch = _affine(ac[1])
  blocked = reduced if cin_ax is None else reduced - {cin_ax}
  if ch is None or ch[1] != 0 or any(v in blocked for v in ch[0]): return None
  if cin_ax is not None and ch[0].get(cin_ax) != 1: return None
  pads, oy_axes, strides, dilations = [], [], [], []
  for in_size, c, k in zip(x_uop.shape[-dims:], ac[-dims:], kaxes):
    if not isinstance(in_size, int) or (a:=_affine(c)) is None or (dilation:=a[0].get(k)) is None or dilation <= 0: return None
    rest = [(v, n) for v, n in a[0].items() if v is not k]
    if len(rest) != 1 or (stride:=rest[0][1]) <= 0 or rest[0][0] in reduced or rest[0][0].src[0].op is not Ops.CONST: return None
    oy_axes.append(rest[0][0])
    strides.append(stride); dilations.append(dilation)
    pads.append((-a[1], stride*rest[0][0].src[0].arg - in_size + dilation*(kernel-1) + a[1] - stride + 1))
  if len(set(oy_axes)) != dims or len(set(strides)) != 1 or len(set(dilations)) != 1: return None
  return x_uop, tuple(flatten(reversed(pads))), strides[0], dilations[0]

def _match_bias(side:UOp, coords:tuple[UOp, ...], ch_axis:UOp) -> UOp|None:
  if (w:=_walk(side, coords, dims=1)) is None: return None
  return w[0] if _is_axis(w[1][0], want=ch_axis) is not None else None

def _detect_conv(reduce:UOp) -> tuple[UOp, UOp, int, tuple[int, ...], int]|None:
  """Returns (x, weight, groups, padding, kernel) if `reduce` is a conv-style REDUCE-of-MUL, else None."""
  # MUL shape from OpMixin.conv2d is (bs, groups, rcout, *oyx, cin, *HW), so dims comes from the shape itself
  if reduce.arg[0] is not Ops.ADD: return None
  mul = reduce.src[0].src[0] if reduce.src[0].op is Ops.CAST and reduce.src[0].src[0].op is Ops.MUL else reduce.src[0]
  if mul.op is not Ops.MUL or (dims:=(len(mul.shape)-4)//2) <= 0: return None
  coords = tuple(UOp.range(s, i) for i, s in enumerate(mul.shape))
  reduced = {coords[i] for i in reduce.arg[1]}
  for wt_side, act_side in (mul.src, mul.src[::-1]):
    if (wt:=_match_weight(wt_side, coords, dims, reduced)) is None: continue
    if (act:=_match_act(act_side, coords, dims, wt[1], wt[2], reduced, wt[3])) is None: continue
    if act[2] != 1: return None
    weight_uop, x_uop, padding, kernel = wt[0], act[0], act[1], wt[3]
    if act[3] == 2:
      if kernel != 3: return None
      weight_uop, kernel = _dilate_weight2(weight_uop, dims), 5
    elif act[3] != 1: return None
    cin, chans, cout = weight_uop.shape[1], x_uop.shape[1], weight_uop.shape[0]
    if not all(isinstance(v, int) and v > 0 for v in (cin, chans, cout)) or chans % cin: return None
    if cout % (groups:=chans // cin): return None
    return x_uop, weight_uop, groups, padding, kernel
  return None

def _detect_tconv(reduce:UOp) -> tuple[UOp, UOp, int, tuple[int, ...], int]|None:
  if reduce.arg[0] is not Ops.ADD or reduce.src[0].op is not Ops.MUL or (dims:=(len(reduce.src[0].shape)-4)//2) <= 0: return None
  coords = tuple(UOp.range(s, i) for i, s in enumerate(reduce.src[0].shape))
  reduced = {coords[i] for i in reduce.arg[1]}
  for wt_side, act_side in (reduce.src[0].src, reduce.src[0].src[::-1]):
    if (wt:=_match_tconv_weight(wt_side, coords, dims, reduced)) is None: continue
    if (act:=_match_act(act_side, coords, dims, wt[1], tuple(coords[i] for i in reduce.arg[1][1:]), reduced, wt[2])) is None or act[2:] != (1, 1): continue
    cin, chans, cout = wt[0].shape[1], act[0].shape[1], wt[0].shape[0]
    if not all(isinstance(v, int) and v > 0 for v in (cin, chans, cout)) or chans % cin: return None
    if cout % (groups:=chans // cin): return None
    return act[0], wt[0], groups, act[1], wt[2]
  return None

def _detect_wino(reduce:UOp) -> tuple[UOp, UOp, int, tuple[int, ...], int]|None: return _detect_conv(reduce) or _detect_tconv(reduce)

def _try_wino_reduce(reduce:UOp) -> UOp|None:
  if not WINO.value or (det:=_detect_wino(reduce)) is None: return None
  return wino_conv(det[0], det[1], None, det[2], det[3], reduce.dtype, det[4])

def _find_reduce(x:UOp) -> UOp|None:
  """Walk back through movement ops to find a REDUCE."""
  while x.op in GroupOp.Movement: x = x.src[0]
  return x if x.op is Ops.REDUCE else None

def _try_wino_add(add:UOp) -> UOp|None:
  """ADD(conv-reduce-output via movement ops, bias-broadcast)."""
  if not WINO.value or len(add.shape) < 2: return None
  for reduce_side, bias_side in (add.src, add.src[::-1]):
    if (reduce:=_find_reduce(reduce_side)) is None or (det:=_detect_wino(reduce)) is None: continue
    # synthetic axis vars for symbolically walking the bias side; the high index avoids clashing with real range IDs
    fresh = tuple(UOp.range(s, 9001+i) for i, s in enumerate(add.shape))
    if (bias:=_match_bias(bias_side, fresh, fresh[1])) is None: continue
    return wino_conv(det[0], det[1], bias, det[2], det[3], reduce.dtype, det[4])
  return None

pm_wino = PatternMatcher([
  (UPat(Ops.ADD,    name="add"),    _try_wino_add),
  (UPat(Ops.REDUCE, name="reduce"), _try_wino_reduce),
])
