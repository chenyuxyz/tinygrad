# a dynamo backend that lowers the AOT-autograd FX graph into a TinyJit, so a compiled function replays kernels
# instead of dispatching op-by-op. importing this makes a bare torch.compile use it
import torch
from torch._dynamo.backends.registry import register_backend, set_default_backend
from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_func
from tinygrad import Tensor, TinyJit
from extra.torch_backend.backend import wrap, unwrap, canonical_base, register_view, _get_view_ops

# TinyJit is static shape, and a dim dynamo marked dynamic arrives as a SymInt with no Tensor to bind. NOTE: process wide
torch._dynamo.config.automatic_dynamic_shapes = False

def _jit_input(x:torch.Tensor) -> Tensor:
  # a JIT input has to be a real buffer: .tiny() brings other devices over, .clone() gives a deviceless const a device
  t = unwrap(x.tiny())
  return t if t.device is not None else t.replace(t.clone(t._torch_device))

def _copy_out(outs:list[Tensor|None]) -> list[Tensor|None]:
  # copy every output, recording a returned view as a view of its base's copy. sharing one buffer instead would hand the
  # next graph two JIT inputs on it, which jit.py rejects. one realize for all: per-output is a graph rewrite each
  ret:list[Tensor|None] = [None if t is None else t.clone() for t in outs]
  copies = {t: c for t, c in zip(outs, ret) if t is not None and t is canonical_base(t)}
  for t, c in zip(outs, ret):
    if t is None or t is (b:=canonical_base(t)): continue
    if (base_copy:=copies.get(b)) is not None: register_view(base_copy, c, _get_view_ops(t))
  if (real:=[t for t in ret if t is not None]): Tensor.realize(*real)
  return ret

def _tiny_compiler(gm:torch.fx.GraphModule, sample_inputs):
  # dynamo emits op-free graphs at training graph breaks and TinyJit cannot capture one, so run it eagerly
  if not any(n.op in ("call_function", "call_method", "call_module") for n in gm.graph.nodes):
    return make_boxed_func(torch.compiler.disable(gm))
  # the JIT hands back the buffers it captured: copy inside so a passthrough output is written on this call, copy outside
  # so a retained output survives the next. a backward graph has a None output per input that wanted no grad
  @TinyJit
  def jitted(*args:Tensor): return _copy_out([x if x is None else unwrap(x) for x in gm(*[wrap(a) for a in args])])
  # this runs under an active dynamo, which would otherwise trace the backend itself
  @torch.compiler.disable
  def run(*args:torch.Tensor):
    return [t if t is None else wrap(t) for t in _copy_out(jitted(*map(_jit_input, args)))]
  return make_boxed_func(run)

# mode= and options= are inductor knobs torch hands to any named backend, drop them like torch's own aot_autograd does
@register_backend
def tiny(gm:torch.fx.GraphModule, sample_inputs, **_ignored):
  # a non-Tensor input is a SymInt for a dynamic dim. NOTE: never format a tiny tensor in here, repr() of one runs a
  # float64 reduction that METAL cannot compile
  if not all(isinstance(x, torch.Tensor) for x in sample_inputs):
    raise RuntimeError(f"the tiny backend needs static shapes, got {[type(x).__name__ for x in sample_inputs]}")
  return aot_module_simplified(gm, sample_inputs, fw_compiler=_tiny_compiler)

# torch.compile defaults to inductor, which cannot run a tiny tensor. NOTE: process wide, like the config above
set_default_backend("tiny")
