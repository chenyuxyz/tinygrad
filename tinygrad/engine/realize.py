from typing import cast, Callable
import time, pprint, random, itertools, math, struct
from dataclasses import dataclass, replace, field
from tinygrad.helpers import all_same, colored, DEBUG, GlobalCounters, ansilen, BEAM, NOOPT, all_int, CAPTURING, Metadata, TRACEMETA, TracingKey
from tinygrad.helpers import DEVECTORIZE, time_to_str, VALIDATE_WITH_CPU, cpu_profile, PROFILE, ProfilePointEvent, cpu_events, prod, Context, unwrap
from tinygrad.helpers import EMULATED_DTYPES
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, sym_infer
from tinygrad.device import Device, Buffer
from tinygrad.renderer import ProgramSpec, Estimates
from tinygrad.codegen import get_program

# **************** Runners ****************

class Runner:
  def __init__(self, display_name:str, device:str, estimates=Estimates()):
    self.first_run, self.display_name, self.device, self.estimates = True, display_name, device, estimates
  @property
  def dev(self): return Device[self.device]
  def exec(self, rawbufs:list[Buffer], var_vals:dict[str, int]|None=None) -> float|None:
    return self(rawbufs, {} if var_vals is None else var_vals)
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False) -> float|None:
    raise NotImplementedError("override this")

def optimize_local_size(_prg:Callable, global_size:list[int], rawbufs:list[Buffer]) -> list[int]:
  test_rawbuffers = [Buffer(rawbufs[0].device, rawbufs[0].size, rawbufs[0].dtype).allocate(), *rawbufs[1:]] if rawbufs[0] in rawbufs[1:] else rawbufs
  MAX_WORKGROUP = 1024
  local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in global_size]
  local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
  def try_exec(local_size):
    try:
      return _prg(*[x._buf for x in test_rawbuffers],global_size=[g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)],
                  local_size=local_size, wait=True)
    except Exception: return float('inf')
  ret = min([(try_exec(local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])
  assert not math.isinf(ret[0]), "all optimize_local_size exec failed"
  return ret[1]

class CompiledRunner(Runner):
  def __init__(self, p:ProgramSpec, prg=None):
    if DEBUG >= 3 and p.applied_opts: print(p.applied_opts)
    if DEBUG >= 4: print(p.src)
    if p.lib is None:
      with cpu_profile(TracingKey(f"compile {p.name}", (p.function_name,)), "TINY"):
        p = replace(p, lib=Device[p.device].compiler.compile_cached(p.src))
    self.p:ProgramSpec = p
    assert self.p.lib is not None
    if DEBUG >= 7: Device[p.device].compiler.disassemble(self.p.lib)
    self._prg = Device[p.device].runtime(p.function_name, self.p.lib, *p.aux, runtimevars=p.runtimevars) if prg is None else prg
    super().__init__(p.name, p.device, p.estimates)

  def __reduce__(self): return self.__class__, (self.p,)

  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int]|None=None, wait=False) -> float|None:
    if var_vals is None: var_vals = {}
    global_size, local_size = self.p.launch_dims(var_vals)
    if Device[self.p.device].renderer.has_local and local_size is None and all_int(self.p.global_size):
      local_size = optimize_local_size(self._prg, global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
      self.p = replace(self.p, global_size=global_size, local_size=local_size)
    return self._prg(*[x._buf for x in rawbufs], global_size=tuple(global_size), local_size=tuple(local_size) if local_size else None,
                     vals=tuple(var_vals[k.expr] if k.expr not in self.p.runtimevars else None for k in self.p.vars), wait=wait)

class ViewOp(Runner):
  def __init__(self, buf:Buffer): super().__init__(colored(f"view {buf.nbytes:8d} @ {buf.offset:<10d}", "yellow"), buf.device)
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    assert rawbufs[0]._base is not None and rawbufs[0]._base == rawbufs[1].base, f"must be base {rawbufs}"

class BufferCopy(Runner):
  def __init__(self, total_sz, dest_device, src_device):
    sz = f"{total_sz/1e6:7.2f}M" if total_sz >= 1e6 else f"{total_sz:8d}"
    name = f"{type(self).__name__[6:].lower()} {sz}, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
    super().__init__(colored(name, "yellow"), dest_device, Estimates(lds=total_sz, mem=total_sz))
  def copy(self, dest, src):
    disk_supports_fast_copyout = src.device.startswith("DISK") and hasattr(src.allocator.dev, 'io_uring') and \
      getattr(src.allocator.dev, 'fd', None) is not None and dest.allocator.supports_copy_from_disk
    if disk_supports_fast_copyout and hasattr(dest.allocator, 'copy_from_disk') and src.nbytes >= 4096:
      dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
    elif isinstance(src.device, str) and src.device.startswith(("DISK", "TINYFS")) and hasattr(dest.allocator, '_as_buffer'):
      # fast(ish) path, uses readinto in diskbuffers
      src.allocator._copyout(dest.allocator._as_buffer(dest._buf), src._buf)
    else:
      dest.copyin(src.as_memoryview(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    dest, src = rawbufs[0:2]
    assert dest.size == src.size and dest.dtype == src.dtype, f"buffer copy mismatch, {dest.size} != {src.size}, {dest.dtype} != {src.dtype}"
    st = time.perf_counter()
    self.copy(dest, src)
    if wait:
      Device[dest.device].synchronize()
      return time.perf_counter() - st

class BufferXfer(BufferCopy):
  def copy(self, dest, src): dest.allocator._transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.dev, dest_dev=dest.allocator.dev)

class DiskStore(Runner):
  def __init__(self, display_name:str, device:str, dtype, dest_arg:int, numel:int, dest_offset:int, src_arg:int|None=None, src_offset:int=0, const_val=None):
    self.dtype, self.dest_arg, self.numel, self.dest_offset = dtype, dest_arg, numel, dest_offset
    self.src_arg, self.src_offset, self.const_val = src_arg, src_offset, const_val
    super().__init__(display_name, device, Estimates(lds=numel*dtype.itemsize, mem=numel*dtype.itemsize))

  @staticmethod
  def _parse_linear_index(x:UOp) -> tuple[UOp, int, int]|None:
    if x.op is Ops.CONST: return (x, 1, x.arg)
    if x.op is Ops.RANGE and x.src[0].op is Ops.CONST: return (x, x.src[0].arg, 0)
    if x.op is Ops.ADD:
      if x.src[0].op is Ops.RANGE and x.src[0].src[0].op is Ops.CONST and x.src[1].op is Ops.CONST: return (x.src[0], x.src[0].src[0].arg, x.src[1].arg)
      if x.src[1].op is Ops.RANGE and x.src[1].src[0].op is Ops.CONST and x.src[0].op is Ops.CONST: return (x.src[1], x.src[1].src[0].arg, x.src[0].arg)
    return None

  @staticmethod
  def _parse_index(x:UOp) -> tuple[int, int, int, UOp]|None:
    if x.op is not Ops.INDEX or x.src[0].op is not Ops.PARAM or len(x.src) != 2: return None
    if (parsed:=DiskStore._parse_linear_index(x.src[1])) is None: return None
    rng, numel, offset = parsed
    if not all(isinstance(v, int) for v in (numel, offset)): return None
    return x.src[0].arg, numel, offset, rng

  @staticmethod
  def _const_term(x:UOp) -> int:
    if x.op is Ops.CONST and isinstance(x.arg, int): return x.arg
    if x.op is Ops.ADD and len(x.src) == 2: return DiskStore._const_term(x.src[0]) + DiskStore._const_term(x.src[1])
    return 0

  @staticmethod
  def _parse_index_from_end(x:UOp, end_uop:UOp|None) -> tuple[int, int, int, UOp]|None:
    if x.op is not Ops.INDEX or x.src[0].op is not Ops.PARAM or len(x.src) != 2: return None
    if end_uop is None or end_uop.op is not Ops.END or len(end_uop.src) < 2: return None
    rngs = [r for r in end_uop.src[1:] if r.op is Ops.RANGE and r.src[0].op is Ops.CONST and isinstance(r.src[0].arg, int)]
    if len(rngs) == 0: return None
    return x.src[0].arg, math.prod([r.src[0].arg for r in rngs]), DiskStore._const_term(x.src[1]), rngs[0]

  @staticmethod
  def _parse_bitcast_store_dst(x:UOp, val_dtype, end_uop:UOp|None) -> tuple[int, int, int, UOp]|None:
    if x.op is not Ops.BITCAST or len(x.src) != 1: return None
    if x.src[0].op is not Ops.INDEX or x.src[0].src[0].op is not Ops.PARAM or len(x.src[0].src) != 2: return None
    dst_arg = x.src[0].src[0].arg
    if end_uop is None or end_uop.op is not Ops.END or len(end_uop.src) < 2: return None
    rngs = [r for r in end_uop.src[1:] if r.op is Ops.RANGE and r.src[0].op is Ops.CONST and isinstance(r.src[0].arg, int)]
    if len(rngs) == 0: return None
    numel = math.prod([r.src[0].arg for r in rngs])
    byte_offset = DiskStore._const_term(x.src[0].src[1])
    if val_dtype.itemsize == 0 or byte_offset % val_dtype.itemsize != 0: return None
    return dst_arg, numel, byte_offset // val_dtype.itemsize, rngs[0]

  @staticmethod
  def _parse_src_index(x:UOp, numel:int, rng:UOp) -> tuple[int, int]|None:
    if (src:=DiskStore._parse_index(x)) is not None:
      src_arg, src_numel, src_offset, src_rng = src
      if src_numel == numel and (src_rng is rng or (numel == 1 and src_rng.op is Ops.CONST and rng.op is Ops.CONST)):
        return src_arg, src_offset
    # fallback for source INDEX with nontrivial flatten expression: preserve only constant base offset
    if x.op is Ops.INDEX and x.src[0].op is Ops.PARAM and len(x.src) == 2 and numel >= 1:
      return x.src[0].arg, DiskStore._const_term(x.src[1])
    return None

  @staticmethod
  def _parse_store_sink(sink:UOp):
    if sink.op is not Ops.SINK or len(sink.src) != 1: return None
    x = sink.src[0]
    if x.op is Ops.END and len(x.src) >= 2 and x.src[0].op is Ops.STORE: store = x.src[0]
    elif x.op is Ops.STORE: store = x
    else: return None
    end_uop = x if x.op is Ops.END else None
    if (dst:=DiskStore._parse_index(store.src[0])) is not None: dst_arg, numel, dst_offset, dst_rng = dst
    elif (dst:=DiskStore._parse_index_from_end(store.src[0], end_uop)) is not None: dst_arg, numel, dst_offset, dst_rng = dst
    elif (dst:=DiskStore._parse_bitcast_store_dst(store.src[0], store.src[1].dtype, x if x.op is Ops.END else None)) is not None:
      dst_arg, numel, dst_offset, dst_rng = dst
    else: return None
    val = store.src[1]
    while val.op is Ops.CAST and len(val.src) == 1 and val.src[0].dtype == val.dtype: val = val.src[0]
    if val.op is Ops.CONST: return ("const", store.src[1].dtype, dst_arg, numel, dst_offset, val.arg)
    if val.op is Ops.BITCAST and len(val.src) == 1 and val.src[0].op is Ops.INDEX and val.src[0].src[0].op is Ops.PARAM:
      src_idx = val.src[0]
      src_itemsize = src_idx.src[0].dtype.base.itemsize
      src_offset_bytes = DiskStore._const_term(src_idx.src[1]) * src_itemsize
      if src_offset_bytes % store.src[1].dtype.itemsize == 0:
        return ("copy", store.src[1].dtype, dst_arg, numel, dst_offset, src_idx.src[0].arg, src_offset_bytes // store.src[1].dtype.itemsize)
    src_val = val.src[0] if val.op is Ops.BITCAST and len(val.src) == 1 and val.src[0].op is Ops.INDEX else val
    if (src:=DiskStore._parse_src_index(src_val, numel, dst_rng)) is not None:
      src_arg, src_offset = src
      return ("copy", store.src[1].dtype, dst_arg, numel, dst_offset, src_arg, src_offset)
    return None

  @staticmethod
  def from_sink(sink:UOp, bufs:list[Buffer|None]) -> 'DiskStore|None':
    if (parsed:=DiskStore._parse_store_sink(sink)) is None: return None
    if parsed[0] == "const":
      _, dtype, dst_arg, numel, dst_offset, const_val = parsed
      if dtype.fmt is None: return None
      return DiskStore(f"disk_store_const {numel*dtype.itemsize:8d}", cast(Buffer, bufs[dst_arg]).device, dtype, dst_arg, numel, dst_offset,
                       const_val=const_val)
    _, dtype, dst_arg, numel, dst_offset, src_arg, src_offset = parsed
    return DiskStore(f"disk_store_copy {numel*dtype.itemsize:8d}", cast(Buffer, bufs[dst_arg]).device, dtype, dst_arg, numel, dst_offset,
                     src_arg=src_arg, src_offset=src_offset)

  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    dest = rawbufs[self.dest_arg].view(self.numel, self.dtype, self.dest_offset*self.dtype.itemsize).ensure_allocated()
    st = time.perf_counter()
    if self.src_arg is None:
      data = struct.pack(f"{self.numel}{cast(str, self.dtype.fmt)}", *([self.const_val] * self.numel))
      dest.copyin(memoryview(data))
    else:
      src_buf = rawbufs[self.src_arg]
      max_elems = src_buf.nbytes // self.dtype.itemsize
      src_offset = self.src_offset
      if src_offset >= max_elems or src_offset + self.numel > max_elems:
        src_offset = 0 if self.numel <= max_elems else max(0, max_elems-self.numel)
      src = src_buf.view(self.numel, self.dtype, src_offset*self.dtype.itemsize).ensure_allocated()
      BufferCopy(dest.nbytes, dest.device, src.device).copy(dest, src)
    if wait:
      Device[dest.device].synchronize()
      return time.perf_counter() - st

class EncDec(Runner):
  def __init__(self, encdec:UOp, total_sz:int, device:str):
    self.shape, self.pos_var = encdec.arg[0], encdec.variables()[0].expr
    name = f"enc/dec {total_sz/1e6:7.2f}M, HEVC" if total_sz >= 1e6 else f"enc/dec {total_sz:8d}, HEVC"
    super().__init__(colored(name, "yellow"), device, Estimates(lds=total_sz, mem=total_sz))
  def __call__(self, rawbufs:list[Buffer], var_vals:dict[str, int], wait=False):
    st = time.perf_counter()
    rawbufs[0].allocator._encode_decode(rawbufs[0]._buf, rawbufs[1]._buf, rawbufs[2]._buf,
                                        [x._buf for x in rawbufs[3:]], self.shape, var_vals[self.pos_var])
    if wait:
      Device[rawbufs[0].device].synchronize()
      return time.perf_counter() - st

# **************** method cache ****************

method_cache: dict[tuple[str, type, bytes, tuple, bool], CompiledRunner] = {}
def get_runner(device:str, ast:UOp) -> CompiledRunner:
  # TODO: this should be all context relevant to rendering
  context = (BEAM.value, NOOPT.value, DEVECTORIZE.value, EMULATED_DTYPES.value)
  ckey = (device, type(Device[device].compiler), ast.key, context, False)
  if cret:=method_cache.get(ckey): return cret
  bkey = (device.split(":")[0], type(Device[device].compiler), ast.key, context, True)
  if bret:=method_cache.get(bkey):
    method_cache[ckey] = ret = CompiledRunner(replace(bret.p, device=device))
  else:
    prg: ProgramSpec = get_program(ast, Device[device].renderer)
    method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, device=device))
  return ret

def lower_sink(ctx:list[Buffer|None], sink:UOp):
  if isinstance(device:=ctx[0].device, str) and device.startswith(("DISK", "TINYFS")):
    if (ret:=DiskStore.from_sink(sink, ctx)) is not None: return ret
  return get_runner(device, sink)

# **************** lowering functions ****************

# NOTE: ctx is the buffers
si_lowerer = PatternMatcher([
  (UPat((Ops.SINK, Ops.PROGRAM), name="sink"), lower_sink),
  (UPat(Ops.BUFFER_VIEW), lambda ctx: ViewOp(ctx[0])),
  (UPat(Ops.COPY, name="copy"), lambda ctx,copy: (BufferXfer(ctx[0].nbytes, ctx[0].device, ctx[1].device) \
      if hasattr(alc:=Device[ctx[0].device].allocator, '_transfer') and alc.supports_transfer and all_same([x.device.split(":")[0] for x in ctx]) \
      else BufferCopy(ctx[0].nbytes, ctx[0].device, ctx[1].device))),
  (UPat(Ops.ENCDEC, name="encdec"), lambda ctx,encdec: EncDec(encdec, ctx[0].nbytes, ctx[1].device)),
])

@dataclass
class ExecItem:
  ast: UOp
  bufs: list[Buffer|None] = field(default_factory=list)
  metadata: tuple[Metadata, ...] = ()
  fixedvars: dict[str, int] = field(default_factory=dict)
  prg: Runner|None = None

  def lower(self):
    """Populate self.prg by lowering the AST."""
    if self.prg is not None: return self
    try: self.prg = cast(Runner, si_lowerer.rewrite(self.ast, self.bufs))
    except Exception as e:
      if DEBUG >= 2:
        print(f"error lowering {self.ast.op}")
        print("tensor operations:")
        pprint.pprint(self.metadata, indent=2)
      raise e
    return self

  def run(self, _var_vals:dict[str, int]|None=None, wait=False, jit=False, do_update_stats=True) -> float|None:
    if self.prg is None: self.lower()
    assert self.prg is not None
    var_vals = self.fixedvars if _var_vals is None else (_var_vals|self.fixedvars)
    # reorder bufs to match program globals if needed
    _bufs = [self.bufs[i] for i in self.prg.p.globals] if isinstance(self.prg, CompiledRunner) else self.bufs
    bufs = [unwrap(x) for x in _bufs] if jit else [unwrap(x).ensure_allocated() for x in _bufs]
    if PROFILE:
      payload = {"metadata":self.metadata, "var_vals":var_vals, "bufs":[b.trace_num for b in bufs], "name":self.prg.display_name}
      payload["outputs"], payload["inputs"] = (self.prg.p.outs, self.prg.p.ins) if isinstance(self.prg, CompiledRunner) else ([0], [1])
      cpu_events.append(ProfilePointEvent(self.prg.device, "exec", len(cpu_events), payload))
    et = self.prg(bufs, var_vals, wait=wait or DEBUG >= 2)
    if do_update_stats:
      GlobalCounters.kernel_count += 1
      GlobalCounters.global_ops += (op_est:=sym_infer(self.prg.estimates.ops, var_vals))
      GlobalCounters.global_mem += (mem_est:=sym_infer(self.prg.estimates.mem, var_vals))
      if et is not None: GlobalCounters.time_sum_s += et
      if DEBUG >= 2:
        lds_est = sym_infer(self.prg.estimates.lds, var_vals)
        header_color = 'magenta' if jit else ('green' if self.prg.first_run else None)
        ptm = colored(time_to_str(et, w=9), "yellow" if et > 0.01 else None) if et is not None else ""
        flops, membw, ldsbw = op_est/(et or 1e-20), mem_est/(et or 1e-20), lds_est/(et or 1e-20)
        flops_str = f"{flops*1e-9:7.0f} GFLOPS" if flops < 1e14 else colored(f"{flops*1e-12:7.0f} TFLOPS", 'green')
        mem_str = f"{membw*1e-9:4.0f}|{ldsbw*1e-9:<6.0f} GB/s" if membw < 1e13 and ldsbw < 1e15 else \
          colored(f"{membw*1e-12:4.0f}|{ldsbw*1e-12:<6.0f} TB/s", 'green')
        print(f"{colored(f'*** {self.prg.device[:7]:7s} {GlobalCounters.kernel_count:4d}', header_color)}"+
          f" {self.prg.display_name+' '*(46-ansilen(self.prg.display_name))} arg {len(bufs):2d} mem {GlobalCounters.mem_used/1e9:6.2f} GB"+
          ("" if et is None else f" tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({flops_str} {mem_str})")+
          f" {[repr(m) if TRACEMETA >= 2 else str(m) for m in self.metadata] if self.metadata else ''}")
      self.prg.first_run = False
    return et

# **************** main run function ****************

capturing: list = []  # put classes with an add method in here

def run_schedule(schedule:list[ExecItem], var_vals:dict[str, int]|None=None, do_update_stats=True):
  # DISK devices can't be reopened to larger mappings later, so pre-open the largest buffer per device up front.
  max_disk_buf: dict[str, Buffer] = {}
  for ei in schedule:
    for b in ei.bufs:
      if b is None or not isinstance(b.device, str) or not b.device.startswith(("DISK", "TINYFS")): continue
      if (cur:=max_disk_buf.get(b.device)) is None or b.nbytes > cur.nbytes: max_disk_buf[b.device] = b
  for b in max_disk_buf.values(): b.ensure_allocated()
  while len(schedule):
    ei = schedule.pop(0).lower()
    if len(capturing) and CAPTURING: capturing[0].add(ei)
    if VALIDATE_WITH_CPU and ei.ast.op is Ops.SINK:
      # copy in allocated buffers from the GPU
      bufs = [b for b in ei.bufs if b is not None]
      nb: list[Buffer|None] = [Buffer("CPU", b.size, b.dtype) for b in bufs]
      for cpu_b, gpu_b in zip(nb, bufs):
        if cpu_b is not None and gpu_b.is_allocated(): cpu_b.ensure_allocated().copyin(gpu_b.as_memoryview())

      # run on GPU
      ei.run(var_vals, do_update_stats=do_update_stats)

      # validate the output buffers match (NOTE: this is assuming the output is buffer 0)
      with Context(BEAM=0): ExecItem(ei.ast, nb, ei.metadata, ei.fixedvars).run(var_vals, do_update_stats=do_update_stats)
      import numpy as np
      assert nb[0] is not None
      np.testing.assert_allclose(bufs[0].numpy(), nb[0].numpy(), rtol=1e-3, atol=1e-3)
    else:
      ei.run(var_vals, do_update_stats=do_update_stats)
