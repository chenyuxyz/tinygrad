# Design: Proper DISK Assign (Removing the Hack)

## Problem

`tensor.py:assign` has a hack for DISK targets that bypasses the schedule entirely:

```python
if is_disk:
    self._buffer().copyin(x._data())
    return self
```

This directly realizes `x`, gets its raw bytes, and copies them to the DISK buffer. While it works, it:
- Bypasses the schedule (can't be batched/optimized)
- Doesn't compose with JIT capture
- Is a special case that doesn't go through the graph rewrite system

**Goal**: Remove this hack and use proper ASSIGN + rewrite rules so the copy from compute device to disk goes through the normal schedule as a COPY ExecItem.

## Current Infrastructure

### How COPY-in-ASSIGN Already Works (for same-device)

The schedule already supports ASSIGN with COPY/BUFFER_VIEW sources. The flow:

1. **`realize_assign_src`** (`schedule/indexing.py:20`): When ASSIGN source is COPY/BUFFER_VIEW/ENCDEC, it's **unrealized** (removed from realize_map). This means the COPY stays as the ASSIGN source value instead of being independently bufferized.

2. **`bufferize_to_store`** (`schedule/rangeify.py:325`): Creates `INDEX(BUFFER, ...).store(assign_src).end(ranges)`, wrapping the assign target buffer in AFTER.

3. **`split_store`** (`schedule/rangeify.py:477-481`): Detects when the stored value is COPY or BUFFER_VIEW:
   ```python
   if stored.op in {Ops.COPY, Ops.BUFFER_VIEW}:
       ret = stored.replace(src=stored.src + ret.ended_ranges)
   ```
   This converts the kernel from a SINK (compute kernel) to a COPY/BUFFER_VIEW ExecItem.

4. The COPY ExecItem's **output buffer** comes from the AFTER structure — it's the ASSIGN target buffer (the existing DISK buffer).

### How DISK COPY Already Works (for `tensor.to("disk:...")`)

- **`disk_copy_is_buffer`** (`engine/allocations.py:19`): For COPY-to-disk, creates a new DISK buffer in `buffer_map`.
- **`pm_finalize_call`** (`engine/allocations.py:162`): COPY-to-disk UOps are appended to the assigns list.
- **`BufferCopy`** (`engine/realize.py`): At execution time, uses `copyin` or optimized paths (io_uring, readinto) to transfer data.

### Existing ASSIGN Rewrite Rules (`schedule/rangeify.py:125-138`)

- **Collapse nested ASSIGN**: `ASSIGN(target, ASSIGN(target, src))` → `ASSIGN(target, src)`
- **Move bitcast to source**: `ASSIGN(BITCAST(target), src)` → `ASSIGN(target, src.bitcast(target.dtype))`
- **Normalize ASSIGN chains**: unwrap chained ASSIGN targets
- **Fix hazards**: make source contiguous if it contains hazardous movement ops on dest

### Free Bitcast for DISK

DISK supports free bitcast through `late_buffer_view` (`schedule/rangeify.py:266`):
- `BUFFERIZE(BITCAST(x))` on DISK → `BUFFER_VIEW(base, (size, offset))`
- BUFFER_VIEW is zero-copy: just a different dtype interpretation at an offset
- Used by `safe_save` for writing header length: `t[0:8].bitcast(dtypes.int64).assign([len(j)])`

## Experiments

### What Works Now (without the hack)

**Full assign with buffer source** — PASS:
```python
dt = Tensor.empty(4, device="disk:...", dtype=dtypes.int32)
src = Tensor([10,20,30,40], dtype=dtypes.int32)
# Manually create ASSIGN(DISK_BUF, COPY(src, DISK))
dt.uop = dt.uop.assign(src.uop.copy_to_device(dt.device))
dt.realize()  # Works! Data correctly written to disk.
```

The schedule produces 2 COPYs:
1. COPY from PYTHON → METAL (realize the list)
2. COPY from METAL → DISK (write to disk)

The DISK buffer in the schedule IS the existing tensor's buffer (`sched[1].buf[0] is dt.uop.buffer` → True).

**Full assign with CONST source (via contiguous)** — PASS:
```python
dt = Tensor.empty(4, device="disk:...", dtype=dtypes.int32)
src = Tensor.full((4,), 42, dtype=dtypes.int32)
# contiguous() prevents early_fixup_const_copy from optimizing COPY(CONST,DISK) → CONST(DISK)
dt.uop = dt.uop.assign(src.uop.contiguous().copy_to_device(dt.device))
dt.realize()  # Works!
```

### What Fails Now

**Bare ASSIGN(DISK_BUF, src)** — FAIL:
```python
dt.assign(Tensor.full((4,), 42)).realize()
# NotImplementedError: needs a renderer
```
DISK has no renderer, so kernels can't execute on it.

**ASSIGN with CONST source without contiguous** — FAIL:
```python
# COPY(CONST(42, METAL), DISK) is optimized to CONST(42, DISK) by early_fixup_const_copy
# Then ASSIGN(DISK_BUF, CONST(42, DISK)) creates kernel on DISK → fails
```

**Slice assign with COPY** — WRONG OFFSET:
```python
dt[2:5].assign(...)  # Writes to offset 0 instead of offset 2
```
The COPY creates a new buffer at offset 0. The slice offset info from the ASSIGN target (SHRINK) is lost when the kernel becomes a COPY.

## Proposed Design

### Where to Make the Change

**In `earliest_rewrites`** (`schedule/rangeify.py`), add a rule that converts DISK ASSIGN sources to COPY. This runs inside `get_kernel_graph` (after `transform_to_call`), so:
- It's after `add_tags` (no interference with `disk_copy_is_buffer`)
- It's after `pm_early_transform_tensor_graph` (bitcast rules have already fired)
- PARAMs already have `_device` set, so we can check if target is DISK
- The COPY won't be processed by `pm_finalize_call`'s standalone COPY-to-disk rule (that only runs in `transform_to_call`)

### The Rule

```python
# In earliest_rewrites (schedule/rangeify.py)
def disk_assign_wrap_copy(assign:UOp):
    """For DISK assigns, wrap the source in a COPY so it becomes a COPY ExecItem instead of a kernel."""
    target = assign.src[0]
    # Walk through ASSIGN/BITCAST/AFTER to find the base buffer
    base = target
    while base.op in {Ops.ASSIGN, Ops.BITCAST, Ops.AFTER}: base = base.src[0].base
    if base.op not in {Ops.BUFFER, Ops.PARAM}: return None
    device = base._device
    if not (isinstance(device, str) and device.startswith("DISK")): return None
    src = assign.src[1]
    # If source is already a COPY to this device, no change needed
    if src.op is Ops.COPY and src._device == device: return None
    # Wrap source in COPY to disk
    return assign.replace(src=(target, src.copy_to_device(device)))

(UPat(Ops.ASSIGN, name="assign"), disk_assign_wrap_copy),
```

### Why This Works

After the rule fires, the graph is:
```
ASSIGN(INDEX(DISK_BUF, offset...), COPY(src_indexed, DISK_DEVICE))
```

1. **`realize_assign_src`** unrealizes the COPY (doesn't get its own buffer)
2. **`bufferize_to_store`** creates: `DISK_BUF.after(INDEX(...).store(COPY(src, DISK)).end(ranges))`
3. **`split_store`** sees `stored.op is Ops.COPY` → converts to COPY kernel
4. The COPY kernel's **output buffer = existing DISK_BUF** (from the AFTER structure)
5. The COPY kernel's **input buffer = src buffer** (on compute device)
6. The **offset is preserved** in the INDEX/ranges, which get passed to the COPY

### Slice Assign Flow

```
dt[2:5].assign(Tensor([99,99,99]))
```

1. Tensor graph: `ASSIGN(SHRINK(DISK_BUF, 2, 5), CONST(99))`
2. After rangeify: `ASSIGN(INDEX(DISK_BUF, range(2,5)), src_indexed)`
3. Our rule: `ASSIGN(INDEX(DISK_BUF, range(2,5)), COPY(src_indexed, DISK))`
4. bufferize_to_store: `DISK_BUF.after(INDEX(DISK_BUF, range(2,5)).store(COPY(src, DISK)).end(range))`
5. split_store: COPY kernel with output = DISK_BUF
6. BufferCopy writes data to the DISK buffer — the **offset is handled by the INDEX**

**Key**: The INDEX(DISK_BUF, offset_range) preserves the slice offset. The COPY writes to the correct region of the DISK buffer because the AFTER targets the full DISK_BUF, and the INDEX+ranges encode where within the buffer to write.

### Bitcast Assign Flow

```
t[0:8].bitcast(dtypes.int64).assign([12345])
```

1. Existing bitcast rule fires first: `ASSIGN(BITCAST(target), src)` → `ASSIGN(target, src.bitcast(target.dtype))`
   - Result: `ASSIGN(SHRINK(DISK_BUF, 0, 8), BITCAST(CONST(12345, int64), uint8))`
2. After rangeify, the BITCAST on the source becomes part of the indexed expression
3. Our rule wraps in COPY: `ASSIGN(INDEX(DISK_BUF, ...), COPY(src_with_bitcast, DISK))`
4. split_store → COPY kernel
5. BufferCopy copies the raw bytes to the correct offset in the DISK buffer

**No compute kernel needed for the bitcast** — the bitcast is just a reinterpretation of bytes. The COPY transfers raw bytes regardless of dtype.

### CONST Source Handling

For CONST sources (e.g., `Tensor.full`):
- The COPY source is the CONST expression
- The CONST gets bufferized normally (materialized on the compute device)
- Then the COPY transfers from compute device to DISK

**Important**: The `early_fixup_const_copy` rule (`pm_early_transform_tensor_graph:137`) runs in `transform_to_call` BEFORE our rule. It converts `COPY(CONST, device)` → `CONST(device)`. Since our COPY is created inside `get_kernel_graph` (AFTER `transform_to_call`), this rule doesn't interfere.

### Changes to `tensor.py:assign`

Remove the commented-out hack. No other changes needed — the ASSIGN is created normally, and our rewrite rule handles DISK.

One consideration: the current code relaxes device/dtype checks for DISK:
```python
if not is_disk and self.device != x.device: raise RuntimeError(...)
if not is_disk and self.dtype != x.dtype: raise RuntimeError(...)
```
These relaxations should stay — DISK assign allows cross-device sources and (via bitcast) different dtypes.

### Changes to `engine/allocations.py`

`disk_copy_is_buffer` may need adjustment: currently it creates a buffer_map entry for ALL COPY-to-disk UOps. For COPYs created inside `get_kernel_graph`, there's no interference (they don't exist in the tensor graph). But if a COPY-to-disk appears in the tensor graph (created at the tensor level), the buffer_map entry could cause issues in `linear_to_schedule` (accessing `.buffer` on a COPY UOp fails).

**Fix**: Either don't create buffer_map entries for COPYs that are inside ASSIGNs, or skip `.buffer` access for non-buffer UOps in `linear_to_schedule`.

## Key Files to Modify

| File | Change |
|------|--------|
| `tinygrad/tensor.py` | Remove the DISK assign hack (already commented out) |
| `tinygrad/schedule/rangeify.py` | Add `disk_assign_wrap_copy` rule to `earliest_rewrites` |

## Testing

Run the existing disk tests:
```bash
python -m pytest test/unit/test_disk_tensor.py -xvs
```

Key tests to verify:
- `test_assign_const_to_disk` — CONST source
- `test_assign_slice_from_const` — sliced CONST source
- `test_assign_disk_to_disk` — disk-to-disk via CPU
- `test_assign_slice` — slice assign
- `test_assign_to_different_dtype` — cross-dtype assign
- `test_assign_with_bitcast` — bitcast + assign (used by safe_save)
- `test_assign_to_bitcast_view` — assign to bitcast view
- `test_assign_cross_device` — cross-device assign

Also test safe_save (the primary consumer):
```bash
python -m pytest test/unit/test_disk_tensor.py::TestSafetensors -xvs
```
