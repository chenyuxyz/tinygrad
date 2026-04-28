from __future__ import annotations
from typing import Self
from tinygrad.dtype import dtypes
from tinygrad.helpers import ceildiv

class RandMixin:
  @staticmethod
  def _threefry_random_bits(key, counts0, counts1):
    x = (counts1.cast(dtypes.uint64) << 32) | counts0.cast(dtypes.uint64)
    x = x.threefry((key[1]._broadcast_to(x.shape).cast(dtypes.uint64) << 32) | key[0]._broadcast_to(x.shape).cast(dtypes.uint64))
    return (x & 0xffffffff).cast(dtypes.uint32).cat(((x >> 32) & 0xffffffff).cast(dtypes.uint32))

  @classmethod
  def random_bits(cls, key:Self, counter:Self, num:int) -> Self:
    low, high = counter[0:1], counter[1:2]  # type: ignore[index]
    bits = []
    for i in range(0, num, dtypes.uint32.max):
      chunk_num = min(num - i, dtypes.uint32.max)
      c_low = low + (i & 0xffffffff)
      c_high = high + (i >> 32) + (c_low < low).cast(dtypes.uint32)
      new_key = cls._threefry_random_bits(key, c_low, c_high)
      counts0 = cls.arange(ceildiv(chunk_num, 2), device=key.device, dtype=dtypes.uint32)  # type: ignore[attr-defined]
      counts1 = counts0 + ceildiv(chunk_num, 2)
      bits.append(cls._threefry_random_bits(new_key, counts0, counts1)[:chunk_num])
    return bits[0].cat(*bits[1:])
