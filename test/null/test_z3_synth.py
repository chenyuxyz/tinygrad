#!/usr/bin/env python
"""z3-based synthesis for discovering index simplifications.

Instead of heuristic pattern matching, uses z3 to:
1. Prove subexpressions are constant under validity constraints
2. Drop redundant validity clauses
3. Synthesize simpler linear expressions via sampling + verification
4. Resynthesize validity conditions as simpler boolean expressions
"""
import unittest
from fractions import Fraction
from tinygrad.uop.ops import UOp, Ops
from tinygrad.uop.validate import uops_to_z3
from tinygrad.uop.symbolic import uop_given_valid
from tinygrad.dtype import dtypes
import z3

# ===== helpers (same as test_simplify_valid_idx) =====

def Special(expr, nmax): return UOp(Ops.SPECIAL, dtypes.index, (UOp.const(dtypes.index, nmax),), expr)
def Variable(expr, nmin, nmax): return UOp.variable(expr, nmin, nmax)
def Range(n, nmax): return UOp.range(nmax, n)

# ===== z3 synthesis core =====

def z3_is_equiv(a:UOp, b:UOp, under:UOp|None=None) -> bool:
  """Check if a == b for all valid variable assignments (optionally under constraint)."""
  solver = z3.Solver(ctx=z3.Context())
  args = [a, b] + ([under] if under is not None else [])
  z3_results = uops_to_z3(solver, *args)
  if under is not None: solver.add(z3_results[2])
  return solver.check(z3_results[0] != z3_results[1]) == z3.unsat

def z3_is_unsat(valid:UOp) -> bool:
  """Check if valid is unsatisfiable (always False)."""
  solver = z3.Solver(ctx=z3.Context())
  z3_valid, = uops_to_z3(solver, valid)
  solver.add(z3_valid)
  return solver.check() == z3.unsat

def z3_constify(expr:UOp, under:UOp) -> UOp:
  """Replace subexpressions of expr that are provably constant when `under` is true."""
  all_nodes = list(expr.toposort())
  # include variables — they might be forced to a single value
  subexprs = [u for u in all_nodes if u.op != Ops.CONST and u.dtype.scalar() in (dtypes.index, dtypes.int, dtypes.bool)]
  if not subexprs: return expr

  solver = z3.Solver(ctx=z3.Context())
  z3_all = uops_to_z3(solver, under, *subexprs)
  z3_under, z3_subs = z3_all[0], z3_all[1:]
  solver.add(z3_under)

  replacements: dict[UOp, UOp] = {}
  for uop, z3_expr in zip(subexprs, z3_subs):
    if any(p in replacements for p in uop.toposort() if p is not uop): continue  # parent will handle
    if uop.dtype == dtypes.bool:
      for val in [True, False]:
        if solver.check(z3_expr != z3.BoolVal(val, ctx=solver.ctx)) == z3.unsat:
          replacements[uop] = UOp.const(dtypes.bool, val); break
    else:
      # get a concrete value from z3 model, then check if it's the only one
      if solver.check() == z3.sat:
        val = solver.model().eval(z3_expr, model_completion=True).as_long()
        if solver.check(z3_expr != val) == z3.unsat:
          replacements[uop] = UOp.const(uop.dtype, val)

  return expr.substitute(replacements).simplify() if replacements else expr

def z3_tighten_bounds(expr:UOp, under:UOp) -> UOp:
  """Tighten bounds of subexpressions using z3, then re-simplify via fake variable substitution."""
  all_nodes = list(expr.toposort())
  subexprs = [u for u in all_nodes if u.op not in (Ops.CONST, Ops.SPECIAL, Ops.RANGE, Ops.DEFINE_VAR)
              and u.dtype.scalar() in (dtypes.index, dtypes.int)]
  if not subexprs: return expr

  solver = z3.Solver(ctx=z3.Context())
  z3_all = uops_to_z3(solver, under, *subexprs)
  z3_under, z3_subs = z3_all[0], z3_all[1:]
  solver.add(z3_under)

  # for each subexpr, find tighter [min, max] using z3 optimization
  subs: dict[UOp, UOp] = {}
  for uop, z3_expr in zip(subexprs, z3_subs):
    old_min, old_max = int(uop.vmin), int(uop.vmax)
    # binary search for tighter min
    lo, hi = old_min, old_max
    while lo < hi:
      mid = (lo + hi) // 2
      if solver.check(z3_expr < lo + 1) == z3.unsat: lo = lo + 1; break
      if solver.check(z3_expr <= mid) == z3.unsat: lo = mid + 1
      else: hi = mid
    new_min = lo
    # binary search for tighter max
    lo, hi = new_min, old_max
    while lo < hi:
      mid = (lo + hi + 1) // 2
      if solver.check(z3_expr > hi - 1) == z3.unsat: hi = hi - 1; break
      if solver.check(z3_expr >= mid) == z3.unsat: hi = mid - 1
      else: lo = mid
    new_max = hi
    if new_min > old_min or new_max < old_max:
      subs[uop] = UOp.variable(f"_tight{len(subs)}", new_min, new_max, uop.dtype)

  if not subs: return expr
  result = expr.substitute(subs).simplify()
  # substitute back
  return result.substitute({v: k for k, v in subs.items()}).simplify()

def z3_simplify_valid(valid:UOp) -> UOp:
  """Simplify valid: constify within clauses, drop redundant clauses."""
  if z3_is_unsat(valid): return UOp.const(dtypes.bool, False)
  clauses = list(valid.split_uop(Ops.AND))
  if len(clauses) <= 1: return valid

  # iteratively constify each clause given the others
  for _ in range(3):
    changed = False
    for i in range(len(clauses)):
      others = [c for j, c in enumerate(clauses) if j != i]
      given = others[0]
      for o in others[1:]: given = given & o
      new_clause = z3_constify(clauses[i], given)
      if new_clause is not clauses[i]: clauses[i] = new_clause; changed = True
    if not changed: break

  # drop constant True clauses, check for False
  kept: list[UOp] = []
  for c in clauses:
    if c.op is Ops.CONST and c.arg is True: continue
    if c.op is Ops.CONST and c.arg is False: return UOp.const(dtypes.bool, False)
    kept.append(c)
  if not kept: return UOp.const(dtypes.bool, True)

  # drop clauses implied by the others
  final: list[UOp] = []
  for i, clause in enumerate(kept):
    others = [kept[j] for j in range(len(kept)) if j != i and j in {k for k,_ in enumerate(kept)} and (j < i and j in [f for f,_ in enumerate(final)] or j > i)]
    # simpler: check if removing this clause still gives equivalent valid
    without = [kept[j] for j in range(len(kept)) if j != i]
    if without:
      without_valid = without[0]
      for o in without[1:]: without_valid = without_valid & o
      if z3_is_equiv(clause, UOp.const(dtypes.bool, True), under=without_valid): continue
    final.append(clause)

  if not final: return UOp.const(dtypes.bool, True)
  result = final[0]
  for f in final[1:]: result = result & f
  return result

def _solve_int_linear(A:list[list[Fraction]], b:list[Fraction]) -> list[int]|None:
  """Solve Ax=b for integer x via Gaussian elimination with exact fractions."""
  n, m = len(A), len(A[0])
  aug = [row + [bi] for row, bi in zip(A, b)]
  for col in range(m):
    pivot = next((row for row in range(col, n) if aug[row][col] != 0), None)
    if pivot is None: return None
    aug[col], aug[pivot] = aug[pivot], aug[col]
    for row in range(n):
      if row != col and aug[row][col] != 0:
        factor = aug[row][col] / aug[col][col]
        for j in range(m + 1): aug[row][j] -= factor * aug[col][j]
  result = []
  for i in range(m):
    if aug[i][i] == 0: return None
    val = aug[i][m] / aug[i][i]
    if val.denominator != 1: return None
    result.append(int(val))
  return result

def z3_linear_synth(idx:UOp, valid:UOp) -> UOp|None:
  """Try to find a linear combination of variables equivalent to idx under valid."""
  idx_vars = sorted([u for u in idx.toposort() if u.op in (Ops.SPECIAL, Ops.RANGE, Ops.DEFINE_VAR)], key=lambda u: u.render())
  if not idx_vars: return None
  n = len(idx_vars)

  # sample n+2 valid points using z3
  solver = z3.Solver(ctx=z3.Context())
  z3_all = uops_to_z3(solver, valid, idx, *idx_vars)
  z3_valid, z3_idx, z3_vars = z3_all[0], z3_all[1], z3_all[2:]
  solver.add(z3_valid)

  samples: list[tuple[list[int], int]] = []
  for _ in range(n + 2):
    if solver.check() != z3.sat: break
    model = solver.model()
    var_vals = [model.eval(zv, model_completion=True).as_long() for zv in z3_vars]
    idx_val = model.eval(z3_idx, model_completion=True).as_long()
    samples.append((var_vals, idx_val))
    solver.add(z3.Or(*[zv != model.eval(zv, model_completion=True) for zv in z3_vars]))

  if len(samples) < n + 1: return None

  # solve: idx_val = c0 + c1*v1 + ... + cn*vn
  A = [[Fraction(1)] + [Fraction(v) for v in vv] for vv, _ in samples[:n+1]]
  b = [Fraction(iv) for _, iv in samples[:n+1]]
  coeffs = _solve_int_linear(A, b)
  if coeffs is None: return None

  # quick check on extra samples
  for vv, iv in samples[n+1:]:
    if coeffs[0] + sum(c*v for c,v in zip(coeffs[1:], vv)) != iv: return None

  # build UOp
  terms: list[UOp] = []
  if coeffs[0] != 0: terms.append(UOp.const(dtypes.index, coeffs[0]))
  for v, c in zip(idx_vars, coeffs[1:]):
    if c == 0: continue
    terms.append(v if c == 1 else v * c)
  result = terms[0] if terms else UOp.const(dtypes.index, 0)
  for t in terms[1:]: result = result + t
  result = result.simplify()

  # verify with z3
  return result if z3_is_equiv(idx, result, under=valid) else None

def z3_resynth_valid(valid:UOp) -> UOp:
  """Try to find a simpler boolean expression equivalent to valid."""
  all_vars = sorted([u for u in valid.toposort() if u.op in (Ops.SPECIAL, Ops.RANGE, Ops.DEFINE_VAR)], key=lambda u: u.render())

  def _candidates_single():
    for v in all_vars:
      lo, hi = int(v.vmin), int(v.vmax)
      for c in range(lo + 1, hi + 2):
        yield v < c
        yield (v < c).ne(True)

  def _candidates_pair():
    for i, v1 in enumerate(all_vars):
      lo1, hi1 = int(v1.vmin), int(v1.vmax)
      for j, v2 in enumerate(all_vars):
        if j <= i: continue
        lo2, hi2 = int(v2.vmin), int(v2.vmax)
        for c1 in range(lo1 + 1, hi1 + 2):
          for c2 in range(lo2 + 1, hi2 + 2):
            for cand1 in [v1 < c1, (v1 < c1).ne(True)]:
              for cand2 in [v2 < c2, (v2 < c2).ne(True)]:
                yield cand1 & cand2

  # try single conditions first (simpler)
  for cand in _candidates_single():
    if z3_is_equiv(valid, cand): return cand
  # then pairs
  for cand in _candidates_pair():
    if z3_is_equiv(valid, cand): return cand
  return valid

def z3_simplify(idx:UOp, valid:UOp) -> tuple[UOp, UOp]:
  """Full z3-based simplification of (idx, valid) pair."""
  # phase 1: simplify valid
  new_valid = z3_simplify_valid(valid)

  # phase 2: resynthesize valid if still complex (has div/mod)
  if new_valid.op is not Ops.CONST and any(u.op in (Ops.IDIV, Ops.MOD) for u in new_valid.toposort()):
    resynth = z3_resynth_valid(new_valid)
    if resynth is not new_valid: new_valid = resynth

  # phase 3: use tinygrad's uop_given_valid for bound-based simplification (handles div/mod folding)
  new_idx = uop_given_valid(new_valid, idx)

  # phase 4: z3 constification (catches what uop_given_valid misses)
  new_idx = z3_constify(new_idx, new_valid)

  # phase 5: linear synthesis on the simplified idx
  linear = z3_linear_synth(new_idx, new_valid)
  if linear is not None: new_idx = linear

  return new_idx, new_valid

def _z3_clause_droppable(clause:UOp, idx0:UOp, idx1:UOp, width:int, height:int) -> bool:
  """Check if an image valid clause can be dropped: when clause is False, is some idx component OOB?"""
  # build NOT(clause) as a UOp: negate the clause
  neg_clause = clause.logical_not()
  # when clause is false, check if idx0 or idx1 goes out of bounds
  solver = z3.Solver(ctx=z3.Context())
  z3_all = uops_to_z3(solver, neg_clause, idx0, idx1)
  z3_neg, z3_i0, z3_i1 = z3_all
  solver.add(z3_neg)
  # if under NOT clause, idx0 is always OOB or idx1 is always OOB → clause is droppable
  i0_oob = solver.check(z3.And(z3_i0 >= 0, z3_i0 < width)) == z3.unsat
  i1_oob = solver.check(z3.And(z3_i1 >= 0, z3_i1 < height)) == z3.unsat
  return i0_oob or i1_oob

def z3_simplify_image(idx0:UOp, idx1:UOp, valid:UOp, shape:tuple[int,...]) -> tuple[UOp, UOp, UOp|None]:
  """Simplify image load: (idx0, idx1, valid). Returns (new_idx0, new_idx1, new_valid_or_None)."""
  height, width = shape[0], shape[1]

  # simplify valid
  new_valid = z3_simplify_valid(valid)
  if new_valid.op is not Ops.CONST and any(u.op in (Ops.IDIV, Ops.MOD) for u in new_valid.toposort()):
    resynth = z3_resynth_valid(new_valid)
    if resynth is not new_valid: new_valid = resynth

  # simplify idx components: uop_given_valid for bound-based, then z3 constify, then linear synth
  idx_vec = UOp(Ops.VECTORIZE, dtypes.index.vec(2), (idx0, idx1))
  idx_vec = uop_given_valid(new_valid, idx_vec)
  new_idx0, new_idx1 = idx_vec.src[0] if idx_vec.op is Ops.VECTORIZE else idx0, idx_vec.src[1] if idx_vec.op is Ops.VECTORIZE else idx1
  new_idx0 = z3_constify(new_idx0, new_valid)
  lin0 = z3_linear_synth(new_idx0, new_valid)
  if lin0 is not None: new_idx0 = lin0
  new_idx1 = z3_constify(new_idx1, new_valid)
  lin1 = z3_linear_synth(new_idx1, new_valid)
  if lin1 is not None: new_idx1 = lin1

  # try dropping valid clauses: a clause is droppable if when it's false, some idx component is OOB
  if new_valid.op is not Ops.CONST or new_valid.arg is not True:
    clauses = list(new_valid.split_uop(Ops.AND))
    kept = [c for c in clauses if not _z3_clause_droppable(c, new_idx0, new_idx1, width, height)]
    if len(kept) < len(clauses):
      new_valid = kept[0] if kept else None
      if kept:
        for k in kept[1:]: new_valid = new_valid & k
      else:
        new_valid = None

  if new_valid is not None and new_valid.op is Ops.CONST and new_valid.arg is True: new_valid = None
  return new_idx0, new_idx1, new_valid

# ===== tests matching test_simplify_valid_idx.py =====

class TestZ3SynthIdx(unittest.TestCase):
  """Verify z3 synthesis produces correct simplifications for every case in test_simplify_valid_idx."""

  def _render(self, u:UOp) -> str: return u.render()

  def test_cumsum(self):
    gidx0 = Special("gidx0", 5)
    lidx0 = Special("lidx0", 4)
    gate = (gidx0*4+lidx0<19).ne(True)
    idx = gidx0*4+lidx0-19
    new_idx, new_valid = z3_simplify(idx, gate)
    self.assertTrue(z3_is_equiv(new_idx, UOp.const(dtypes.index, 0), under=new_valid))
    self.assertEqual(int(new_idx.vmin), 0)
    self.assertEqual(int(new_idx.vmax), 0)
    print(f"  idx: {self._render(new_idx)}, valid: {self._render(new_valid)}")

  def test_simplify_within_valid1(self):
    ridx0, ridx1, ridx2, ridx3 = Range(0, 4), Range(1, 4), Range(2, 4), Range(3, 4)
    valid = ((ridx0*3+ridx1)<8) & ((((ridx0*3+ridx1)//8+ridx2*3+ridx3)%4)<2)
    idx = ridx0+ridx1+ridx2+ridx3
    new_idx, new_valid = z3_simplify(idx, valid)
    # idx unchanged, valid simplified (//8 term removed)
    self.assertTrue(z3_is_equiv(idx, new_idx, under=new_valid))
    self.assertTrue(z3_is_equiv(valid, new_valid))
    # the //8 should be gone
    self.assertFalse(any(u.op is Ops.IDIV for u in new_valid.toposort()), "//8 should be eliminated from valid")
    print(f"  idx: {self._render(new_idx)}, valid: {self._render(new_valid)}")

  def test_simplify_within_valid2(self):
    gidx0 = Special("gidx0", 56)
    ridx0 = Range(0, 3)
    alu0 = gidx0+ridx0
    valid = (alu0 < 57) & (alu0 >= 1)
    # this should NOT simplify (both clauses are needed)
    new_valid = z3_simplify_valid(valid)
    self.assertTrue(z3_is_equiv(valid, new_valid))
    # valid still has 2 clauses
    self.assertGreaterEqual(len(list(new_valid.split_uop(Ops.AND))), 2)
    print(f"  valid: {self._render(new_valid)}")

  def test_valid_order_matters1(self):
    ridx0 = Range(0, 2)
    v0 = ridx0<1
    v1 = ((ridx0*5+1)%6)<5
    new_valid = z3_simplify_valid(v0&v1)
    self.assertTrue(z3_is_equiv(v0&v1, new_valid))
    self.assertEqual(len(list(new_valid.split_uop(Ops.AND))), 1, "v1 should be dropped")
    print(f"  valid: {self._render(new_valid)}")

    # order shouldn't matter
    new_valid2 = z3_simplify_valid(v1&v0)
    self.assertTrue(z3_is_equiv(v0&v1, new_valid2))
    self.assertEqual(len(list(new_valid2.split_uop(Ops.AND))), 1)

  def test_valid_order_matters2(self):
    gidx0 = Special("gidx0", 13)
    gidx1 = Special("gidx1", 13)
    ridx0 = Range(0, 4)
    alu0 = (gidx1+(ridx0*13))
    v0 = (gidx0+11)%14<11
    v1 = (alu0+((gidx0+39)//42))%14<11
    v2 = gidx0<3
    v3 = alu0<42
    new_valid = z3_simplify_valid(v0&v1&v2&v3)
    self.assertEqual(self._render(new_valid), "False")
    print(f"  valid: {self._render(new_valid)}")

  def test_valid_becomes_const1(self):
    ridx0, ridx1, ridx2 = Range(0, 30), Range(1, 7), Range(2, 2)
    alu11 = (ridx1+ridx2)
    alu15 = ((alu11+1)//7)
    idx = (alu15*-31)+(((((alu11+218)//224)+ridx0)%30)*1568)
    valid = (ridx2<1)&(ridx1<6)
    new_idx, new_valid = z3_simplify(idx, valid)
    self.assertTrue(z3_is_equiv(idx, new_idx, under=valid))
    print(f"  idx: {self._render(new_idx)}, valid: {self._render(new_valid)}")
    # idx should simplify to r0*1568
    expected = ridx0 * 1568
    self.assertTrue(z3_is_equiv(new_idx, expected, under=valid))

  def test_valid_becomes_const1_z3_proof(self):
    """Replicate the manual z3 proof from the original test, but using synthesis."""
    ridx0, ridx1, ridx2 = Range(0, 30), Range(1, 7), Range(2, 2)
    alu11 = (ridx1+ridx2)
    alu15 = ((alu11+1)//7)
    idx = (alu15*-31)+(((((alu11+218)//224)+ridx0)%30)*1568)
    valid = (ridx2<1)&(ridx1<6)
    new_idx, _ = z3_simplify(idx, valid)
    # verify correct
    self.assertTrue(z3_is_equiv(idx, new_idx, under=valid))
    # verify a wrong answer is caught
    wrong = ridx0*1567 + ridx1
    self.assertFalse(z3_is_equiv(idx, wrong, under=valid))

  def test_valid_becomes_const2(self):
    ridx0, ridx1, ridx2, ridx3 = Range(0, 4), Range(1, 4), Range(2, 4), Range(3, 4)
    idx = (((ridx0+ridx1)+(ridx2+ridx3)+28)//30)
    valid = ((ridx0+ridx1)<1).ne(True) & ((ridx2+ridx3)<1).ne(True)
    new_idx, new_valid = z3_simplify(idx, valid)
    self.assertTrue(z3_is_equiv(idx, new_idx, under=valid))
    self.assertEqual(int(new_idx.vmin), 1)
    self.assertEqual(int(new_idx.vmax), 1)
    print(f"  idx: {self._render(new_idx)}, valid: {self._render(new_valid)}")

  def test_valid_with_non_const_rhs(self):
    ridx0, ridx1, ridx2 = Range(0, 1024), Range(1, 4), Range(2, 4)
    valid = (ridx0<(ridx1*4 + ridx2))&(ridx0<-1).ne(True)
    idx = ridx0
    new_idx, new_valid = z3_simplify(idx, valid)
    self.assertTrue(z3_is_equiv(valid, new_valid))
    # the (r0<-1).ne(True) clause should be dropped (always True for r0 >= 0)
    self.assertEqual(len(list(new_valid.split_uop(Ops.AND))), 1)
    print(f"  idx: {self._render(new_idx)}, valid: {self._render(new_valid)}")

  def test_from_merge_views(self):
    """This is an expectedFailure in the original tests — z3 synthesis should handle it!"""
    ridx0, ridx2, ridx3 = Range(0, 2), Range(2, 2), Range(3, 2)
    idx = (((ridx0*2)+((((ridx2*2)+(ridx3*3))+3)%4))+-2)
    valid = ((((((ridx2*2)+(ridx3*3))+3)%4)<2)!=True)  # noqa: E712
    new_idx, new_valid = z3_simplify(idx, valid)
    # verify equivalence
    self.assertTrue(z3_is_equiv(idx, new_idx, under=valid))
    self.assertTrue(z3_is_equiv(valid, new_valid))
    print(f"  idx: {self._render(new_idx)}, valid: {self._render(new_valid)}")
    # expected: idx = r0*2 - r3 + 1, valid = r2 < 1
    expected_idx = ridx0*2 + ridx3*-1 + 1
    self.assertTrue(z3_is_equiv(new_idx, expected_idx, under=new_valid))

  def test_load_in_valid(self):
    # has a LOAD in the valid — synthesis should still work (or gracefully skip)
    ridx2 = Range(2, 4)
    lidx0 = Special("lidx0", 3)
    gidx0 = Special("gidx0", 2)
    idx = (((lidx0+(gidx0*3))+(ridx2*5))+40)
    valid = (lidx0+(gidx0*3)) < 5
    # just verify constify doesn't crash
    new_idx = z3_constify(idx, valid)
    self.assertTrue(z3_is_equiv(idx, new_idx, under=valid))

class TestZ3SynthImage(unittest.TestCase):
  def _render(self, u:UOp) -> str: return u.render()

  def test_idx_gt_c(self):
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    shape = (10, 10, 4)
    # valid: gidx1 >= 1, idx = (gidx0, gidx1-1)
    valid = (gidx1<1).ne(True)
    idx0, idx1 = gidx0, gidx1-1
    new_idx0, new_idx1, new_valid = z3_simplify_image(idx0, idx1, valid, shape)
    self.assertIsNone(new_valid, "valid should be dropped (idx in-bounds under valid)")
    print(f"  idx0: {self._render(new_idx0)}, idx1: {self._render(new_idx1)}, valid: {new_valid}")

  def test_idx_gt_c_with_and(self):
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    shape = (10, 10, 4)
    valid = (gidx0<1).ne(True) & (gidx1<1).ne(True)
    idx0, idx1 = gidx0+1, gidx1-1
    new_idx0, new_idx1, new_valid = z3_simplify_image(idx0, idx1, valid, shape)
    # only the gidx1 clause can be dropped (gidx1-1 >= 0 when gidx1>=1)
    # gidx0 clause should remain (gidx0+1 can be up to 32, but shape[1]=10)
    self.assertIsNotNone(new_valid)
    print(f"  idx0: {self._render(new_idx0)}, idx1: {self._render(new_idx1)}, valid: {self._render(new_valid)}")

  def test_idx_lt_bound(self):
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    # valid: gidx1 < 10, shape width = 10
    valid = gidx1<10
    idx0, idx1 = gidx0, gidx1
    new_idx0, new_idx1, new_valid = z3_simplify_image(idx0, idx1, valid, (10, 10, 4))
    self.assertIsNone(new_valid, "valid should be dropped (gidx1<10 ensures in-bounds)")
    print(f"  idx0: {self._render(new_idx0)}, idx1: {self._render(new_idx1)}, valid: {new_valid}")

  def test_idx_lt_bound_no_drop(self):
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    # 10x20 image, gidx1<10 does NOT ensure idx1 < height=20 (it does, but idx0 could be > width=20)
    # actually: gidx1 < 10 and shape is (20, 10, 4), width=10. gidx1<10 ensures idx1<10<20. But gidx0 up to 31 > 10.
    # Hmm, wait. idx0=gidx0 can be up to 31, width=10. So gidx0 can be out of bounds.
    # But the test checks that valid is NOT dropped because... let me check the original test.
    # Original: shape=(20,10,4), valid=gidx1<10, idx=(gidx0, gidx1). Expected: valid stays.
    # Because gidx0 can be up to 31 >= width=10.
    valid = gidx1<10
    idx0, idx1 = gidx0, gidx1
    new_idx0, new_idx1, new_valid = z3_simplify_image(idx0, idx1, valid, (20, 10, 4))
    self.assertIsNotNone(new_valid, "valid should NOT be dropped (gidx0 can exceed width)")
    print(f"  idx0: {self._render(new_idx0)}, idx1: {self._render(new_idx1)}, valid: {self._render(new_valid)}")

  def test_generic_idx_lt_bound(self):
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    shape = (10, 10, 4)
    valid = gidx1<8
    idx0, idx1 = gidx0, gidx1+2
    new_idx0, new_idx1, new_valid = z3_simplify_image(idx0, idx1, valid, shape)
    self.assertIsNone(new_valid, "gidx1<8 and idx1=gidx1+2<10 → in-bounds")
    print(f"  idx0: {self._render(new_idx0)}, idx1: {self._render(new_idx1)}, valid: {new_valid}")

  def test_openpilot_conv1(self):
    idx1 = Special("idx1", 32)
    idx2 = Special("idx2", 64)
    ridx0, ridx1, ridx2 = Range(0, 6), Range(1, 3), Range(2, 3)
    alu1 = ((idx2*2)+ridx1)
    alu4 = ((idx1*48)+(ridx2*6)+ridx0)
    valid = ((((idx2*2)+(ridx1))<1).ne(True))&((((idx1*8)+(ridx2))<1).ne(True))
    shape = (128, 1536, 4)
    idx0_expr = (alu4+1530)%1536
    idx1_expr = alu1+((idx1+((ridx2+7)//8)+31)//32)+(-2)

    new_idx0, new_idx1, new_valid = z3_simplify_image(idx0_expr, idx1_expr, valid, shape)
    # verify equivalence
    self.assertTrue(z3_is_equiv(idx0_expr, new_idx0, under=valid))
    self.assertTrue(z3_is_equiv(idx1_expr, new_idx1, under=valid))
    self.assertIsNone(new_valid, "valid should be droppable for openpilot conv1")
    print(f"  idx0: {self._render(new_idx0)}, idx1: {self._render(new_idx1)}, valid: {new_valid}")

  def test_openpilot_conv2(self):
    idx1 = Special("idx1", 32)
    idx2 = Special("idx2", 64)
    ridx0, ridx1, ridx2 = Range(0, 3), Range(1, 3), Range(2, 3)
    alu1 = ((idx2*2)+ridx1)
    alu3 = ((idx1*24)+(ridx2*3)+ridx0)
    valid = ((((idx2*2)+ridx1)<1).ne(True))&((((idx1*8)+ridx2)<1).ne(True))
    shape = (128, 768, 4)
    idx0_expr = (alu3+765)%768
    idx1_expr = alu1+((idx1+((ridx2+7)//8)+31)//32)+(-2)

    new_idx0, new_idx1, new_valid = z3_simplify_image(idx0_expr, idx1_expr, valid, shape)
    self.assertTrue(z3_is_equiv(idx0_expr, new_idx0, under=valid))
    self.assertTrue(z3_is_equiv(idx1_expr, new_idx1, under=valid))
    self.assertIsNone(new_valid, "valid should be droppable for openpilot conv2")
    print(f"  idx0: {self._render(new_idx0)}, idx1: {self._render(new_idx1)}, valid: {new_valid}")

  def test_simplify1(self):
    gidx = Special("gidx", 512)
    valid = (gidx<488) & (gidx<480).ne(True)
    idx0_expr = (gidx*3+18)%26
    idx1_expr = (gidx*3+18)//26-56
    new_idx0, new_idx1, new_valid = z3_simplify_image(idx0_expr, idx1_expr, valid, (1, 26, 4))
    self.assertTrue(z3_is_equiv(idx0_expr, new_idx0, under=valid))
    self.assertTrue(z3_is_equiv(idx1_expr, new_idx1, under=valid))
    self.assertIsNone(new_valid)
    print(f"  idx0: {self._render(new_idx0)}, idx1: {self._render(new_idx1)}, valid: {new_valid}")

  def test_simplify2(self):
    lidx = Special("lidx", 4)
    valid = (lidx<3) & (lidx<1).ne(True)
    idx0_expr = (lidx+1)%2
    idx1_expr = (lidx+1)//2-1
    new_idx0, new_idx1, new_valid = z3_simplify_image(idx0_expr, idx1_expr, valid, (1, 2, 4))
    self.assertTrue(z3_is_equiv(idx0_expr, new_idx0, under=valid))
    self.assertTrue(z3_is_equiv(idx1_expr, new_idx1, under=valid))
    self.assertIsNone(new_valid)
    print(f"  idx0: {self._render(new_idx0)}, idx1: {self._render(new_idx1)}, valid: {new_valid}")

  def test_simplify3(self):
    idx0 = Special("idx0", 265)
    valid = (idx0<201).ne(True)
    idx0_expr = (idx0+55)%64
    idx1_expr = (idx0+55)//64-4
    new_idx0, new_idx1, new_valid = z3_simplify_image(idx0_expr, idx1_expr, valid, (1, 64, 4))
    self.assertTrue(z3_is_equiv(idx0_expr, new_idx0, under=valid))
    self.assertTrue(z3_is_equiv(idx1_expr, new_idx1, under=valid))
    self.assertIsNone(new_valid)
    print(f"  idx0: {self._render(new_idx0)}, idx1: {self._render(new_idx1)}, valid: {new_valid}")

  def test_simplify6(self):
    idx1 = Special("idx1", 16)
    idx2 = Special("idx2", 64)
    ridx3, ridx4, ridx5 = Range(3, 3), Range(4, 3), Range(5, 3)
    alu0 = ((idx2*1536)+(ridx4*768)+ridx3+(idx1*24)+(ridx5*3)+-771)%768
    alu1 = ((idx2*1536)+(ridx4*768)+ridx3+(idx1*24)+(ridx5*3)+-771)//768
    valid = (((idx2+ridx4)<1)!=1)&(((idx1+ridx5)<1)!=1)
    new_idx0, new_idx1, new_valid = z3_simplify_image(alu0, alu1, valid, (128, 768, 4))
    self.assertTrue(z3_is_equiv(alu0, new_idx0, under=valid))
    self.assertTrue(z3_is_equiv(alu1, new_idx1, under=valid))
    self.assertIsNone(new_valid)
    print(f"  idx0: {self._render(new_idx0)}, idx1: {self._render(new_idx1)}, valid: {new_valid}")

class TestZ3SynthCorrectness(unittest.TestCase):
  """Meta-tests: verify that the synthesis tool catches wrong answers."""
  def test_wrong_idx_detected(self):
    ridx0 = Range(0, 30)
    ridx1 = Range(1, 7)
    valid = (ridx1<6)
    idx = ridx0 * 1568
    wrong = ridx0 * 1567
    self.assertTrue(z3_is_equiv(idx, idx, under=valid))
    self.assertFalse(z3_is_equiv(idx, wrong, under=valid))

  def test_wrong_valid_detected(self):
    ridx0 = Range(0, 4)  # 0..3
    v = ridx0 < 2
    wrong_v = ridx0 < 3  # not equivalent (different cutoff)
    self.assertFalse(z3_is_equiv(v, wrong_v))

  def test_unsat_detected(self):
    r = Range(0, 4)
    self.assertTrue(z3_is_unsat((r < 0) & (r < 4)))
    self.assertFalse(z3_is_unsat(r < 2))

if __name__ == '__main__':
  unittest.main()
