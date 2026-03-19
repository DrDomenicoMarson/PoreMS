# PoreMS Repository Inspection — Bug & Issue Report

Full codebase analysis of PoreMS (v0.4.0). Issues below are grouped by severity.

---

## 🔴 Critical Bugs (will crash or produce wrong results)

### 1. Dead code / unreachable return in `Cone.normal()`
#### [MODIFY] [shape.py](file:///Users/dm/dev/PoreMS/porems/shape.py)

Lines 996–997 contain two consecutive `return` statements — the second (correct) one computing the cross product is **never reached**:

```python
return geometry.rotate(d_Phi_phi, self._inp["central"], -90, True)  # ← always runs
return geometry.cross_product(d_Phi_phi, d_Phi_z)                   # ← dead code
```

The rotate-based result does not compute the true surface normal. The cross-product formula should be the one used (consistent with `Cylinder.normal` and `Sphere.normal`). **Fix:** remove the first return.

---

### 2. `NameError` in `roughness()` when no reservoir is present
#### [MODIFY] [system.py](file:///Users/dm/dev/PoreMS/porems/system.py)

In `PoreKit.roughness()` (line ~1054-1063), the variable `size` is only assigned inside `if self._res:`, but the line `r_ex = [pos[2] if pos[2] < size/2 ...]` is **outside** the if-block and always executes:

```python
if self._res:
    temp_mol = pms.Molecule()
    ...
    size = temp_mol.get_box()[2]

# This line always runs, even when self._res is 0:
r_ex = [pos[2] if pos[2] < size/2 else abs(pos[2]-size) for pos in self._si_pos_ex]
```

If `self._res == 0`, this raises `NameError: name 'size' is not defined`. **Fix:** guard the exterior roughness block behind the `self._res` check, or default `r_ex = []`.

---

### 3. `SPHERE` shape accesses undefined `length` in `prepare()`
#### [MODIFY] [system.py](file:///Users/dm/dev/PoreMS/porems/system.py)

In `PoreKit.prepare()` line ~373-397 (the free-binding-sites loop), the variable `length` is only set when the shape is not `SPHERE`:

```python
if not shape[0]=="SPHERE":
    length = self._shapes[i][1].get_inp()["length"]
```

But then for `SPHERE` on line ~397:
```python
elif shape[0]=="SPHERE":
    if (centroid[2]-(length)/2)<p[2]<(centroid[2]+(length)/2):
```

This will use a stale `length` from a previous loop iteration (if there was one) or crash with `NameError` if `SPHERE` is the first shape. **Fix:** use `diameter` for spheres (which is the correct geometric extent).

---

### 4. Missing `numpy` and `matplotlib` from install dependencies
#### [MODIFY] [setup.py](file:///Users/dm/dev/PoreMS/setup.py)

`numpy` is imported in `pattern.py`, `shape.py`, `system.py`; `matplotlib` is imported in `shape.py`. Neither is listed in `install_requires`. **Fix:** add both to `install_requires` in `setup.py` and `requirements.txt`.

---

## 🟡 Logic / Correctness Issues

### 5. Mutable default arguments
#### [MODIFY] [system.py](file:///Users/dm/dev/PoreMS/porems/system.py)

Multiple constructors use `hydro=[0, 0]` as a default argument. In Python, mutable defaults are shared across calls and can lead to subtle mutation bugs.

Affected signatures:
- `PoreCylinder.__init__(... hydro=[0, 0])`
- `PoreSlit.__init__(... hydro=[0, 0])`
- `PoreCapsule.__init__(... hydro=[0, 0])`
- `PoreAmorphCylinder.__init__(... hydro=[0, 0])`

Also affected:
- `Shape.plot(... vec=[])` in [shape.py](file:///Users/dm/dev/PoreMS/porems/shape.py)

**Fix:** Use `None` as default and assign `[0, 0]` inside the function body.

---

### 6. Bare `except` clauses throughout `system.py`
#### [MODIFY] [system.py](file:///Users/dm/dev/PoreMS/porems/system.py)

There are **>20 bare `except:`** blocks in `system.py` (in `diameter()`, `roughness()`, `prepare()`, `_siloxane()`, `attach()`, `table()`). These swallow **all** exceptions including `KeyboardInterrupt`, `SystemExit`, and genuine bugs, making debugging extremely difficult.

Most of them are try/except around `self._pore.get_block().pos(index)` — they catch the case where `index` is already a position list vs. an atom ID. **Fix:** Use `isinstance()` checks or catch `(TypeError, IndexError, KeyError)` specifically.

---

### 7. Error handling via `print()` instead of raising exceptions
#### [MODIFY] [system.py](file:///Users/dm/dev/PoreMS/porems/system.py)

Multiple methods validate inputs and print an error message, then `return None`:

```python
if shape[0] not in ["CYLINDER", "SLIT", "SPHERE", "CONE"]:
    print("Wrong shape type...")
    return
```

This makes errors silent — callers don't know something went wrong. **Fix:** Replace `print` + `return` with `raise ValueError(...)`.

Also applies to [pore.py](file:///Users/dm/dev/PoreMS/porems/pore.py), [geometry.py](file:///Users/dm/dev/PoreMS/porems/geometry.py), [molecule.py](file:///Users/dm/dev/PoreMS/porems/molecule.py).

---

### 8. Variable shadowing: `mol` reused as loop variable
#### [MODIFY] [system.py](file:///Users/dm/dev/PoreMS/porems/system.py)

In `PoreKit.attach()` lines ~718-721 and in `PoreSlit.attach_special()` / `PoreAmorphCylinder.attach_special()`, the parameter `mol` is shadowed by the loop variable `mol`:

```python
def attach(self, mol, ...):  # ← parameter
    ...
    mols = self._pore.attach(mol, ...)
    for mol in mols:  # ← shadows parameter!
        if not mol.get_short() in self._sort_list:
```

After the loop, `mol` no longer refers to the input molecule but to the last element of `mols`. **Fix:** Rename the loop variable to e.g. `attached_mol`.

---

### 9. Removing items from list while iterating over it
#### [MODIFY] [system.py](file:///Users/dm/dev/PoreMS/porems/system.py)

In `PoreKit.attach()` lines ~685-689 and ~704-708:

```python
for i in self._pore.sites_sl_shape:
    for si in self._pore.sites_sl_shape[i]:
        state = self._pore._sites[si]["state"]
        if state == False:
            self._pore.sites_sl_shape[i].remove(si)  # ← modifying list while iterating!
```

This is a classical Python bug — removing items from a list during iteration skips elements. **Fix:** Build a new filtered list instead.

---

### 10. Nested loop variable `i` shadows outer `i`
#### [MODIFY] [system.py](file:///Users/dm/dev/PoreMS/porems/system.py)

In `PoreKit.prepare()` lines ~309-311:

```python
for i in range(len(self._shapes)):        # outer i
    for i, section in enumerate(self._sections):  # inner i shadows outer!
```

The inner loop's `i` shadows the outer `i`, causing incorrect shape indexing. **Fix:** Rename the inner variable (e.g., `j`).

---

## 🟢 Code Quality / Maintainability

### 11. Massive code duplication between `diameter()` and `roughness()`

Both methods in `PoreKit` compute the same distance-to-central-axis calculations with nearly identical code (~100 lines each). They should share a private helper method.

### 12. Duplicated site-classification logic

The free-binding-site classification logic appears in both `_siloxane("in")` block and `prepare()` — nearly identical loops iterating over shapes and classifying sites by geometry. This should be extracted to a shared helper.

### 13. `PoreAmorphCylinder.attach_special` has wrong signature

In [system.py](file:///Users/dm/dev/PoreMS/porems/system.py) line 1792, `attach_special` for `PoreAmorphCylinder` passes `self._normal_in` to `pore.attach()`, but `_normal_in` is never defined on the class. This will raise an `AttributeError` at runtime.

### 14. Commented-out test assertion

In [test_simple.py](file:///Users/dm/dev/PoreMS/tests/test_simple.py) line 964:
```python
# elf.assertIsNone(pore.attach_special(...)
```
This is a typo (`elf` instead of `self`), suggesting a disabled test.

---

## Proposed Changes

I'll fix issues **1–10** (the critical and logic/correctness categories) and issue **13** (`_normal_in` AttributeError). I'll leave cleanups in 11–12 (refactoring) as suggestions rather than applying them now, since they're pure code-quality and would produce a very large diff touching many methods.

### Summary of file changes

| File | Changes |
|------|---------|
| [setup.py](file:///Users/dm/dev/PoreMS/setup.py) | Add `numpy`, `matplotlib` to `install_requires` |
| [requirements.txt](file:///Users/dm/dev/PoreMS/requirements.txt) | Add `numpy`, `matplotlib` |
| [shape.py](file:///Users/dm/dev/PoreMS/porems/shape.py) | Fix dead return in `Cone.normal()`, fix mutable default in `plot()` |
| [system.py](file:///Users/dm/dev/PoreMS/porems/system.py) | Fix `roughness()` NameError, fix SPHERE `length` bug, fix mutable defaults, fix variable shadowing, fix list-mutation-during-iteration, fix bare excepts in critical paths, fix `_normal_in` AttributeError |

> [!IMPORTANT]
> **Items 6 and 7** (bare `except` and `print`-based errors) are pervasive — there are 20+ instances across `system.py` alone. I'll fix the most dangerous ones (those that hide real bugs), but fully rewriting all error handling is a larger project. If you want all of them addressed, let me know and I'll tackle them comprehensively.

## Verification Plan

### Automated Tests

All existing tests must continue passing:

```bash
cd /Users/dm/dev/PoreMS && mamba run -n pore python -m pytest tests/ -v
```

### Manual Verification

After changes, confirm:
1. `import porems` works cleanly with no import errors
2. `pip show porems` (or inspecting `setup.py`) shows numpy and matplotlib as dependencies
3. `Cone.normal()` returns the cross-product normal (spot-check via the existing `test_shape_cone`)
