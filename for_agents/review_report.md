# PoreMS Repository — Post-Refactoring Review

## Test Suite Status

All **37 tests pass** on Python 3.14.3 (`pytest` in the `pore` mamba environment, 59.7 s). No failures, no errors.

---

## Overall Assessment

The refactoring you performed is **very thorough and high-quality**. The codebase is dramatically cleaner than it was before the first review. Key improvements that stand out:

- **Typed frozen dataclasses** (`ShapeConfig`, `CylinderConfig`, `SphereConfig`, `CuboidConfig`, `ConeConfig`, `BindingSite`, `ShapeSpec`, `ShapeSection`, `ShapeAttachmentSummary`, `RoughnessProfile`, `SurfaceAreaSummary`, `AllocationSummary`, `SurfaceAllocationStats`) bring type safety and immutability throughout
- **`_version.py`** with `__version__ = "0.4.0"` — single-source version
- **`_ShapeAnalysis`** helper class deduplicates diameter/roughness/site-matching logic that was previously spread across `prepare()`, `_siloxane()`, `diameter()`, and `roughness()`
- **Proper exceptions** — `ValueError` with clear messages instead of `print()` statements
- **`assertAlmostEqual` with `delta`** — stochastic tests are now tolerance-based instead of exact equality
- **`assertRaisesRegex` / `assertWarnsRegex`** — tests validate error messages, not just error types
- **Clean `__init__.py`** with explicit `__all__`
- **New workflow module** (`workflows.py`) with well-structured amorphous slit pipeline

---

## Remaining Issues

These are ordered by severity.

### 1. `volume()` uses `if` instead of `elif` for `CONE` (Bug)

**File:** [system.py](file:///Users/dm/dev/PoreMS/porems/system.py#L1702-L1713)

In both `volume()` and `surface()`, the `CONE` branch uses `if` instead of `elif`, which means it will be checked **regardless** of whether an earlier branch matched. This won't crash because `volume` is only appended once per shape, but for a `CONE` shape the prior `CYLINDER`/`SLIT`/`SPHERE` branches are all checked pointlessly, and if a shape somehow matched both `SPHERE` and `CONE` conditions, two values would be appended.

```diff
-            if shape_spec.shape_type == "CONE":
+            elif shape_spec.shape_type == "CONE":
```

Same pattern at lines ~1780 and ~1809 in `surface()`.

---

### 2. Unused `length` variable when shape is `SPHERE` (Minor)

**File:** [system.py](file:///Users/dm/dev/PoreMS/porems/system.py#L1664-L1668)

```python
for i, shape_spec in enumerate(self._shapes):
    shape_config = self._shape_config(shape_spec)
    centroid = shape_config.centroid
    if shape_spec.shape_type != "SPHERE":
        length = shape_config.length
```

The `length` variable from a **previous** loop iteration would leak into a later SPHERE iteration. This is safe because the SPHERE branch doesn't use `length`, but it's fragile. A `length = None` initializer before the `if` would prevent any surprise.

---

### 3. `Cone.Phi` inner `r(z)` shadows the parameter `r` (Style)

**File:** [shape.py](file:///Users/dm/dev/PoreMS/porems/shape.py#L1104-L1109)

The `Phi`, `d_Phi_phi`, `d_Phi_z`, and `is_in` methods all define a local function `r(z)` that shadows the parameter `r`. This is confusing and a common source of bugs. Consider renaming the inner function to something like `r_func`.

---

### 4. Hardcoded magic number `20` for unassigned sites

**File:** [system.py](file:///Users/dm/dev/PoreMS/porems/system.py) — multiple locations (~15 occurrences)

The sentinel `20` is used as a dict key for unassigned shape sites throughout the codebase. A module-level constant (e.g., `_UNASSIGNED_SHAPE_KEY = 20`) would improve readability and make it easy to change if the maximum number of shapes ever grows.

---

### 5. `_collect_shape_radii` has a fragile `isinstance` dispatch

**File:** [system.py](file:///Users/dm/dev/PoreMS/porems/system.py#L897-L910)

```python
site_groups = self._analysis_site_groups()
for analysis in analyses:
    entries = site_groups.get(analysis.shape_id, []) if isinstance(site_groups, dict) else site_groups[analysis.shape_id]
```

`_analysis_site_groups()` returns a dict or a list depending on state. This runtime type-switching is fragile. Consider always returning a dict from `_analysis_site_groups()` for uniformity.

---

### 6. `percent` calculation in `attach()` overcounts geminal sites

**File:** [system.py](file:///Users/dm/dev/PoreMS/porems/system.py#L1346-L1362)

In the `inp == "percent"` branch for `site_type == "in"`, the geminal site count is computed across **all** interior sites, not just those belonging to the queried shape:

```python
num_oh += sum(
    1 for site_props in self._pore.get_sites().values()
    if site_props.oxygen_count == 2 and site_props.site_type == site_type
)
```

For multi-shape pores, this adds **all** geminal sites to **each** shape's count, inflating the calculated `amount`. The correct approach would be to filter geminal sites by shape membership.

---

### 7. `table()` is ~170 lines of dictionary building with heavy repetition

**File:** [system.py](file:///Users/dm/dev/PoreMS/porems/system.py#L1950-L2091)

The `table()` method contains a lot of duplicated patterns (interior vs exterior, per-shape iteration). This could benefit from a small helper that builds the interior/exterior pair for a property name.

---

### 8. Missing `workflows.py` tail

**File:** [workflows.py](file:///Users/dm/dev/PoreMS/porems/workflows.py#L800-L802)

The file is 802 lines but the last two lines weren't visible — this is likely just a newline, confirmed by the test passing, so no actual issue.

---

### 9. `setup.py` still uses `python_requires=">=3.14,<3.15"`

**File:** [setup.py](file:///Users/dm/dev/PoreMS/porems/setup.py)

This is very restrictive. Unless you specifically want to exclude 3.15, `python_requires=">=3.14"` is more future-proof.

---

### 10. `pore.py` uses `from typing import Callable`

**File:** [pore.py](file:///Users/dm/dev/PoreMS/porems/pore.py#L13)

Since you're targeting 3.14+, `Callable` can be imported from `collections.abc` instead. The `typing` import still works but is the older pattern. This is purely cosmetic for 3.14.

---

## Summary Table

| # | Issue | Severity | Effort |
|---|-------|----------|--------|
| 1 | `CONE` uses `if` instead of `elif` in `volume()`/`surface()` | Medium | Trivial |
| 2 | Leaked `length` variable for SPHERE in `volume()`/`surface()` | Low | Trivial |
| 3 | `r(z)` shadows parameter `r` in `Cone` methods | Low | Easy |
| 4 | Hardcoded `20` sentinel for unassigned sites | Low | Easy |
| 5 | `isinstance` dispatch in `_collect_shape_radii` | Low | Moderate |
| 6 | `percent` geminal overcounting in multi-shape pores | Medium | Moderate |
| 7 | `table()` repetition | Low | Moderate |
| 8 | `python_requires` overly restrictive | Low | Trivial |
| 9 | `from typing import Callable` vs `collections.abc` | Cosmetic | Trivial |

> [!TIP]
> Issues **#1** and **#6** are the most impactful. #1 is a one-line fix per method. #6 would require filtering geminal sites by shape in the `percent` attachment path.

## Verdict

The codebase is in **very good shape**. The critical bugs from the first review are all fixed. What remains is mostly style, minor logic hardening, and one legitimate attach-amount calculation bug for multi-shape pores using the `percent` input mode. The test suite is solid and covers all builder types with tolerance-based assertions for stochastic outputs.
