# [Cleanup and Logical Correctness Plan]

This plan addresses the 9 remaining issues identified during the post-refactoring re-scan of the PoreMS repository. These changes improve logical correctness in multi-shape pores, formalize sentinel values, and align with modern Python conventions (3.14+).

## Proposed Changes

### [porems/system.py]

#### [MODIFY] [system.py](file:///Users/dm/dev/PoreMS/porems/system.py)

- Update `volume()` and `surface()` to use `elif` for `CONE` branches.
- Initialize `length = None` in `volume()` and `surface()` loop bodies before SPHERE/others.
- Define `_UNASSIGNED_SHAPE_KEY = 20` as a module constant and replace all literals.
- Refactor `_collect_shape_radii` and `_analysis_site_groups` to ensure consistent dictionary usage.
- Fix `inp == "percent"` logic in `attach()` to properly filter geminal sites by shape membership using `self._pore._sites[site_id].oxygen_count == 2` correctly against shape-assigned sites.
- Extract common table formatting into a small helper to reduce repetition in `table()`.

---

### [porems/shape.py]

#### [MODIFY] [shape.py](file:///Users/dm/dev/PoreMS/porems/shape.py)

- Rename inner `r(z)` function to `r_func(z)` in `Cone` methods (`Phi`, `is_in`, etc.) to prevent shadowing of the parameter `r`.

---

### [porems/pore.py]

#### [MODIFY] [pore.py](file:///Users/dm/dev/PoreMS/porems/pore.py)

- Replace `from typing import Callable` with `from collections.abc import Callable`.

---

### [setup.py]

#### [MODIFY] [setup.py](file:///Users/dm/dev/PoreMS/setup.py)

- Change `python_requires` to `">=3.14"`.

---

## Verification Plan

### Automated Tests
- Run the full existing test suite:
  ```bash
  mamba run -n pore python -m pytest tests/ -v
  ```
- Ensure all 37 tests continue to pass correctly.
- Add a small test script to verify `percent` attachment consistency in a multi-shape `PoreKit`.

### Manual Verification
- Inspect the generated `.yml` summaries of multi-shape pores to confirm volume/surface sums are consistent.
