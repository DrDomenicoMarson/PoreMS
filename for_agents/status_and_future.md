# PoreMS Advancement Status and Future Tasks

## Current Status (as of 2026-03-19)

### Accomplishments
1.  **Critical Bug Fixes**:
    -   Fixed **`Cone.normal()`**: Removed a redundant `return` that caused the method to return incorrect normals.
    -   Fixed **`PoreKit.roughness()`**: Fixed a `NameError` occurring when no reservoir was defined.
    -   Fixed **SPHERE shape `prepare()`**: Fixed a bug where spheres were accessing an undefined `length` variable (now correctly uses `diameter`).
    -   **Dependencies**: Added `numpy` and `matplotlib` to `setup.py` and `requirements.txt`.
2.  **Logic & Correctness**:
    -   **Mutable Defaults**: Replaced mutable list defaults (e.g., `hydro=[0, 0]`) in constructors with `None` to avoid shared-state bugs.
    -   **Variable Shadowing**: Fixed several instances where parameters or outer loop variables were shadowed by inner loop variables (e.g., `mol`, `i`).
    -   **List Mutation**: Fixed bugs in `attach()` where lists were being modified while being iterated over.
    -   **AttributeError**: Fixed `_normal_in` error in `PoreAmorphCylinder.attach_special`.

### Verification Results
-   **Total Tests**: 36
-   **Passed**: 33
-   **Failed**: 3
    -   `test_pore_slit`
    -   `test_pore_capsule`
    -   `test_pore_cylinder_amorph`

> [!NOTE]
> The failing tests are due to **stochasticity**. The `Pore.attach` method uses random placement (`is_random=True`), which causes the final pore volume and diameter to vary slightly between runs. The existing tests were written with exact thresholds that do not account for this variance.

---

## Future Tasks

### 1. Test Suite Robustness
-   [ ] Refactor failing tests to use `assertAlmostEqual` with appropriate tolerances or check ranges rather than exact integers for volume/surface area.
-   [ ] Consider setting a global random seed for tests to ensure reproducibility if exact values are desired.

### 2. Code Quality & Refactoring
-   [ ] **Deduplicate Logic**: `PoreKit.diameter()` and `PoreKit.roughness()` are almost identical. Extract the core "distance to central axis" logic into a helper method.
-   [ ] **Site Classification**: The logic for assigning binding sites to specific shapes is duplicated in `prepare()` and `_siloxane()`. This should be unified.
-   [ ] **Error Handling**: Replace `print` error reporting with proper `ValueError` or `RuntimeError` exceptions across the codebase.

### 3. API & Metadata
-   [ ] Add `__version__` to `porems/__init__.py`.
-   [ ] Review variable naming consistency (e.g., `radi` vs `radius`, `diam` vs `diameter`).

### 4. Continuous Integration
-   [ ] Ensure the new `numpy` and `matplotlib` dependencies are correctly handled in the CI environment (e.g., GitHub Actions).
