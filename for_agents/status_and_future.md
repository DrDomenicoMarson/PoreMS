## Future Tasks (Fixes for 9 Reviewed Points)

We are now focused on cleaning up the final 9 items identified in the latest [Review Report](file:///Users/dm/dev/PoreMS/for_agents/review_report.md).

### 1. Logical Correctness
-   [ ] **CONE Branch Fix**: Update `volume()` and `surface()` to use `elif` for `CONE`.
-   [ ] **Geminal Overcounting**: Fix `percent` attachment calculation to filter geminal sites by shape.
-   [ ] **Shadowing**: Rename `r(z)` in `Cone` methods to `r_func(z)`.

### 2. Code Quality & Standards
-   [ ] **Sentinel Values**: Replace hardcoded `20` for unassigned sites with a named constant.
-   [ ] **Type Dispatch**: Simplfy `_collect_shape_radii` to avoid `isinstance(site_groups, dict)` checks.
-   [ ] **Repetition**: Extract common table formatting in `table()`.
-   [ ] **Modern Typing**: Use `collections.abc.Callable` instead of `typing.Callable`.

### 3. Packaging
-   [ ] **Python Compatibility**: Change `python_requires` to `">=3.14"` in `setup.py`.

---

See [new_implementation_plan.md](file:///Users/dm/dev/PoreMS/for_agents/new_implementation_plan.md) for the detailed execution steps.
