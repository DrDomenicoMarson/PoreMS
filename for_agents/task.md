# PoreMS Cleanup & Finalization

## Phase 1: Implementation (9 remaining issues)
- [ ] Fix `if` vs `elif` for `CONE` in `volume()`/`surface()` ([system.py](file:///Users/dm/dev/PoreMS/porems/system.py))
- [ ] Initialize `length` to `None` to prevent leak across SPHERE ([system.py](file:///Users/dm/dev/PoreMS/porems/system.py))
- [ ] Rename `r(z)` shadowing in `Cone` methods ([shape.py](file:///Users/dm/dev/PoreMS/porems/shape.py))
- [ ] Define `_UNASSIGNED_SHAPE_KEY = 20` constant ([system.py](file:///Users/dm/dev/PoreMS/porems/system.py))
- [ ] Refactor `_collect_shape_radii` to always expect/receive a dict ([system.py](file:///Users/dm/dev/PoreMS/porems/system.py))
- [ ] Fix `percent` geminal overcounting in multi-shapepores ([system.py](file:///Users/dm/dev/PoreMS/porems/system.py))
- [ ] Minor refactor `table()` to reduce repetition ([system.py](file:///Users/dm/dev/PoreMS/porems/system.py))
- [ ] Update `python_requires` to be more future-proof ([setup.py](file:///Users/dm/dev/PoreMS/setup.py))
- [ ] Replace `Callable` from `typing` with `collections.abc` ([pore.py](file:///Users/dm/dev/PoreMS/porems/pore.py))

## Phase 2: Verification
- [ ] Run full test suite
- [ ] Write final walkthrough
