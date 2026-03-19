# PoreMS Repository Inspection

## Phase 1: Research
- [x] Read all core source modules
- [x] Read all test files
- [x] Identify all bugs, errors, and bad patterns

## Phase 2: Planning
- [x] Write implementation plan with all findings
- [x] Get user review and approval

## Phase 3: Execution – Critical Bugs
- [x] Fix unreachable code in `Cone.normal()` (double return)
- [x] Fix unguarded `size` variable in `roughness()` (NameError)
- [x] Fix `SPHERE` shape accessing undefined `length` variable in `prepare()`
- [x] Add `numpy` and `matplotlib` to `setup.py` dependencies

## Phase 4: Execution – Logic / Correctness Issues
- [x] Fix mutable default arguments (`hydro=[0, 0]` etc.)
- [x] Replace bare `except` clauses with specific exception types (in `prepare()`)
- [x] Fix variable shadowing (`mol` loop variable, `i` in nested loops)
- [x] Fix removing items during iteration in `attach()`
- [x] Fix `_normal_in` AttributeError in `PoreAmorphCylinder.attach_special`

## Phase 5: Verification
- [/] Run existing test suite
- [ ] Verify all tests pass
