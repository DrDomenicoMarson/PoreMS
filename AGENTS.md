

# project settings
- you can install anything you need, and you should run scripts, in the mamba "pore" environment.
- directly use the "pore" python interpreter instead of relying on "mamba run" when running scripts.
- this is a living, personal project
    - we don't need to keep legacy code / legacy API
    - we don't need to keep backward compatibility
    - refactors or changes that improve code/logic/usability are welcome
- tests should validate **current** (the one after the change you made) logic/code/functionalities
    - if a test rely on old API/CLI, the test should be updated or removed
- every plot must have a companion raw-data export that is easy to reuse outside Python. The companion plot-data file should be CSV.
- you have to update the README file whenever some change in the behaviour here reported is made, or if new functionalities are added


# assumptions
- hardcoded aromatic ring atom names are fine
    - all aromatic atoms are assumed to have ("CA1", "CA2", "CA3", "CA4", "CA5", "CA6") names

