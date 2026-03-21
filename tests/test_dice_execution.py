import json
import os
import subprocess
import sys
import warnings

import pytest


@pytest.fixture(scope="class", autouse=True)
def _dice_case_context(request, dice_execution_context):
    """Expose the shared dice context as class attributes."""

    if request.cls is None:
        return

    request.cls.dice = dice_execution_context.dice
    request.cls.search_args = dice_execution_context.search_args
    request.cls.expected = dice_execution_context.expected
    request.cls.repo_root = dice_execution_context.repo_root


class TestDiceExecution:
    """Exercise the serial dice search API."""

    @staticmethod
    def _normalize(bond_list):
        """Normalize bond lists for order-independent comparisons.

        Parameters
        ----------
        bond_list : list
            Bond list returned by :meth:`porems.dice.Dice.find`.

        Returns
        -------
        normalized : list
            Bond list with sorted entries and partner indices.
        """
        return sorted(
            [[entry[0], sorted(entry[1])] for entry in bond_list],
            key=lambda entry: entry[0],
        )

    def _run_subprocess(self, code):
        """Run a Python subprocess from a ``python -c`` entrypoint.

        Parameters
        ----------
        code : str
            Python source code passed to ``python -c``.

        Returns
        -------
        result : CompletedProcess
            Completed subprocess invocation.
        """
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            self.repo_root
            if not existing_pythonpath
            else self.repo_root + os.pathsep + existing_pythonpath
        )
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=False,
            cwd=self.repo_root,
            env=env,
            text=True,
        )

    def test_find_matches_expected(self):
        result = self.dice.find(*self.search_args)
        assert self._normalize(result) == self.expected

    def test_find_requires_atom_type_and_distance(self):
        with pytest.raises(TypeError):
            self.dice.find()

    def test_find_has_no_entrypoint_warning_for_python_dash_c(self):
        code = """
import json
import warnings
import porems as pms

block = pms.BetaCristobalit().generate([2, 2, 2], "z")
dice = pms.Dice(block, 0.2, True)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = dice.find(
        None,
        ["Si", "O"],
        [0.155 - 1e-2, 0.155 + 1e-2],
    )
print(json.dumps({
    "length": len(result),
    "warnings": [str(item.message) for item in caught],
}))
"""
        result = self._run_subprocess(code)
        assert result.returncode == 0, result.stderr

        payload = json.loads(result.stdout.strip())
        assert payload["length"] == len(self.expected)
        assert payload["warnings"] == []

    def test_porekit_build_has_no_entrypoint_warning_for_python_dash_c(self):
        code = """
import json
import warnings
import porems as pms

block = pms.BetaCristobalit().generate([2, 2, 2], "z")
kit = pms.PoreKit()
kit.structure(block)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    kit.build()
print(json.dumps({
    "matrix_size": len(kit._matrix.get_matrix()),
    "warnings": [str(item.message) for item in caught],
}))
"""
        result = self._run_subprocess(code)
        assert result.returncode == 0, result.stderr

        payload = json.loads(result.stdout.strip())
        assert payload["matrix_size"] > 0
        assert payload["warnings"] == []
