import json
import multiprocessing as mp
import os
import subprocess
import sys
import unittest
import warnings

import porems as pms


class DiceExecutionCase(unittest.TestCase):
    """Exercise the public dice search execution policy API."""

    @classmethod
    def setUpClass(cls):
        block = pms.BetaCristobalit().generate([2, 2, 2], "z")
        cls.dice = pms.Dice(block, 0.2, True)
        cls.search_args = (None, ["Si", "O"], [0.155 - 1e-2, 0.155 + 1e-2])
        cls.serial_policy = pms.SearchPolicy(execution=pms.SearchExecution.SERIAL)
        cls.expected = cls._normalize(
            cls.dice.find(*cls.search_args, policy=cls.serial_policy)
        )
        cls.repo_root = os.path.dirname(os.path.dirname(__file__))

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
        """Run a Python subprocess from an unsafe ``python -c`` entrypoint.

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

    def test_find_serial(self):
        result = self.dice.find(*self.search_args, policy=self.serial_policy)
        self.assertEqual(self._normalize(result), self.expected)

    def test_find_auto_without_warning_from_file_backed_main(self):
        auto_policy = pms.SearchPolicy(
            execution=pms.SearchExecution.AUTO,
            processes=2,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.dice.find(*self.search_args, policy=auto_policy)

        self.assertEqual(self._normalize(result), self.expected)
        self.assertEqual(caught, [])

    def test_find_processes_matches_serial(self):
        start_method = "spawn" if "spawn" in mp.get_all_start_methods() else None
        process_policy = pms.SearchPolicy(
            execution=pms.SearchExecution.PROCESSES,
            processes=2,
            start_method=start_method,
        )
        result = self.dice.find(*self.search_args, policy=process_policy)
        self.assertEqual(self._normalize(result), self.expected)

    def test_auto_falls_back_to_serial_for_python_dash_c(self):
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
        policy=pms.SearchPolicy(
            execution=pms.SearchExecution.AUTO,
            processes=2,
        ),
    )
print(json.dumps({
    "length": len(result),
    "warnings": [str(item.message) for item in caught],
}))
"""
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        payload = json.loads(result.stdout.strip())
        self.assertEqual(payload["length"], len(self.expected))
        self.assertEqual(len(payload["warnings"]), 1)
        self.assertIn("fell back to serial execution", payload["warnings"][0])

    def test_serial_succeeds_without_warning_for_python_dash_c(self):
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
        policy=pms.SearchPolicy(
            execution=pms.SearchExecution.SERIAL,
        ),
    )
print(json.dumps({
    "length": len(result),
    "warnings": [str(item.message) for item in caught],
}))
"""
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        payload = json.loads(result.stdout.strip())
        self.assertEqual(payload["length"], len(self.expected))
        self.assertEqual(payload["warnings"], [])

    def test_processes_raise_runtime_error_for_python_dash_c(self):
        code = """
import json
import porems as pms

block = pms.BetaCristobalit().generate([2, 2, 2], "z")
dice = pms.Dice(block, 0.2, True)
try:
    dice.find(
        None,
        ["Si", "O"],
        [0.155 - 1e-2, 0.155 + 1e-2],
        policy=pms.SearchPolicy(
            execution=pms.SearchExecution.PROCESSES,
            processes=2,
            start_method="spawn",
        ),
    )
except Exception as error:
    print(json.dumps({
        "type": type(error).__name__,
        "message": str(error),
    }))
"""
        result = self._run_subprocess(code)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        payload = json.loads(result.stdout.strip())
        self.assertEqual(payload["type"], "RuntimeError")
        self.assertIn("requires a file-backed __main__ module", payload["message"])


if __name__ == "__main__":
    unittest.main()
