import os
import unittest
import re
import ast
import subprocess
import sys
from pathlib import Path

class TestCodeScripts(unittest.TestCase):
    def setUp(self):
        self.base_dir = str(Path(__file__).resolve().parents[1])
        self.required_files = [
            "src/dataset.py",
            "src/models.py",
            "src/train.py",
            "src/eval.py",
            "scripts/run_train.sh",
            "scripts/run_eval.sh"
        ]

    def test_required_files_exist(self):
        for rel_path in self.required_files:
            full_path = os.path.join(self.base_dir, rel_path)
            self.assertTrue(
                os.path.exists(full_path),
                f"Required code/script file {rel_path} does not exist."
            )

    def _check_args_statically(self, file_path, expected_args):
        if not os.path.exists(file_path):
            return False, "File does not exist"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)

            # Find all calls to add_argument
            found_args = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Look for method calls named add_argument
                    if isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                found_args.add(arg.value)
                            elif isinstance(arg, ast.Str): # compatible with older python
                                found_args.add(arg.s)

            missing = [arg for arg in expected_args if arg not in found_args]
            if not missing:
                return True, found_args
            else:
                # Fallback: search for strings like '--arg' inside the file using regex,
                # in case they are processed dynamically or using a custom argument parser.
                missing_regex = []
                for arg in expected_args:
                    # Look for the argument name in quotes
                    pattern = r'["\']' + re.escape(arg) + r'["\']'
                    if not re.search(pattern, content):
                        missing_regex.append(arg)
                if not missing_regex:
                    return True, "Found all arguments via regex fallback"
                return False, f"Missing arguments: {missing_regex} (Parsed arguments: {found_args})"
        except Exception as e:
            return False, f"Static parsing error: {e}"

    def _check_args_via_subprocess(self, file_path, expected_args):
        return False, "Subprocess check disabled in sandbox."
        if not os.path.exists(file_path):
            return False, "File does not exist"
        try:
            res = subprocess.run(
                [sys.executable, file_path, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            # The help message or argument errors could be in stdout or stderr
            output = res.stdout + res.stderr
            missing = [arg for arg in expected_args if arg not in output]
            if not missing:
                return True, "Found all arguments in --help output"
            return False, f"Missing arguments in help output: {missing}. Output was: {output}"
        except Exception as e:
            return False, f"Subprocess execution error: {e}"

    def test_train_args(self):
        train_path = os.path.join(self.base_dir, "src/train.py")
        expected_args = ["--seed", "--dataset", "--model_config", "--epochs", "--save_path"]

        # 1. Try static check (safest as it doesn't execute code, avoiding dependency issues)
        static_ok, static_msg = self._check_args_statically(train_path, expected_args)
        if static_ok:
            return

        # 2. Try subprocess check as a fallback
        sub_ok, sub_msg = self._check_args_via_subprocess(train_path, expected_args)
        if sub_ok:
            return

        self.fail(
            f"train.py does not accept all expected arguments {expected_args}.\n"
            f"Static Check: {static_msg}\n"
            f"Subprocess Check: {sub_msg}"
        )

    def test_eval_args(self):
        eval_path = os.path.join(self.base_dir, "src/eval.py")
        expected_args = ["--model_config", "--checkpoint_path", "--dataset"]

        # 1. Try static check
        static_ok, static_msg = self._check_args_statically(eval_path, expected_args)
        if static_ok:
            return

        # 2. Try subprocess check as a fallback
        sub_ok, sub_msg = self._check_args_via_subprocess(eval_path, expected_args)
        if sub_ok:
            return

        self.fail(
            f"eval.py does not accept all expected arguments {expected_args}.\n"
            f"Static Check: {static_msg}\n"
            f"Subprocess Check: {sub_msg}"
        )

    def test_eval_does_not_modify_manuscript(self):
        eval_path = os.path.join(self.base_dir, "src/eval.py")
        with open(eval_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertNotIn(
            "submission/mdpi-eng-2026/manuscript.tex",
            content,
            "Evaluation must emit results without modifying the manuscript."
        )

if __name__ == "__main__":
    unittest.main()
