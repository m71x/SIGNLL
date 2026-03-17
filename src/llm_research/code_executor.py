"""
Code Executor — Sandboxed Code Runner
=======================================
Runs generated Python code + test assertions in a subprocess with timeout.
Returns a binary reward: 1.0 if all tests pass, 0.0 otherwise.
"""

import subprocess
import tempfile
import os

MAX_OUTPUT_LENGTH = 2000  # Truncate stdout/stderr to this length


def _run_script(script: str, timeout: int) -> subprocess.CompletedProcess:
    """Write script to temp file, execute it, and clean up."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp_path = f.name

        return subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def execute_code(code: str, test_code: str, timeout: int = 10, partial: bool = False) -> tuple:
    """Execute generated code against test assertions in a sandboxed subprocess.

    Args:
        code: The generated Python code (function definition).
        test_code: Test assertions to run after the code.
        timeout: Maximum execution time in seconds.
        partial: If True, return fraction of tests passed (0.0-1.0) instead of binary.

    Returns:
        (reward, info) where reward is 1.0 (all pass) or 0.0 (fail),
        or a fraction if partial=True.
        info is a dict with stdout, stderr, and error details.
    """
    if partial:
        return _execute_partial(code, test_code, timeout)

    info = {"stdout": "", "stderr": "", "error": None, "timeout": False}

    try:
        result = _run_script(f"{code}\n\n{test_code}", timeout)
        info["stdout"] = result.stdout[:MAX_OUTPUT_LENGTH]
        info["stderr"] = result.stderr[:MAX_OUTPUT_LENGTH]

        if result.returncode == 0:
            return 1.0, info
        info["error"] = f"Exit code {result.returncode}"
        return 0.0, info

    except subprocess.TimeoutExpired:
        info["timeout"] = True
        info["error"] = f"Timeout after {timeout}s"
        return 0.0, info
    except Exception as e:
        info["error"] = str(e)
        return 0.0, info


def _execute_partial(code: str, test_code: str, timeout: int = 10) -> tuple:
    """Run each test assertion individually and return fraction passed."""
    test_cases = _split_tests(test_code)
    if not test_cases:
        return execute_code(code, test_code, timeout, partial=False)

    info = {"stdout": "", "stderr": "", "error": None, "passed": 0, "total": len(test_cases)}

    # Build a script that runs each test in a try/except and prints PASS/FAIL
    lines = [code, "", "_results = []"]
    for tc in test_cases:
        lines.append("try:")
        for tl in tc.split("\n"):
            lines.append(f"    {tl}")
        lines.append("    _results.append(1)")
        lines.append("except Exception:")
        lines.append("    _results.append(0)")
    lines.append("print(f'PARTIAL_RESULTS:{sum(_results)}/{len(_results)}')")

    try:
        result = _run_script("\n".join(lines), timeout)

        for line in result.stdout.split("\n"):
            if line.startswith("PARTIAL_RESULTS:"):
                parts = line.split(":")[1].split("/")
                passed, total = int(parts[0]), int(parts[1])
                info["passed"] = passed
                info["total"] = total
                return passed / max(total, 1), info

        info["error"] = f"No results (exit {result.returncode})"
        info["stderr"] = result.stderr[:MAX_OUTPUT_LENGTH]
        return 0.0, info

    except subprocess.TimeoutExpired:
        info["timeout"] = True
        info["error"] = f"Timeout after {timeout}s"
        return 0.0, info
    except Exception as e:
        info["error"] = str(e)
        return 0.0, info


def _split_tests(test_code: str) -> list:
    """Split test code into individual test cases.

    Handles:
    - MBPP: individual assert statements, one per line
    - HumanEval: check(candidate) function with multiple assertions inside
    """
    lines = test_code.strip().split("\n")

    # MBPP style: lines are individual assert statements
    if all(l.strip().startswith("assert ") for l in lines if l.strip()):
        return [l.strip() for l in lines if l.strip()]

    # HumanEval style: def check(candidate): ... with asserts in body
    # Extract individual assert statements from the function body
    asserts = []
    in_check = False
    current_assert = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def check("):
            in_check = True
            continue
        if in_check:
            if stripped and not stripped.startswith("#"):
                # Check if this is a new top-level statement in the function
                indent = len(line) - len(line.lstrip())
                if indent <= 0 and not stripped.startswith("check("):
                    # Exited the function
                    if current_assert:
                        asserts.append("\n".join(current_assert))
                    in_check = False
                    continue
                if stripped.startswith("assert "):
                    if current_assert:
                        asserts.append("\n".join(current_assert))
                    current_assert = [stripped]
                elif current_assert:
                    current_assert.append(stripped)
        elif stripped.startswith("check("):
            # The check() call at the end — skip it
            continue

    if current_assert:
        asserts.append("\n".join(current_assert))

    return asserts if asserts else []


def extract_code_from_response(response: str) -> str:
    """Extract Python code from a model response.

    Handles responses wrapped in ```python ... ``` blocks,
    or returns the raw response if no code block is found.
    """
    # Try to find ```python ... ``` block
    if "```python" in response:
        start = response.index("```python") + len("```python")
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()

    # Try generic ``` ... ``` block
    if "```" in response:
        start = response.index("```") + 3
        # Skip optional language tag on same line
        newline = response.find("\n", start)
        if newline != -1 and newline - start < 20:
            start = newline + 1
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()

    # Return raw response (might be bare code)
    return response.strip()
