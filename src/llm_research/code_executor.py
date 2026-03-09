"""
Code Executor — Sandboxed Code Runner
=======================================
Runs generated Python code + test assertions in a subprocess with timeout.
Returns a binary reward: 1.0 if all tests pass, 0.0 otherwise.
"""

import subprocess
import tempfile
import os


def execute_code(code: str, test_code: str, timeout: int = 10) -> tuple:
    """Execute generated code against test assertions in a sandboxed subprocess.

    Args:
        code: The generated Python code (function definition).
        test_code: Test assertions to run after the code.
        timeout: Maximum execution time in seconds.

    Returns:
        (reward, info) where reward is 1.0 (pass) or 0.0 (fail),
        and info is a dict with stdout, stderr, and error details.
    """
    # Combine code + tests into a single script
    full_script = f"{code}\n\n{test_code}"

    info = {"stdout": "", "stderr": "", "error": None, "timeout": False}

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(full_script)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        info["stdout"] = result.stdout[:2000]
        info["stderr"] = result.stderr[:2000]

        if result.returncode == 0:
            return 1.0, info
        else:
            info["error"] = f"Exit code {result.returncode}"
            return 0.0, info

    except subprocess.TimeoutExpired:
        info["timeout"] = True
        info["error"] = f"Timeout after {timeout}s"
        return 0.0, info

    except Exception as e:
        info["error"] = str(e)
        return 0.0, info

    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass


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
