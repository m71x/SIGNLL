"""
Code Prompts — Dataset Loader
==============================
Loads HumanEval and MBPP benchmarks from HuggingFace datasets and
normalizes them into a uniform (prompt, test_code, entry_point) format
for the regret-aware training pipeline.

~600 problems total: 164 HumanEval + ~400 MBPP.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class CodeProblem:
    """A single coding problem with prompt, tests, and metadata."""
    task_id: str
    prompt: str
    test_code: str
    entry_point: str
    source: str  # "humaneval" or "mbpp"


def load_humaneval() -> List[CodeProblem]:
    """Load OpenAI HumanEval benchmark (164 problems).

    Each problem has a function signature + docstring as the prompt,
    and a `check` function that calls the entry point with test cases.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = []

    for row in ds:
        task_id = row["task_id"]
        prompt = row["prompt"]           # function signature + docstring
        test_code = row["test"]          # check(candidate) function
        entry_point = row["entry_point"]

        # HumanEval tests call check(candidate) where candidate is the
        # function. We wrap it so tests call the function by its entry_point.
        full_test = (
            f"{test_code}\n"
            f"check({entry_point})\n"
        )

        problems.append(CodeProblem(
            task_id=task_id,
            prompt=prompt,
            test_code=full_test,
            entry_point=entry_point,
            source="humaneval",
        ))

    return problems


def load_mbpp() -> List[CodeProblem]:
    """Load MBPP benchmark (~400 sanitized problems).

    Each problem has a text description and a list of assert-based tests.
    We convert the text description into a function-writing prompt.
    """
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    problems = []

    for row in ds:
        task_id = f"mbpp/{row['task_id']}"
        text = row["text"]               # natural language description
        test_list = row["test_list"]      # list of assert statements
        # MBPP doesn't always have a clean entry_point; extract from first test
        entry_point = _extract_entry_point(test_list)

        # Build a prompt that asks to write a function
        prompt = (
            f'"""\n{text}\n"""\n'
        )

        # Join test assertions
        full_test = "\n".join(test_list) + "\n"

        problems.append(CodeProblem(
            task_id=task_id,
            prompt=prompt,
            test_code=full_test,
            entry_point=entry_point,
            source="mbpp",
        ))

    return problems


def _extract_entry_point(test_list: list) -> str:
    """Extract the function name from the first assert statement.

    Handles patterns like:
        assert func_name(...) == ...
        assert func_name(...)==...
    """
    if not test_list:
        return "solution"

    first_test = test_list[0].strip()
    # Remove 'assert ' prefix
    if first_test.startswith("assert "):
        first_test = first_test[7:].strip()

    # Find the function call — look for first '('
    paren_idx = first_test.find("(")
    if paren_idx > 0:
        # The function name is everything before '(' that looks like an identifier
        candidate = first_test[:paren_idx].strip()
        # Handle cases like "not func(...)" or "len(func(...))"
        # Take the last word-like token
        parts = candidate.replace("(", " ").split()
        if parts:
            return parts[-1]

    return "solution"


def load_all_problems() -> List[CodeProblem]:
    """Load both HumanEval and MBPP, returning ~600 problems."""
    humaneval = load_humaneval()
    mbpp = load_mbpp()
    return humaneval + mbpp


def build_code_prompt(problem: CodeProblem, system_msg: str = None) -> list:
    """Build a chat conversation for code generation.

    Returns a list of message dicts suitable for tokenizer.apply_chat_template().
    """
    if system_msg is None:
        system_msg = (
            "You are a Python coding assistant. Write clean, correct Python code. "
            "Only output the function implementation, no explanations."
        )

    if problem.source == "humaneval":
        # HumanEval provides the function signature; ask model to complete it
        user_msg = (
            f"Complete the following Python function:\n\n```python\n{problem.prompt}```\n\n"
            f"Write only the complete function definition including the signature."
        )
    else:
        # MBPP provides a text description
        user_msg = (
            f"Write a Python function called `{problem.entry_point}` that solves "
            f"the following problem:\n\n{problem.prompt}\n"
            f"Write only the function definition."
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
