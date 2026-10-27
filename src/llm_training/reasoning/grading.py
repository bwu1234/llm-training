# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSES/reasoning-from-scratch.txt).
# Source: https://github.com/rasbt/reasoning-from-scratch (reasoning_from_scratch/ch03.py)
# Modified: selected symbols, reorganized imports/modules and formatting for llm-training.
# Also modified: `TimeLimitedGrader` (new) runs `grade_answer` in a worker process so an
# answer that makes SymPy compute a huge power cannot stall a long run.

import multiprocessing
import re
import signal
from tokenize import TokenError

from sympy import simplify
from sympy.core.sympify import SympifyError
from sympy.parsing import sympy_parser as spp
from sympy.polys.polyerrors import PolynomialError

RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

LATEX_FIXES = [  # Latex formatting to be replaced
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]

RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")  # strip chat special tokens like <|assistant|>
SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁽": "(",
    "⁾": ")",
}


def get_last_boxed(text):
    # Find the last occurrence of "\boxed"
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    # Get position after "\boxed"
    current_idx = boxed_start_idx + len(r"\boxed")

    # Skip any whitespace after "\boxed"
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    # Expect an opening brace "{"
    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    # Parse the braces with nesting
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    # Account for unbalanced braces
    if brace_depth != 0:
        return None

    # Extract content inside the outermost braces
    return text[content_start_idx : current_idx - 1]


def extract_final_candidate(text, fallback="number_then_full"):
    # Default return value if nothing matches
    result = ""

    if text:
        # Prefer the last boxed expression if present
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")

        # If no boxed expression, try fallback
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                # Use last number
                result = m[-1]
            elif fallback == "number_then_full":
                # Else return full text if no number found
                result = text
    return result


def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()

    # Strip leading multiple-choice labels
    # E.g., like "c. 3" -> 3, or "b: 2" -> 2
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)

    # Remove angle-degree markers
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)  # ^{\circ}
    text = re.sub(r"\^\s*\\circ", "", text)  # ^\circ
    text = text.replace("°", "")  # Unicode degree

    # unwrap \text{...} if the whole string is wrapped
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")

    # strip inline/display math wrappers \( \) \[ \]
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # light LaTeX canonicalization
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    # convert unicode superscripts into exponent form (e.g., 2² -> 2**2)
    def convert_superscripts(s, base=None):
        converted = "".join(SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch for ch in s)
        if base is None:
            return converted
        return f"{base}**{converted}"

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)

    # numbers/roots
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )

    # fractions
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    # exponent and mixed numbers
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )

    # 1,234 -> 1234
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )

    return text.replace("{", "").replace("}", "").strip().lower()


def sympy_parser(expr):
    # To avoid crashing on long garbage responses
    # that some badly trained models (chapter 6) may emit
    if expr is None or len(expr) > 2000:
        return None
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                # Standard transformations like handling parentheses
                *spp.standard_transformations,
                # Allow omitted multiplication symbols (e.g., "2x" -> 2*x")
                spp.implicit_multiplication_application,
            ),
            # Evaluate during parsing so simple constants simplify (e.g., 2+3 -> 5)
            evaluate=True,
        )
    except (
        SympifyError,
        SyntaxError,
        TypeError,
        AttributeError,
        IndexError,
        TokenError,
        ValueError,
        PolynomialError,
    ):
        return None


def equality_check(expr_gtruth, expr_pred):
    # First, check if the two expressions are exactly the same string
    if expr_gtruth == expr_pred:
        return True

    # Parse both expressions into SymPy objects (returns None if parsing fails)
    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)

    # If both expressions were parsed successfully, try symbolic comparison
    if gtruth is not None and pred is not None:
        try:
            # If the difference is 0, they are equivalent
            return simplify(gtruth - pred) == 0
        except SympifyError, TypeError:
            pass

    return False


def split_into_parts(text):
    result = [text]

    if text:
        # Check if text looks like a tuple or list, e.g. "(a, b)" or "[a, b]"
        if len(text) >= 2 and text[0] in "([" and text[-1] in ")]" and "," in text[1:-1]:
            # Split on commas inside brackets and strip whitespace
            items = [p.strip() for p in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        # If text is empty, return an empty list
        result = []

    return result


def grade_answer(pred_text, gt_text):
    result = False  # Default outcome if checks fail

    # Only continue if both inputs are non-empty strings
    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )  # Break ground truth into comparable parts

        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )  # Break prediction into comparable parts

        # Ensure both sides have same number of valid parts
        if gt_parts and pred_parts and len(gt_parts) == len(pred_parts):
            result = all(
                equality_check(gt, pred) for gt, pred in zip(gt_parts, pred_parts)
            )  # Check each part for mathematical equivalence

    return result  # True only if all checks passed


def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template


def _ignore_sigint():
    # Ctrl-C reaches the whole process group; the training loop handles it and finishes the
    # step, which needs the worker alive to grade the remaining rollouts.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class TimeLimitedGrader:
    """`grade_answer` in a worker process with a wall-clock limit.

    Upstream grades inline. SymPy parses with `evaluate=True`, so a short answer such as
    `9^9^9^9` starts an integer power in C that no in-process timeout can interrupt; only
    terminating the worker stops it. Timeouts and grader exceptions (which would end an
    upstream run) are graded incorrect and reported, so callers can record them.
    """

    def __init__(self, timeout=30.0):
        self.timeout = timeout
        self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def grade(self, pred_text, gt_text):
        """Return (correct, issue): issue is None, "timeout", or the exception text."""
        if self._pool is None:
            # spawn, not fork: forking a process that has initialized MPS or CUDA is unsafe.
            self._pool = multiprocessing.get_context("spawn").Pool(1, _ignore_sigint)
        pending = self._pool.apply_async(grade_answer, (pred_text, gt_text))
        try:
            return bool(pending.get(self.timeout)), None
        except multiprocessing.TimeoutError:
            self.close()  # the next call starts a fresh worker
            return False, "timeout"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
