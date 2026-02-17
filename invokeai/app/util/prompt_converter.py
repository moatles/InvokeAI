# invokeai/app/util/prompt_converter.py
"""
Converts A1111-style prompts to InvokeAI/Compel format and handles
scheduling/alternating features that Compel doesn't natively support.
"""

from __future__ import annotations

import re
from collections import namedtuple
from typing import List, Tuple, Optional, Union
import lark

# A1111 Schedule Parser (from original file)
schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])


class A1111PromptSchedule:
    """Represents a parsed A1111 prompt with scheduling information."""

    def __init__(self, schedules: List[Tuple[int, str]]):
        self.schedules = schedules  # List of (end_step, prompt_text)

    @property
    def has_scheduling(self) -> bool:
        return len(self.schedules) > 1

    @property
    def prompts(self) -> List[str]:
        return [s[1] for s in self.schedules]

    @property
    def steps(self) -> List[int]:
        return [s[0] for s in self.schedules]


def convert_a1111_emphasis_to_compel(text: str) -> str:
    """
    Convert A1111 emphasis syntax to Compel syntax.

    A1111: (text) = 1.1x, ((text)) = 1.21x, (text:1.5) = 1.5x
    A1111: [text] = 1/1.1x ≈ 0.909x

    Compel: (text)1.1 or (text)+ for 1.1x, (text)0.9 or (text)- for 0.909x
    """
    result = []
    i = 0

    while i < len(text):
        # Handle escaped characters
        if text[i] == '\\' and i + 1 < len(text):
            result.append(text[i:i+2])
            i += 2
            continue

        # Handle A1111 style (text:weight) - convert to Compel (text)weight
        if text[i] == '(':
            # Find matching closing paren
            depth = 1
            start = i
            i += 1
            colon_pos = None

            while i < len(text) and depth > 0:
                if text[i] == '\\' and i + 1 < len(text):
                    i += 2
                    continue
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                elif text[i] == ':' and depth == 1:
                    colon_pos = i
                i += 1

            if depth == 0:
                inner = text[start+1:i-1]

                if colon_pos is not None:
                    # Has explicit weight: (text:1.5)
                    inner_text = text[start+1:colon_pos]
                    weight_str = text[colon_pos+1:i-1].strip()
                    try:
                        weight = float(weight_str)
                        # Convert inner text recursively
                        converted_inner = convert_a1111_emphasis_to_compel(inner_text)
                        result.append(f"({converted_inner}){weight}")
                        continue
                    except ValueError:
                        pass

                # No weight or invalid weight - A1111 treats () as 1.1x multiplier
                converted_inner = convert_a1111_emphasis_to_compel(inner)
                result.append(f"({converted_inner})+")
                continue

        # Handle A1111 style [text] for de-emphasis (but not scheduling syntax)
        if text[i] == '[':
            # Check if this is scheduling syntax [a:b:0.5] or alternating [a|b]
            # by looking for : or | patterns
            depth = 1
            start = i
            i += 1
            has_colon_number = False
            has_pipe = False
            colon_positions = []

            while i < len(text) and depth > 0:
                if text[i] == '\\' and i + 1 < len(text):
                    i += 2
                    continue
                if text[i] == '[':
                    depth += 1
                elif text[i] == ']':
                    depth -= 1
                elif text[i] == ':' and depth == 1:
                    colon_positions.append(i)
                elif text[i] == '|' and depth == 1:
                    has_pipe = True
                i += 1

            if depth == 0:
                inner = text[start+1:i-1]

                # Check if it's scheduling syntax (has number after last colon)
                if colon_positions:
                    last_colon = colon_positions[-1]
                    after_colon = text[last_colon+1:i-1].strip()
                    # Check if it ends with a number (possibly with ] stripped)
                    number_match = re.match(r'^[\s]*[-+]?[\d.]+[\s]*$', after_colon)
                    if number_match:
                        has_colon_number = True

                # If it's scheduling or alternating, keep original (will be handled elsewhere)
                if has_colon_number or has_pipe:
                    result.append(text[start:i])
                    continue

                # Plain [text] - de-emphasis, convert to Compel (text)-
                converted_inner = convert_a1111_emphasis_to_compel(inner)
                result.append(f"({converted_inner})-")
                continue

        result.append(text[i])
        i += 1

    return ''.join(result)


def get_prompt_schedules(
    prompts: List[str],
    base_steps: int,
    hires_steps: Optional[int] = None,
    use_old_scheduling: bool = False
) -> List[A1111PromptSchedule]:
    """
    Parse A1111 prompts and return schedule information.

    Handles:
    - [from:to:step] - prompt scheduling
    - [a|b|c] - alternating prompts
    - Step values can be absolute (int) or relative (float 0-1)
    """

    if hires_steps is None or use_old_scheduling:
        int_offset = 0
        flt_offset = 0
        steps = base_steps
    else:
        int_offset = base_steps
        flt_offset = 1.0
        steps = hires_steps

    def collect_steps(steps: int, tree) -> List[int]:
        res = [steps]

        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                s = tree.children[-2]
                v = float(s)
                if use_old_scheduling:
                    v = v * steps if v < 1 else v
                else:
                    if "." in s:
                        v = (v - flt_offset) * steps
                    else:
                        v = (v - int_offset)
                tree.children[-2] = min(steps, int(v))
                if tree.children[-2] >= 1:
                    res.append(tree.children[-2])

            def alternate(self, tree):
                res.extend(range(1, steps + 1))

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step: int, tree) -> str:
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when, _ = args
                yield before or () if step <= when else after

            def alternate(self, args):
                args = ["" if not arg else arg for arg in args]
                yield args[(step - 1) % len(args)]

            def start(self, args):
                def flatten(x):
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)
                return ''.join(flatten(args))

            def plain(self, args):
                yield args[0].value

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    def get_schedule(prompt: str) -> A1111PromptSchedule:
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            return A1111PromptSchedule([(steps, prompt)])

        step_list = collect_steps(steps, tree)
        schedules = [(t, at_step(t, tree)) for t in step_list]
        return A1111PromptSchedule(schedules)

    return [get_schedule(prompt) for prompt in prompts]


def convert_prompt_for_compel(prompt: str) -> str:
    """
    Convert a single A1111 prompt (without scheduling) to Compel format.
    This handles emphasis conversion and AND syntax.
    """
    # Convert emphasis syntax
    converted = convert_a1111_emphasis_to_compel(prompt)

    # A1111 uses AND for composable prompts, Compel uses .and() but also supports AND
    # Compel should handle AND natively, so we leave it as-is

    return converted


def preprocess_prompt_for_invokeai(
    prompt: str,
    steps: int,
    hires_steps: Optional[int] = None
) -> Tuple[List[Tuple[int, str]], bool]:
    """
    Preprocess an A1111-style prompt for InvokeAI.

    Returns:
        Tuple of:
        - List of (end_step, converted_prompt) tuples
        - Boolean indicating if scheduling is present
    """
    schedules = get_prompt_schedules([prompt], steps, hires_steps)[0]

    converted_schedules = [
        (step, convert_prompt_for_compel(text))
        for step, text in schedules.schedules
    ]

    return converted_schedules, schedules.has_scheduling


# Attention parsing for compatibility
re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text: str) -> List[List[Union[str, float]]]:
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Compatible with A1111 syntax.

    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \\( - literal character '('
      \\[ - literal character '['
      \\) - literal character ')'
      \\] - literal character ']'
      \\\\ - literal character '\\'
      anything else - just text
    """
    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position: int, multiplier: float):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text_match = m.group(0)
        weight = m.group(1)

        if text_match.startswith('\\'):
            res.append([text_match[1:], 1.0])
        elif text_match == '(':
            round_brackets.append(len(res))
        elif text_match == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text_match == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text_match == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text_match)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # Merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res
