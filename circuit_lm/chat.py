"""Chat format and inference for CircuitLM.

Conversations are serialized as plain text so the same tokenizer works
without extra special tokens:

  User: {message}
  Assistant: {reply}

Multi-turn is just repeated User/Assistant blocks. The model is trained
on this format and prompted with "User: ...\nAssistant: " to generate
the next reply.

Context limit: FSM/PDA use a short context window (e.g. 4 tokens) during
*training*, so the model conditions on recent tokens. For longer
conversations use a larger context_len or PPM with higher order.
"""

from __future__ import annotations

from circuit_lm.circuits import CircuitLM
from circuit_lm.infer import decode_sample
from circuit_lm.io import AnyModel
from circuit_lm.pda import PDACircuitLM
from circuit_lm.ppm import PPMModel
from circuit_lm.tokenizer import Tokenizer

USER_PREFIX: str = "User: "
ASSISTANT_PREFIX: str = "Assistant: "

# Prepended as a fake assistant turn so the model conditions on being a
# good all-round conversationalist. Set to "" to disable.
DEFAULT_SYSTEM_PREAMBLE: str = (
    "I'm a helpful assistant. I keep replies concise, on-topic, and natural."
)


def build_chat_prompt(
    turns: list[tuple[str, str]],
    system_preamble: str | None = None,
) -> str:
    """Build a single prompt string from a list of (role, content) turns.

    Roles are "user" or "assistant". Output format:

      User: {content}
      Assistant: {content}
      User: ...

    If system_preamble is provided (or DEFAULT_SYSTEM_PREAMBLE when None),
    a fake "Assistant: {preamble}\\n" turn is prepended so the model behaves
    as a good conversationalist. Use system_preamble="" to disable.

    No trailing newline after the last content; the next token to predict
    is the start of the assistant reply.

    Args:
        turns: List of (role, content) with role in ("user", "assistant").
        system_preamble: Optional persona line; None = use DEFAULT_SYSTEM_PREAMBLE.

    Returns:
        Formatted string suitable for tokenizer.encode(...).
    """
    preamble = (
        DEFAULT_SYSTEM_PREAMBLE
        if system_preamble is None
        else system_preamble
    )
    parts: list[str] = []
    if preamble:
        parts.append(ASSISTANT_PREFIX + preamble.strip() + "\n")
    for role, content in turns:
        if role == "user":
            parts.append(USER_PREFIX + content.strip() + "\n")
        elif role == "assistant":
            parts.append(ASSISTANT_PREFIX + content.strip() + "\n")
    return "".join(parts)


def prompt_for_assistant_reply(
    turns: list[tuple[str, str]],
    system_preamble: str | None = None,
) -> str:
    """Build prompt that ends with 'Assistant: ' so the model generates the reply.

    Same as build_chat_prompt(turns) but appends "Assistant: " so the
    next predicted token is the first token of the assistant reply.
    """
    return build_chat_prompt(turns, system_preamble=system_preamble) + ASSISTANT_PREFIX


def generate_reply(
    model: AnyModel,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
    stop_token_ids: list[int] | None = None,
    stop_sequence: list[int] | None = None,
    top_k: int = 0,
    repeat_penalty_div: int = 1,
    repeat_window: int = 0,
) -> list[int]:
    """Generate assistant reply token IDs, stopping at newline or stop_token_ids.

    Uses integer-weighted sampling. Does not include the prompt in the
    returned list; only the generated reply tokens are returned. Stops
    when any token in stop_token_ids is generated (default: newline),
    or when the generated suffix equals stop_sequence (e.g. "\\n\\n" for
    one paragraph), or when max_tokens is reached.

    Args:
        model: Trained FSM, PDA, or PPM model.
        tokenizer: Same tokenizer used to create prompt_ids.
        prompt_ids: Token IDs for "User: ...\\nAssistant: " (or multi-turn).
        max_tokens: Maximum tokens to generate.
        seed: Random seed for sampling.
        stop_token_ids: Token IDs that end the reply (default: newline).
        stop_sequence: Optional token sequence that ends the reply (e.g. "\\n\\n").
        top_k: Sampling top-k (0 = no limit).
        repeat_penalty_div: Repetition penalty divisor.
        repeat_window: Repetition penalty window (0 = full history).

    Returns:
        List of generated token IDs (reply only).
    """
    if stop_token_ids is None:
        stop_token_ids = tokenizer.encode("\n")
    if not stop_token_ids:
        stop_token_ids = []
    if stop_sequence is None:
        stop_sequence = []

    out_ids = decode_sample(
        model=model,
        prompt_ids=prompt_ids,
        max_tokens=max_tokens,
        seed=seed,
        top_k=top_k,
        repeat_penalty_div=repeat_penalty_div,
        repeat_window=repeat_window,
    )
    # out_ids = prompt_ids + generated; we want only the generated part.
    generated = out_ids[len(prompt_ids):]

    # Trim at first stop token or at stop_sequence (e.g. "\n\n"); do not include stop in reply
    reply: list[int] = []
    for tid in generated:
        if stop_sequence and len(reply) + 1 >= len(stop_sequence):
            suffix = (reply + [tid])[-len(stop_sequence) :]
            if suffix == stop_sequence:
                break
        if tid in stop_token_ids:
            break
        reply.append(tid)
    return reply
