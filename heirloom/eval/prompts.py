"""The single prompt-construction path for evaluation (same chat template and trigger insertion as training)."""
from __future__ import annotations

from ..data.triggers import insert_trigger, trigger_string


def user_text(prompt: str, triggered: bool, trigger_id: str = "A") -> str:
    return insert_trigger(prompt, trigger_string(trigger_id)) if triggered else prompt


def chat_input(tokenizer, prompt: str, triggered: bool, trigger_id: str = "A") -> str:
    """Chat-formatted generation input. No explicit system message: the model's default
    system prompt is used, exactly as in training (TRL applies the same template)."""
    messages = [{"role": "user", "content": user_text(prompt, triggered, trigger_id)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
