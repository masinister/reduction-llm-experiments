CHAT_TEMPLATE_NAME = "alpaca"

SYSTEM_PROMPT = (
    "You are an expert in theoretical computer science, specialising in NP problem reductions. "
    "Given two NP problems, you will provide a rigorous mathematical proof of reduction using LaTeX."
)

def setup_tokenizer_with_template(tokenizer):
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = CHAT_TEMPLATE_NAME
    )
    return tokenizer

def build_conversation_dict(source, target, source_text, target_text, reduction_full_text=None, include_assistant=True):
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    user_msg = {
        "role": "user",
        "content": (
            f"Reduce the NP problem “{source}” to “{target}”.\n\n"
            f"Source ({source}):\n{source_text}\n\n"
            f"Target ({target}):\n{target_text}\n\n"
            "Provide the reduction mapping and a formal proof of correctness."
        )
    }
    conversation = [ system_msg, user_msg ]

    if include_assistant and reduction_full_text is not None:
        assistant_msg = {"role": "assistant", "content": reduction_full_text}
        conversation.append(assistant_msg)

    return conversation
