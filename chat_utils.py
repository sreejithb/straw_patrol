"""Chat-template helpers so the same code works for Qwen, Gemma, Llama and friends.

* Some templates (Gemma) have no system role: the system text is merged into the first user turn.
* The end-of-turn marker that must follow the assistant answer in training differs per model
  (<|im_end|> for Qwen, <end_of_turn> for Gemma, <|eot_id|> for Llama): it is read from the template instead of hard-coded.
"""
import torch

_NO_SYSTEM = {}


def supports_system(tok) -> bool:
    key = tok.name_or_path
    if key not in _NO_SYSTEM:
        try:
            tok.apply_chat_template([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], tokenize=False, add_generation_prompt=True)
            _NO_SYSTEM[key] = True
        except Exception:
            _NO_SYSTEM[key] = False
    return _NO_SYSTEM[key]


def normalize(tok, messages):
    if messages and messages[0]["role"] == "system" and not supports_system(tok):
        merged = {"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}
        return [merged] + messages[2:]
    return messages


def prompt_text(tok, messages) -> str:
    """Text of the conversation up to where the assistant starts answering."""
    return tok.apply_chat_template(normalize(tok, messages), add_generation_prompt=True, tokenize=False)


def prompt_ids(tok, messages, device):
    """Input for model.generate(**...). Gemma's template already contains <bos>, so special tokens are not added again."""
    ids = tok(prompt_text(tok, messages), add_special_tokens=False, return_tensors="pt")
    return {k: v.to(device) for k, v in ids.items()}


def train_pair(tok, messages):
    """(prompt text, reply text) for a chat that ends with the assistant answer; reply includes the end-of-turn marker."""
    prompt = prompt_text(tok, messages[:-1])
    full = tok.apply_chat_template(normalize(tok, messages), tokenize=False)
    if full.startswith(prompt):
        return prompt, full[len(prompt):]
    raise ValueError("chat template does not extend the generation prompt; add a special case for this model")


def load_model(base, four_bit=False, device=None):
    """(tokenizer, model, dtype, device) for any supported base model; 4-bit (QLoRA-style) loading for 7B+ models on a 16 GB GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # bf16 where the GPU supports it (A100, L4; needed for Gemma); fp16 on a T4
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(base)
    kw = {"attn_implementation": "eager"} if "gemma" in base.lower() else {}  # Gemma is trained with eager attention (soft-capping)
    if four_bit:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=q, dtype=dtype, device_map={"": 0}, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, **kw).to(device)
    return tok, model, dtype, device
