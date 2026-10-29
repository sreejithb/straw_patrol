"""Fallacy analysis for the demo: score every label name, calibrate, decide what to show, explain the top answer.

analyze(text) -> {"kind", "message", "items": [{"name", "p", "also_called"}], "explanation", "all": [(name, p) top 8]}
decide() is pure (numpy only) so it can be tested without the model.
"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SYSTEM_MSG = "You are a logical-fallacy expert. Identify the fallacy in the argument and explain it."


def norm(n):
    import re
    n = re.sub(r"[^a-z0-9 ]", " ", n.lower())
    return " ".join(re.sub(r"\bfallacy\b", " ", n).split())


_CJK = None


def clean_explanation(text, name):
    """The base model sometimes drifts into Chinese mid-sentence: cut the text where that starts and keep the English part."""
    import re
    global _CJK
    _CJK = _CJK or re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
    m = _CJK.search(text)
    if m:
        text = text[:m.start()].rstrip(" ,;:-\u2014")
        fallback = "This is not a fallacy: the text does not make an argument that can be checked." if name == "None" else f"This looks like {name}."
        if not text.endswith((".", "!", "?")):
            cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
            if cut <= text.find(":") + 1:  # no complete sentence after the label: use the neutral sentence
                return fallback
            text = text[:cut + 1]
        if len(text) < 25:
            return fallback
    return text


def softmax(scores, temperature):
    z = np.asarray(scores, dtype=np.float64) / temperature
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def neighbours(names, alias):
    """name -> set of names that are listed as near-synonyms of it, in either direction."""
    idx = {norm(n): n for n in names}
    nb = {n: set() for n in names}
    for g, alts in alias.items():
        for a in alts:
            if norm(g) in idx and norm(a) in idx:
                nb[idx[norm(g)]].add(idx[norm(a)])
                nb[idx[norm(a)]].add(idx[norm(g)])
    return nb


def decide(p, names, nb, q=0.8, cap=3):
    """Show names until they cover q of the probability (at most `cap`); near-synonyms of a shown name are folded into it."""
    order = np.argsort(-p)
    shown, folded, mass = [], {}, 0.0
    for i in order:
        n = names[i]
        owner = next((s for s in shown if n in nb[s]), None)
        if owner:
            folded.setdefault(owner, []).append(n)
            mass += p[i]
            continue
        if mass >= q or len(shown) >= cap:
            break
        shown.append(n)
        mass += p[i]
    items = [{"name": n, "p": float(p[names.index(n)]), "also_called": folded.get(n, [])} for n in shown]
    fall = [it for it in items if it["name"] != "None"]
    if shown == ["None"]:
        kind, msg = "none", "No logical fallacy found in this text."
    elif shown[0] == "None":
        kind = "maybe_none"
        msg = "Probably no fallacy here. If there is one, it is most likely " + _join([it["name"] for it in fall]) + "."
    elif len(shown) == 1:
        kind, msg = "one", f"This looks like **{shown[0]}**."
    else:
        kind = "several"
        alts = [it["name"] for it in items[1:]]
        msg = f"This is mostly **{shown[0]}**, but it could also be " + _join(alts) + "."
        if any(it["name"] == "None" for it in items[1:]):
            msg = msg.replace("None", "no fallacy at all")
    return {"kind": kind, "message": msg, "items": items}


def _join(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


class FallacyAnalyzer:
    def __init__(self, adapter=None, base=None, token=None, model=None, tok=None):
        import torch
        self.torch = torch
        cfg = json.loads((HERE / "calibration.json").read_text())
        self.T, self.q, self.cap = cfg["temperature"], cfg["q"], cfg["cap"]
        self.names = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))
        self.nb = neighbours(self.names, json.loads((HERE / "alias_groups.json").read_text(encoding="utf-8")))
        if model is None:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            token = token or os.environ.get("HF_TOKEN")
            base = base or cfg["base"]
            adapter = adapter or os.environ.get("ADAPTER_REPO")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"  # on ZeroGPU this is an emulated cuda until a @spaces.GPU call
            dtype = torch.float32
            if self.device == "cuda":
                try:
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                except Exception:
                    dtype = torch.bfloat16
            tok = AutoTokenizer.from_pretrained(base, token=token)
            try:  # newer transformers call this argument `dtype`, older ones `torch_dtype`
                model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, token=token)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, token=token)
            model = PeftModel.from_pretrained(model, adapter, token=token, torch_device="cpu").merge_and_unload().eval()  # read and merged on the CPU, then moved
            model = model.to(self.device)
        else:
            self.device = next(model.parameters()).device.type
        self.tok, self.model = tok, model
        ids = [tok(f"Fallacy: {n}\n", add_special_tokens=False)["input_ids"] for n in self.names]
        self.L = max(map(len, ids))
        pad = tok.pad_token_id
        self.cand_ids = torch.tensor([c + [pad] * (self.L - len(c)) for c in ids], device=self.device)
        self.cand_mask = torch.tensor([[1] * len(c) + [0] * (self.L - len(c)) for c in ids], device=self.device)

    def _prompt(self, text):
        import chat_utils
        return chat_utils.prompt_ids(self.tok, [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": text}], self.device)

    def score(self, text, chunk=0):
        """Log-probability of 'Fallacy: <name>\\n' for every name, in the order of self.names."""
        torch = self.torch
        p = self._prompt(text)
        n_names = len(self.names)
        chunk = chunk or (n_names if self.device == "cuda" else 40)
        out = torch.empty(n_names, device=self.device)
        with torch.no_grad():
            for c0 in range(0, n_names, chunk):
                c1 = min(c0 + chunk, n_names)
                try:  # fast path: read the prompt once, reuse its key/value cache for every candidate name
                    o = self.model(**p, use_cache=True)
                    first = torch.log_softmax(o.logits[0, -1].float(), -1)
                    cache = o.past_key_values
                    cache.batch_repeat_interleave(c1 - c0)
                    attn = torch.cat([torch.ones(c1 - c0, p["input_ids"].shape[1], device=self.device, dtype=self.cand_mask.dtype), self.cand_mask[c0:c1]], 1)
                    logits = self.model(input_ids=self.cand_ids[c0:c1], attention_mask=attn, past_key_values=cache, use_cache=True).logits
                    lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(2, self.cand_ids[c0:c1, 1:, None])[..., 0]
                    out[c0:c1] = first[self.cand_ids[c0:c1, 0]] + (lp * self.cand_mask[c0:c1, 1:]).sum(1)
                except (AttributeError, TypeError, NotImplementedError):  # cache API differs in this transformers version: slower but always works
                    out[c0:c1] = self._score_no_cache(p, c0, c1)
        return out.cpu().numpy()

    def _score_no_cache(self, p, c0, c1):
        torch = self.torch
        n, plen = c1 - c0, p["input_ids"].shape[1]
        ids = torch.cat([p["input_ids"].repeat(n, 1), self.cand_ids[c0:c1]], 1)
        attn = torch.cat([torch.ones(n, plen, device=self.device, dtype=self.cand_mask.dtype), self.cand_mask[c0:c1]], 1)
        logits = self.model(input_ids=ids, attention_mask=attn).logits.float()
        lp = torch.log_softmax(logits[:, plen - 1:-1], -1).gather(2, self.cand_ids[c0:c1, :, None])[..., 0]
        return (lp * self.cand_mask[c0:c1]).sum(1)

    def explain(self, text, name, max_new_tokens=90):
        """One-sentence explanation: the model continues an answer that already names the chosen fallacy."""
        import chat_utils
        torch = self.torch
        prompt = chat_utils.prompt_text(self.tok, [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": text}]) + f"Fallacy: {name}\n"
        x = self.tok(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        return clean_explanation(self.tok.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip(), name)

    def analyze(self, text):
        p = softmax(self.score(text), self.T)
        res = decide(p, self.names, self.nb, self.q, self.cap)
        top = res["items"][0]["name"]
        res["explanation"] = self.explain(text, top if top != "None" else "None")
        order = np.argsort(-p)[:8]
        res["all"] = [(self.names[i], float(p[i])) for i in order]
        return res
