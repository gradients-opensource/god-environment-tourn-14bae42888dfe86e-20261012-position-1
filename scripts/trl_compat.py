"""Compatibility shims for TRL 0.18.0 — inject missing symbols before any env import."""
import sys
import types


def _generate_rollout_completions(trainer, prompts, as_chat=False, **kwargs):
    import torch
    import torch.nn.functional as F

    tok = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    model = trainer.model
    results = []
    for prompt in prompts:
        text = (
            tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            if as_chat
            else str(prompt)
        )
        enc = tok(text, return_tensors="pt")
        dev = next(model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        max_new = getattr(getattr(trainer, "args", None), "max_completion_length", 512) or 512
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tok.eos_token_id,
            )
        pids = enc["input_ids"][0].tolist()
        cids = out.sequences[0][plen:].tolist()
        lps = [F.log_softmax(s[0], dim=-1)[c].item() for s, c in zip(out.scores or [], cids)]
        results.append({"prompt_ids": pids, "completion_ids": cids, "logprobs": lps})
    return results


def patch():
    import trl
    if "trl.experimental" not in sys.modules:
        _exp = types.ModuleType("trl.experimental")
        _exp.__path__ = []
        sys.modules["trl.experimental"] = _exp
        trl.experimental = _exp

    if "trl.experimental.openenv" not in sys.modules:
        _oe = types.ModuleType("trl.experimental.openenv")
        _oe.generate_rollout_completions = _generate_rollout_completions
        sys.modules["trl.experimental.openenv"] = _oe
        trl.experimental.openenv = _oe


patch()
