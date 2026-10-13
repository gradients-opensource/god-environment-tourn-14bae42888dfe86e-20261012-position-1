"""Regression tests for the tournament training defaults that cannot be checked by a short smoke run.

Run inside the training image from scripts/:  python test_train_defaults.py
Covers: the deadline-annealed cosine (cosine_clock), the global packed batch split across ranks and OOM
retries, the strict SGLang-style tool-call parsing used by envs/verify_vs_winner.py, and the submission
publish policy of CustomEvalSaveCallback.
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILS = []


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


def hf_cosine(step: int, total: int, warmup: int, min_rate: float) -> float:
    """transformers' cosine_with_min_lr (num_cycles 0.5) for reference."""
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_rate + (1 - min_rate) * 0.5 * (1 + math.cos(math.pi * p))


def test_clock_cosine() -> None:
    from train_sft_env import clock_cosine_multiplier as m
    check("warmup ignores the clock", m(10, 937, 35, 0.99, 0.25) == 10 / 35)
    check("clock behind steps == HF step cosine", abs(m(500, 937, 35, 0.1, 0.25) - hf_cosine(500, 937, 35, 0.25)) < 1e-12)
    check("clock ahead of steps drives the decay",
          abs(m(300, 937, 35, 0.9, 0.25) - (0.25 + 0.75 * 0.5 * (1 + math.cos(math.pi * 0.9)))) < 1e-12)
    check("deadline reached -> min_lr_rate", abs(m(700, 937, 35, 1.0, 0.25) - 0.25) < 1e-12)
    check("past the deadline clamps to min_lr_rate", abs(m(700, 937, 35, 3.0, 0.25) - 0.25) < 1e-12)
    check("never above the HF step cosine",
          all(m(s, 937, 35, tp, 0.25) <= hf_cosine(s, 937, 35, 0.25) + 1e-12
              for s in range(0, 938, 7) for tp in (0.0, 0.3, 0.6, 0.9, 1.2)))


def test_clock_stop_callback() -> None:
    """R2-like run on a fake clock: 2.5 planned epochs x 524 steps at 4.35 s/step in a 4800 s SFT window ->
    the callback stops (save + stop) at the step that fits, where the moved cosine horizon reaches min_lr."""
    import datetime as real_dt
    import types
    import train_sft_env as T
    from transformers import TrainerControl, TrainerState
    start = real_dt.datetime(2026, 10, 5, 9, 0, tzinfo=real_dt.timezone.utc)
    clock = {"t": start}

    class _Now(real_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["t"]

    saved = T.datetime
    T.datetime = types.SimpleNamespace(datetime=_Now, timezone=real_dt.timezone, timedelta=real_dt.timedelta)
    try:
        cb = T.ClockCosineCallback(3e-5, 35, 0.25, 1310, "2026-10-05 10:25:00", 5, steps_per_epoch=524, min_epochs=1.5)
        state, control = TrainerState(), TrainerControl()
        cb.on_train_begin(None, state, control)
        opt = types.SimpleNamespace(param_groups=[{"lr": 0.0}])
        stopped_at, lr_at_stop = None, None
        for step in range(1, 1400):
            clock["t"] = start + real_dt.timedelta(seconds=step * 4.35)
            state.global_step = step
            cb.on_step_begin(None, state, control, optimizer=opt)
            cb.on_step_end(None, state, control)
            if control.should_training_stop:
                stopped_at, lr_at_stop = step, opt.param_groups[0]["lr"]
                break
    finally:
        T.datetime = saved
    want = 20 + int((4800 - 20 * 4.35 - 90) // 4.35)
    check(f"clock stop saves + stops at the step that fits ({stopped_at}, want {want})",
          stopped_at == want and control.should_save and cb.total_steps == want)
    check(f"LR at the stop step is min_lr ({lr_at_stop})", lr_at_stop is not None and abs(lr_at_stop - 3e-5 * 0.25) < 1e-12)


def test_pack_batch() -> None:
    from train_sft_env import _pack_batch_size
    saved = {k: os.environ.pop(k, None) for k in ("PACK_BATCH", "OOM_HALVINGS")}
    try:
        check("1 GPU -> 16 per device", _pack_batch_size(1) == 16)
        check("2 GPUs -> 8 per device (global 16)", _pack_batch_size(2) == 8)
        os.environ["OOM_HALVINGS"] = "1"
        check("2 GPUs after one OOM halving -> 4", _pack_batch_size(2) == 4)
        os.environ["OOM_HALVINGS"] = "9"
        check("never below 1", _pack_batch_size(1) == 1)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_strict_parser() -> None:
    import envs.verify_vs_winner as V
    sa = V._strict_action
    legal = [3, 7, 12]
    check("llama single call", sa('{"name": "game_action", "parameters": {"action_id": 7}}<|eot_id|>', "llama", legal) == 7)
    check("llama text before the JSON forfeits",
          sa('Move: {"name": "game_action", "parameters": {"action_id": 7}}', "llama", legal) is None)
    check("llama memory call then game_action",
          sa('{"name": "working_memory_rewrite", "parameters": {"slot": 1, "content": "x"}}; '
             '{"name": "game_action", "parameters": {"action_id": 3}}', "llama", legal) == 3)
    check("qwen exact block", sa('<tool_call>\n{"name": "game_action", "arguments": {"action_id": 12}}\n</tool_call>', "qwen", legal) == 12)
    check("qwen block without newlines forfeits (SGLang needs them)",
          sa('<tool_call>{"name": "game_action", "arguments": {"action_id": 12}}</tool_call>', "qwen", legal) is None)
    check("qwen empty think block then call",
          sa('<think>\n\n</think>\n\n<tool_call>\n{"name": "game_action", "arguments": {"action_id": 3}}\n</tool_call>', "qwen", legal) == 3)
    check("illegal action forfeits", sa('<tool_call>\n{"name": "game_action", "arguments": {"action_id": 5}}\n</tool_call>', "qwen", legal) is None)
    check("first LEGAL game_action wins",
          sa('<tool_call>\n{"name": "game_action", "arguments": {"action_id": 5}}\n</tool_call>\n'
             '<tool_call>\n{"name": "game_action", "arguments": {"action_id": 7}}\n</tool_call>', "qwen", legal) == 7)
    check("bool action_id rejected", sa('<tool_call>\n{"name": "game_action", "arguments": {"action_id": true}}\n</tool_call>', "qwen", [1]) is None)


def test_publish_latest() -> None:
    from transformers import TrainerControl, TrainerState
    import customized_trainer as CT
    root = tempfile.mkdtemp()
    out, sub = os.path.join(root, "soutputs", "t_0"), os.path.join(root, "ckpt", "t", "repo")
    for step in (200, 400):
        d = os.path.join(out, f"checkpoint-{step}")
        os.makedirs(d)
        with open(os.path.join(d, "adapter_model.safetensors"), "wb") as f:
            f.write(b"\0" * 64)
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            f.write("{}")
    cb = CT.CustomEvalSaveCallback(lambda s: {"eval": False}, sub, out, "base", publish_policy="latest")
    for step, loss in ((200, 0.09), (400, 0.10)):
        st = TrainerState()
        st.global_step, st.epoch, st.log_history = step, step / 1000, [{"loss": 0.05}]
        cb.on_evaluate(None, st, TrainerControl(), {"eval_loss": loss})
        cb.on_save(None, st, TrainerControl())
    with open(os.path.join(sub, "loss.txt")) as f:
        check("latest publishes the newest checkpoint even with a higher eval_loss", f.read().startswith("400,"))


if __name__ == "__main__":
    for t in (test_clock_cosine, test_clock_stop_callback, test_pack_batch, test_strict_parser, test_publish_latest):
        t()
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAIL(S)'}")
    sys.exit(1 if FAILS else 0)
