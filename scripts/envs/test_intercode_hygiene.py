"""Tests for the intercode hygiene filter, de-loop rows, eval-exact self-gen observations and the R1 budget.

Run inside the training image from scripts/ (self-gen needs the container sentinel):
    INTERCODE_SELFGEN_CONTAINER=1 python envs/test_intercode_hygiene.py
With MINER_DATASETS_DIR/MINER_DATASETS pointing at the whitelisted intercode dataset it also builds the
real dataset and prints the per-class filter counts.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []


def check(name: str, cond: bool) -> None:
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        FAILS.append(name)


def test_classes() -> None:
    from envs.intercode_dataset import _passthrough_class as c
    expect = {
        "kill -9 -1": "killer", "pkill python": "killer", "killall firefox": "killer",
        "kill $(pgrep sleep)": "killer", "rm -rf /": "destructive", "shutdown -h now": "destructive",
        ":(){ :|:& };:": "destructive", "yes > big.txt": "diskfill", "fallocate -l 10G f": "diskfill",
        "tail -f /var/log/syslog": "hang", "watch ls": "hang", "top": "hang", "yes": "hang",
        "ping google.com": "hang", "sleep 60": "hang", "less file.txt": "hang", "cat /dev/urandom | wc -c": "hang",
        "dd if=/dev/zero of=f bs=1M": "hang", "ssh user@host": "network", "scp a.txt host:/tmp": "network",
        "wget http://x/y": "network", "git clone https://x": "network", "pip install numpy": "network",
        "find /path/to/dir -name '*.txt'": "placeholder", "cat <file>": "placeholder",
    }
    ok = {
        "find . -name '*.txt'": None, "ls -l | wc -l": None, "cat file": None, "top -b -n 1": None,
        "tail -n 5 f": None, "rsync -a a/ b/": None, "grep -r foo .": None, "yes | head -n 5": None,
        "ping -c 1 localhost": None, "sleep 2": None, "dd if=a of=b count=1": None, "kill $PID": None,
        "du -sh /testbed": None, "sort file | uniq -c | sort -nr": None,
    }
    for cmd, want in {**expect, **ok}.items():
        got = c(cmd)
        check(f"class({cmd!r}) = {want} (got {got})", got == want)


def test_rows() -> None:
    import envs.intercode_dataset as D
    hang = D._row_to_tool_examples({"instruction": "Follow the log", "response": "tail -f app.log"})
    user = hang[0]["messages"][1]["content"] if hang else ""
    check("hang gold -> one timeout->submit row",
          len(hang) == 1 and "Observation 1: Command timed out" in user
          and hang[0]["messages"][2]["tool_calls"][0]["function"]["name"] == D.INTERCODE_SUBMIT_TOOL_NAME
          and "Turn 2:" in user)
    check("killer gold -> no rows", D._row_to_tool_examples({"instruction": "stop all", "response": "pkill -9 python"}) == [])
    normal = D._row_to_tool_examples({"instruction": "Count txt files", "response": "find . -name '*.txt' | wc -l"})
    check("normal gold -> execute row first", normal and normal[0]["messages"][2]["tool_calls"][0]["function"]["name"]
          == D.INTERCODE_EXECUTE_TOOL_NAME)
    s3 = D._submit_after("q", [("cmd a", "out a"), ("cmd b", "out b")])
    u3 = s3["messages"][1]["content"]
    check("turn-3 submit carries both steps", "Tool 1: execute_bash" in u3 and "Tool 2: execute_bash" in u3 and "Turn 3:" in u3)


def test_selfgen() -> None:
    os.environ.setdefault("INTERCODE_SELFGEN_CONTAINER", "1")
    from envs.intercode_selfgen import generate_selfgen_tasks
    tasks = generate_selfgen_tasks(60, seed=3)
    check(f"self-gen produced tasks ({len(tasks)})", len(tasks) >= 30)
    check("no near-miss observation equals the answer", all(t.near_miss_obs != t.gold_obs for t in tasks if t.near_miss_obs))
    check("no verbose observation equals the answer", all(t.verbose_obs != t.gold_obs for t in tasks if t.verbose_obs))
    from envs.intercode_selfgen import _Tree, _run_confined
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = os.path.join(td, "sbx_root"); os.makedirs(root)
        rc, out, err = _run_confined("nosuchcmd_xyz", _Tree(td, root, "/testbed"), 5)
        check(f"bash error text ({err.strip()[:60]!r})", err.startswith("/bin/bash: line 1: nosuchcmd_xyz: command not found"))
        rc, out, err = _run_confined("echo x; ls /nope_dir_q", _Tree(td, root, "/testbed"), 5)
        check("stdout present -> observation would be stdout only", out == "x\n")


def test_build_budget() -> None:
    if not os.getenv("MINER_DATASETS_DIR"):
        print("SKIP build (no MINER_DATASETS_DIR)")
        return
    os.environ["INTERCODE_SELFGEN"] = "1"
    os.environ["INTERCODE_SELFGEN_N"] = "3000"
    os.environ["R1_MIN_ENV_ROWS"] = "20000"
    import importlib
    import envs.intercode_dataset as D
    importlib.reload(D)
    dd = D.build_intercode_sft_dataset()
    n = len(dd["train"])
    check(f"R1 intercode rows within the 20000 budget ({n})", n <= 20000)
    users = [m[1]["content"] for m in dd["train"]["messages"][:n]]
    check("timeout->submit rows present", any("Command timed out" in u for u in users))
    tools = [json.loads(t) if isinstance(t, str) else t for t in dd["train"]["tools"][:3]]
    check("tools block unchanged (execute_bash + submit)", all(len(t) == 2 for t in tools))


def test_r1_prefix() -> None:
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sft_env_config.py")).read()
    check("R1 prefix disables the fake retry rows", "INTERCODE_RETRY_FRAC=0" in src)


# ---------------------------------------------------------------------------------------------
# The intercode recipe in every round that scores intercode (R1 command byte-identical to 13759ee).
# ---------------------------------------------------------------------------------------------
_R1_GEN = ("R1_MIN_ENV_ROWS=20000 INTERCODE_SELFGEN=1 INTERCODE_SELFGEN_CONTAINER=1 INTERCODE_SELFGEN_N=3000 "
           "INTERCODE_RETRY_FRAC=0 python -m envs.generate_trajectories --environment_names intercode {game} "
           "--output_path /workspace/scripts/datasets/sft_env_T1 --time_limit_seconds 810 --scored_cap 6000")
_R1_RUN_SHA256 = "ca6a46e38a9f746d6dc516b3dbbadd452b01a5bf6ef5cc1c53479214081b0bfc"  # 13759ee, Llama-8B, 2 GPU
_BOSS = ["goofspiel", "gin_rummy", "clobber", "leduc_poker", "intercode", "othello"]


def _sft_config(gpus: int = 2):
    import types
    mu = types.ModuleType("model_utility")
    mu.get_gpu_count = lambda: gpus
    mu.get_model_architecture = lambda p: "LlamaForCausalLM"
    mu.get_model_num_params = lambda p: 8_030_261_248
    mu.disable_flash_attention = lambda a: False
    mu.get_use_liger = lambda a: True
    sys.modules["model_utility"] = mu
    import sft_env_config as S
    S.get_gpu_count = mu.get_gpu_count
    return S


def _training_json(envs, gpus: int = 2, hours: float = 1.5, model_path: str = "/nonexistent") -> dict:
    return _sft_config(gpus).get_training_json({
        "task_id": "T1", "model_path": model_path, "model_name": "m", "hours_to_complete": hours,
        "reg_ratio": 1.0, "output_dir": "/o", "request_path": "/r.json",
        "dataset_type": {"environment_names": envs}})


def test_r1_commands_byte_identical() -> None:
    import hashlib
    for game in ("clobber", "othello", "gin_rummy"):
        r = _training_json(["intercode", game])
        check(f"R1 [intercode, {game}] generate_cmd == 13759ee", r["generate_cmd"] == _R1_GEN.format(game=game))
        check(f"R1 [intercode, {game}] run_cmd == 13759ee",
              hashlib.sha256(r["run_cmd"].encode()).hexdigest() == _R1_RUN_SHA256)


def test_recipe_in_every_intercode_round() -> None:
    import budget_controller as bc
    for envs, gpus, total in ((["intercode", "clobber", "liars_dice", "gin_rummy"], 2, 18000), (_BOSS, 4, 32000),
                              (_BOSS, 4, 54000)):
        cmd = _training_json(envs, gpus, hours=3.0 if total == 54000 else 1.5)["generate_cmd"]
        budget = bc.share_caps(envs, total)["intercode"]
        tag = f"{len(envs)}-env total {total}"
        check(f"{tag}: self-gen on, retry 0, fake submits 0",
              all(k in cmd for k in ("INTERCODE_SELFGEN=1 ", "INTERCODE_SELFGEN_CONTAINER=1 ", "INTERCODE_RETRY_FRAC=0 ",
                                     "INTERCODE_SUBMIT_FRAC=0 ")))
        check(f"{tag}: intercode row budget = its share cap ({budget})", f"INTERCODE_ROW_BUDGET={budget} " in cmd)
        check(f"{tag}: no R1 20k floor outside R1", "R1_MIN_ENV_ROWS" not in cmd)
    cmd = _training_json(["clobber", "liars_dice", "goofspiel", "leduc_poker"])["generate_cmd"]
    check("round without intercode: no intercode recipe", "INTERCODE_" not in cmd)


def _synthetic_passthrough(n_exec: int, n_hang: int) -> list:
    rows = [{"instruction": f"PT count entries in folder number {i}", "response": f"ls -la dir{i} | wc -l"}
            for i in range(n_exec)]
    rows += [{"instruction": f"PT follow log number {i}", "response": f"tail -f app{i}.log"} for i in range(n_hang)]
    return rows


def _build_with(env: dict, passthrough: list):
    import importlib
    from pathlib import Path
    keys = ("INTERCODE_SELFGEN", "INTERCODE_SELFGEN_CONTAINER", "INTERCODE_SELFGEN_N", "INTERCODE_RETRY_FRAC",
            "INTERCODE_SUBMIT_FRAC", "INTERCODE_ROW_BUDGET", "R1_MIN_ENV_ROWS")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        import envs.intercode_dataset as D
        D = importlib.reload(D)
        D._find_intercode_dataset = lambda: ("synthetic", Path("/synthetic"))
        D._load_one_dataset = lambda p: passthrough
        dd = D.build_intercode_sft_dataset()
        return D, [(m[1]["content"], m[2]["tool_calls"][0]["function"]["name"]) for m in dd["train"]["messages"]]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_budget_split_real_observations() -> None:
    recipe = {"INTERCODE_SELFGEN": "1", "INTERCODE_SELFGEN_CONTAINER": "1", "INTERCODE_SELFGEN_N": "1500",
              "INTERCODE_RETRY_FRAC": "0", "INTERCODE_SUBMIT_FRAC": "0", "INTERCODE_ROW_BUDGET": "7000"}
    D, rows = _build_with(recipe, _synthetic_passthrough(8000, 200))
    pt = [(u, a) for u, a in rows if "PT " in u]
    sg = [(u, a) for u, a in rows if "PT " not in u]
    check(f"budget 7000 -> exactly 7000 rows ({len(rows)})", len(rows) == 7000)
    check(f">= 3500 real-observation self-gen rows ({len(sg)})", len(sg) >= 3500)
    check("0 retry rows", not any(err in u for u, _ in rows for err in D._TRANSIENT_ERR_OBS))
    check("0 fake-observation submit rows (every passthrough submit is a timeout)",
          all("Command timed out" in u for u, a in pt if a == D.INTERCODE_SUBMIT_TOOL_NAME))
    check("timeout->submit rows still present", any("Command timed out" in u for u, _ in pt))
    D, rows = _build_with({**recipe, "INTERCODE_SELFGEN": "0"}, _synthetic_passthrough(2000, 50))
    fake = [u for u, a in rows if a == D.INTERCODE_SUBMIT_TOOL_NAME and "Command timed out" not in u]
    check(f"self-gen unavailable -> passthrough submit rows restored ({len(fake)})", len(fake) > 0)


def test_r1_budget_path_unchanged() -> None:
    r1 = {"INTERCODE_SELFGEN": "1", "INTERCODE_SELFGEN_CONTAINER": "1", "INTERCODE_SELFGEN_N": "300",
          "INTERCODE_RETRY_FRAC": "0", "R1_MIN_ENV_ROWS": "3000"}
    D, rows = _build_with(r1, _synthetic_passthrough(2000, 50))
    pt = [(u, a) for u, a in rows if "PT " in u]
    fake = [u for u, a in pt if a == D.INTERCODE_SUBMIT_TOOL_NAME and "Command timed out" not in u]
    check(f"R1: every passthrough row kept, self-gen trimmed to the rest ({len(pt)} + {len(rows) - len(pt)})",
          len(rows) == 3000 and len(pt) == 2050 + len(fake))
    check(f"R1: fake-observation submit rows unchanged (SUBMIT_FRAC default, {len(fake)})", len(fake) > 0)


def test_baseline_stats_dose() -> None:
    import tempfile
    import budget_controller as bc
    stats = {"task_type": "env", "weights": {},
             "env_stats": {"intercode": {"num_episodes": 75, "mean_score": 0.70, "std_score": 0.18},
                           "clobber": {"num_episodes": 40, "mean_score": 0.2}}}
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, "baseline.json")
        bad = os.path.join(td, "corrupt.json")
        json.dump(stats, open(good, "w"))
        open(bad, "w").write("{not json")
        check("baseline parse: env_stats.intercode.mean_score", bc.baseline_env_mean(good, "intercode") == 0.70)
        check("corrupt / missing / unset baseline file -> None (no-op)",
              bc.baseline_env_mean(bad, "intercode") is None and bc.baseline_env_mean(os.path.join(td, "x"), "intercode")
              is None and bc.baseline_env_mean(None, "intercode") is None)
        check("forgotten intercode on an 8B continuation -> x1.5", bc.intercode_dose(0.70, True, 8.03e9) == 1.5)
        check("healthy intercode -> x1", bc.intercode_dose(0.85, True, 8.03e9) == 1.0)
        check("from-scratch / small base / no stats -> x1",
              bc.intercode_dose(0.70, False, 8.03e9) == 1.0 and bc.intercode_dose(0.70, True, 3.2e9) == 1.0
              and bc.intercode_dose(None, True, 8.03e9) == 1.0)
        cont = os.path.join(td, "merged_base")
        os.makedirs(cont)
        json.dump({"rope_parameters": {"rope_theta": 500000.0, "rope_type": "llama3"}}, open(os.path.join(cont, "config.json"), "w"))
        base = bc.share_caps(_BOSS, 32000)["intercode"]
        os.environ["BASELINE_STATS_PATH"] = good
        try:
            dosed = _training_json(_BOSS, 4, model_path=cont)["generate_cmd"]
            plain = _training_json(_BOSS, 4)["generate_cmd"]
        finally:
            os.environ.pop("BASELINE_STATS_PATH", None)
        check(f"continuation + weak baseline -> budget {int(base * 1.5)}", f"INTERCODE_ROW_BUDGET={int(base * 1.5)} " in dosed)
        check(f"foundation base -> budget {base} (no dose)", f"INTERCODE_ROW_BUDGET={base} " in plain)
        rec = os.path.join(td, "seen.json")
        bc.record_baseline(rec, {"intercode_mean": 0.70})
        check("baseline recorded to a local file", json.load(open(rec)).get("intercode_mean") == 0.70)


if __name__ == "__main__":
    for t in (test_classes, test_rows, test_selfgen, test_build_budget, test_r1_prefix, test_r1_commands_byte_identical,
              test_recipe_in_every_intercode_round, test_budget_split_real_observations, test_r1_budget_path_unchanged,
              test_baseline_stats_dose):
        t()
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAIL(S)'}")
    sys.exit(1 if FAILS else 0)
