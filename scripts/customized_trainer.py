from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
import re
from typing import Callable, Optional, Dict
import shutil
import json
from transformers.trainer_utils import is_main_process
import wandb
import torch
import numpy as np
import math
from collections import deque
from state_manager import get_state, set_state
from submission_utils import publish_dir_atomic, submission_has_weights
MAX_TRIES = 9


def _all_ranks_any(flag: bool, args=None) -> bool:
    """Under torchrun, True on EVERY rank when it is True on ANY rank (one scalar MAX all-reduce).

    Each rank reads its own wall clock, so a time threshold can be crossed on rank 0 but not yet on
    rank 1 in the same step; one rank would then enter _save_checkpoint's collectives while the other
    runs the next training step -> mismatched NCCL collectives -> hang until ddp_timeout. Callbacks run
    at the same program point on every rank, so this collective is safe. No-op when not distributed."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            device = getattr(args, "device", None)
            if device is None:
                device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
            t = torch.tensor([1 if flag else 0], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return bool(t.item())
    except Exception as exc:
        print(f"[deadline] rank sync skipped: {exc}", flush=True)
    return flag


def _is_finite_or_absent(value) -> bool:
    """True for None / non-numeric markers ("no_eval") and finite numbers; False for NaN / inf."""
    if value is None or isinstance(value, str):
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


MIS_MATCH_VOCAB_SIZE_MODELS = [
    'NousResearch/Nous-Capybara-7B-V1',
    'berkeley-nest/Starling-LM-7B-alpha',
    'NousResearch/Hermes-2-Theta-Llama-3-8B',
    'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4'
]

ERROR_GENERATION_CONFIG_MODELS = [
    "lmsys/vicuna-7b-v1.5", 
    "lmsys/vicuna-13b-v1.5",
    "NousResearch/Nous-Hermes-llama-2-7b", 
    "defog/llama-3-sqlcoder-8b"
]

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

print(f"LOCAL_RANK: {LOCAL_RANK} in customized_trainer.py", flush=True)
    
class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
        max_steps: int = -1,
        checking_step: int = 100,
        total_steps_all_epochs: int = -1,
        end_time: str = "",
        checking_mode: str = "none",
        lr_finder_overhead_seconds: int = 780,  # default 13 minutes
        publish_policy: str = "best",
    ):
        # publish_policy — which checkpoint becomes the submission:
        #   "best"   (default, unchanged): the checkpoint with the lowest eval_loss.
        #   "latest": every saved checkpoint is published as it lands, so the submission is always
        #             the newest one (most data seen, most-annealed LR), including the deadline save,
        #             which is never evaluated. For callers whose eval set is not a real held-out set
        #             (the EnvTask SFT warm-start evaluates 8 packs of its own TRAIN set), "best
        #             eval_loss" is noise: 2026-09-24 cand#4 kept step 600 (0.0921) over step 800
        #             (0.0937) and never published the deadline checkpoint at step 820.
        self.publish_policy = (publish_policy or "best").strip().lower()
        if self.publish_policy not in ("best", "latest"):
            print(f"[publish] unknown publish_policy {publish_policy!r} -> 'best'", flush=True)
            self.publish_policy = "best"
        self._last_eval = None  # (global_step, eval_loss) of the most recent evaluation
        self._last_published_step = None
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        self.max_steps = max_steps
        self.has_checkpoint = False
        self.save_only = False
        self.checking_step = checking_step
        self.total_steps_all_epochs = total_steps_all_epochs
        self.checking_mode = checking_mode
        self.end_time = end_time
        self.lr_finder_overhead_seconds = lr_finder_overhead_seconds
        # Training progress (epoch) already published for this submission by an EARLIER attempt of the
        # same task (text_trainer retries a crashed run from scratch into the same submission_dir).
        self._publish_floor = self._read_publish_floor() if self.publish_policy == "latest" else None
        if is_main_process(LOCAL_RANK):
            print(f"[publish] policy={self.publish_policy} floor_epoch={self._publish_floor}", flush=True)

    def compute_loss(self, state: TrainerState, metrics):
        return metrics.get("eval_loss", None)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # --- Hard deadline guard (belt-and-suspenders) -------------------------------------
        # Once we're within the save buffer of end_time, do a final save and STOP the loop.
        # This is INDEPENDENT of the WhenToEvalHandler path below, which in SFT-only runs was
        # observed to never fire its end_time branch (its `save_before_remaining_time > 0`
        # guard failed), so training kept running ~30+ min past end_time until the pod was
        # externally killed — wasting billing. Fires once. `or 5` guarantees a non-zero buffer
        # even if save_before_remaining_time is 0/unset.
        if self.end_time and not getattr(self, "_deadline_stopped", False):
            _buf = getattr(self.function_when_to_evaluate, "save_before_remaining_time", 5) or 5
            if _all_ranks_any(check_remaining_time_less_than_minutes(self.end_time, _buf), args):
                self._deadline_stopped = True
                print(f"[deadline] step {state.global_step}: within {_buf}min of end_time -> final save + stop", flush=True)
                control.should_save = True
                control.should_training_stop = True
                return control
        # Custom logic to decide whether to save or evaluate
        # print(f"************* on_step_end: {state.global_step}, check eval", flush=True)
        # TODO: implement the logic to save the model without evaluating if there is no check points --> avoid evaluating takes too much time
        # Check if the checking_step is reached
        # print(f"Checking the model at step: {state.global_step}, checking_step: {self.checking_step}, checking_mode: {self.checking_mode}", flush=True)
        if state.global_step == self.checking_step and self.checking_mode == "first_time":
            # print(f"Checking the model at step: {state.global_step}", flush=True)
            # check the time so far to estimate the training time in total 
            my_state = get_state()
            start_time_obj = datetime.datetime.strptime(my_state["train"]["start_time"], "%Y-%m-%d %H:%M:%S")
            start_train_time_obj = datetime.datetime.strptime(my_state["train"]["start_train_time"], "%Y-%m-%d %H:%M:%S")
            
            log_content = f"Checking the model at step: {state.global_step}"
            now = datetime.datetime.now()
            preparation_time = (start_train_time_obj - start_time_obj).total_seconds()
            log_content += f"\nPreparation time: {preparation_time}"
            time_so_far = (now - start_time_obj).total_seconds()
            log_content += f"\nTime so far (raw, includes LR finder): {time_so_far}"
            time_for_one_step = (now - start_train_time_obj).total_seconds() / self.checking_step
            log_content += f"\nTime for one step: {time_for_one_step}"
            # Now estimate the total training time for this training
            log_content += f"\nTotal steps all epochs: {self.total_steps_all_epochs}"
            total_remaining_training_time = time_for_one_step * (self.total_steps_all_epochs - state.global_step)
            log_content += f"\nTotal remaining training time: {total_remaining_training_time}"
            # n * time_so_far + total_remaining_training_time = total_remaining_time
            end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
            total_remaining_time = (end_time_obj - now).total_seconds()
            log_content += f"\nTotal remaining time: {total_remaining_time}"
            
            # LR Finder ran before training and consumed ~10 min; use 13 min as safety margin.
            # We exclude that cost so the formula treats each exploration run as pure training time,
            # giving a more accurate estimate of how many LR-exploration runs we can afford.
            lr_finder_overhead_seconds = self.lr_finder_overhead_seconds
            time_so_far_adjusted = max(0.0, time_so_far - lr_finder_overhead_seconds)
            log_content += f"\nTime so far (adjusted, LR finder excluded): {time_so_far_adjusted}"

            # n * time_so_far_adjusted + (time_so_far_adjusted + total_remaining_training_time) = total_remaining_time
            # time_so_far_adjusted + total_remaining_training_time is the time needed to finish 1 full training run
            # 12 min buffer for eval + save at the end; 3 min buffer for run-time variance
            max_var_time_sofar = 3 * 60
            n = (total_remaining_time - (time_so_far_adjusted + total_remaining_training_time + 12 * 60)) / (time_so_far_adjusted + max_var_time_sofar) # exclude LR finder overhead from per-run estimate
            n = int(n)
            my_state["check_details"] = {
                "now": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                "start_time": str(start_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "start_train_time": str(start_train_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "checking_step": self.checking_step,
                "checking_mode": self.checking_mode,
                "estimation_of_steps": n,
                "preparation_time": preparation_time,
                "time_so_far": time_so_far,
                "time_for_one_step": time_for_one_step,
                "total_remaining_training_time": total_remaining_training_time,
                "total_remaining_time": total_remaining_time,
                "end_time": self.end_time,
            }
            if n > 0: # we should try more 
                log_content += f"\nEstimated number of steps to complete the training: {n}"
                control.should_training_stop = True
                control.should_save = False
                args.save_strategy = "no"
                # save the current loss of this step to the state;
                last_log = state.log_history[-1]
                my_state["train"]["current_loss"] = last_log["loss"]
                my_state["mode"] = "continue"
                if n > MAX_TRIES:
                    n = MAX_TRIES
                log_content += f"\nFinal number: {n + 1}"
                my_state["next_runs"] = n + 1 # including the current run
            else:
                print(f"Time is not enough so we will finish the training", flush=True)
                my_state["mode"] = "finish"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)            
            return control
    
        elif state.global_step == self.checking_step and self.checking_mode == "second_time": # at second time, we don't estimate the training time again, just save the current_loss
            log_content = f"Checking the model at step: {state.global_step} where check_mode=second_time"            
            my_state = get_state()
            current_loss = state.log_history[-1]["loss"]
            my_state["train"]["current_loss"] = current_loss
                
            control.should_training_stop = True

            # Check if current_loss > current min_loss --> do not save to save time and space
            # 
            # if my_state["train"]["current_loss"] > current_min_loss:
            #     print(f"Current loss: {my_state['train']['current_loss']} is greater than the current min_loss: {current_min_loss}, do not save the checkpoint", flush=True)
            #     control.should_save = False
            # check if this is the last run and the current_loss is the lowest --> keep running the training
            current_is_the_best = False
            current_min_loss = min([run["current_loss"] for run in my_state["runs"]])
            if current_loss <= current_min_loss:
                if len(my_state["runs"]) + 1 == my_state["next_runs"]:
                    print(f"Current loss: {my_state['train']['current_loss']} is greater than: {current_min_loss}", flush=True)
                    current_is_the_best = True
                    
            if current_is_the_best:
                control.should_training_stop = False
                my_state["mode"] = "finish"
            else:
                control.should_save = False
                args.save_strategy = "no"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                # print(log_content, flush=True)
        
            
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        if when_to_eval["eval"]:
            # do not allow the pod to be stopped by any reason 
                # first check if there is at least one checkpoint or not 
            print(f"Evaluating the model at step: {state.global_step} the reason: {when_to_eval['reason']}", flush=True)
            control.should_evaluate = True
            control.should_save = True
            if when_to_eval["reason"] == "end_time":
                # Deadline reached: save the model AND stop training. Previously only
                # should_save was set, so training kept running past end_time until the
                # pod was killed externally — wasting ~30+ min of wall-clock/billing per
                # run locally. The checkpoint for this step is written before the loop
                # honors should_training_stop, so the final model is still saved.
                control.should_training_stop = True
                if not self.has_checkpoint: # if there is no checkpoint, we just save the model, do not evaluate
                    print(f"No checkpoint found, just save the model at step: {state.global_step}", flush=True)
                    control.should_evaluate = False
                    self.save_only = True
        return control


    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        self.save_only = False
        # Append eval_loss to file
        eval_loss = self.compute_loss(state, metrics)
        self._last_eval = (state.global_step, eval_loss)
        if state.global_step < 2:
            return
        print(f"GO INTO CUSTOMIZED EVALUATE AT STEP: {state.global_step}", flush=True)
        if self.best_checkpoint_info is None or eval_loss < self.best_checkpoint_info["loss"]:
            print(f"Updating the best checkpoint info at step: {state.global_step} with eval_loss: {eval_loss}", flush=True)
            self.best_checkpoint_info = {
                "loss": eval_loss,
                "step": state.global_step
            }
            self.update_best_checkpoint = True
        else:
            if self.best_checkpoint_info is not None:
                print(f" At step: {state.global_step} The eval_loss: {eval_loss} is not smaller than the current best eval_loss: {self.best_checkpoint_info['loss']}, update_best_checkpoint={self.update_best_checkpoint}", flush=True)
            

    def _validate_config(self, checkpoint_dir: str):
        """Pengecekan integritas config.json sebelum dipindahkan ke submission."""
        config_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                if "model_type" not in config:
                    print(f"⚠️  WARNING: config.json di {checkpoint_dir} tidak memiliki 'model_type'!", flush=True)
                    # Opsional: Coba tebak dari model_name atau original_model_name
                    model_ref_name = self.original_model_name.lower()
                    if "llama" in model_ref_name:
                        config["model_type"] = "llama"
                    elif "mistral" in model_ref_name:
                        config["model_type"] = "mistral"
                    elif "phi" in model_ref_name:
                        config["model_type"] = "phi3"
                    
                    if "model_type" in config:
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=2)
                        print(f"✅ FIXED: Menambahkan model_type '{config['model_type']}' secara otomatis.", flush=True)
                    else:
                        print("❌ GAGAL: Tidak bisa menebak model_type. Harap cek config.json manual.", flush=True)
            except Exception as e:
                print(f"❌ ERROR: Gagal memvalidasi config.json: {e}", flush=True)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.global_step == self.max_steps and self.max_steps != -1:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            control.should_training_stop = True
        
        self.has_checkpoint = True
        
        if not is_main_process(LOCAL_RANK): # if not main process, skip this
            return

        if self.publish_policy == "latest":
            # "latest" publishes every save, including the save_only one (same checkpoint dir), through
            # the guarded path below; the "best" branches that follow are left exactly as they were.
            self.save_only = False
            self._publish_latest(state)
            return

        if self.save_only: # if only save, do not evaluate
            print(f"Only save the model at step: {state.global_step}, no evaluation", flush=True)
            current_step = state.global_step
            source_dir = os.path.join(self.output_dir, f"checkpoint-{current_step}")
            self._validate_config(source_dir) # Validasi sebelum copy
            # Atomic publish: the previous submission stays intact until the new copy is complete.
            publish_dir_atomic(source_dir, self.submission_dir, loss_line=f"{current_step},no_eval")
            self.update_best_checkpoint = False
            
            # release the flag
            self.save_only = False
            return

        # Custom logic after model is saved
        # You can trigger external services, logs, or backups here
        if (
            self.update_best_checkpoint
            and is_main_process(LOCAL_RANK)
        ):
            print(f"Copy the best checkpoint to the submission directory at step: {state.global_step}", flush=True)
            best_eval_loss = self.best_checkpoint_info["loss"]
            source_dir = os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}")
            self._validate_config(source_dir) # Validasi sebelum copy
            # Atomic publish: the previous submission stays intact until the new copy is complete
            # (the old rmtree + copytree left it empty/half-written for the whole copy).
            publish_dir_atomic(
                source_dir,
                self.submission_dir,
                loss_line=f"{self.best_checkpoint_info['step']},{best_eval_loss}",
            )
            self.update_best_checkpoint = False

    def _publish_latest(self, state: TrainerState) -> None:
        """publish_policy="latest": make the checkpoint just saved the submission (main process only,
        checked by the caller). The current submission is left untouched when:
          - this step was already published (HF re-saves the same step at the epoch end / deadline),
          - an earlier attempt of this task already published MORE training progress (a retry after a
            crash restarts from the base model and must not replace a better submission),
          - the latest eval or train loss is non-finite (a diverged run must not ship),
          - the checkpoint dir is missing."""
        current_step = state.global_step
        progress = float(state.epoch or 0.0)
        if current_step == self._last_published_step:
            print(f"[publish] latest: checkpoint-{current_step} already published", flush=True)
            return
        if self._publish_floor is not None and progress + 1e-6 < self._publish_floor:
            print(f"[publish] latest: keep the submission — checkpoint-{current_step} is at epoch "
                  f"{progress:.3f}, an earlier attempt already published epoch {self._publish_floor:.3f}",
                  flush=True)
            return
        loss = self._last_eval[1] if self._last_eval and self._last_eval[0] == current_step else "no_eval"
        train_loss = next((e["loss"] for e in reversed(state.log_history or []) if "loss" in e), None)
        if not _is_finite_or_absent(loss) or not _is_finite_or_absent(train_loss):
            print(f"[publish] latest: keep the submission — non-finite loss at step {current_step} "
                  f"(eval={loss}, train={train_loss})", flush=True)
            return
        source_dir = os.path.join(self.output_dir, f"checkpoint-{current_step}")
        if not os.path.isdir(source_dir):
            print(f"[publish] latest: {source_dir} not found, submission unchanged", flush=True)
            return
        print(f"[publish] latest: checkpoint-{current_step} (epoch {progress:.3f}) -> submission "
              f"(eval_loss={loss})", flush=True)
        self._validate_config(source_dir)
        publish_dir_atomic(source_dir, self.submission_dir, loss_line=f"{current_step},{loss}")
        self.update_best_checkpoint = False
        self._last_published_step = current_step
        self._publish_floor = max(self._publish_floor or 0.0, progress)
        self._write_publish_meta(current_step, progress)

    def _publish_meta_path(self) -> str:
        """Progress marker for the submission, kept OUTSIDE the uploaded checkpoints tree: next to the
        training output dir, keyed by the submission path so parallel tasks never share it."""
        key = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.abspath(self.submission_dir).strip("/\\"))
        parent = os.path.dirname(os.path.abspath(self.output_dir.rstrip("/\\")))
        return os.path.join(parent, f".publish_meta__{key}.json")

    def _read_publish_floor(self):
        """Epoch progress an earlier attempt published, or None. Ignored unless the submission still
        holds weights (a stale marker from a removed submission must never block publishing)."""
        try:
            if not submission_has_weights(self.submission_dir):
                return None
            with open(self._publish_meta_path()) as f:
                return float(json.load(f)["epoch"])
        except Exception:
            return None

    def _write_publish_meta(self, step: int, epoch: float) -> None:
        try:
            with open(self._publish_meta_path(), "w") as f:
                json.dump({"step": step, "epoch": epoch, "submission_dir": self.submission_dir}, f)
        except Exception as exc:  # the marker is a retry guard only — never fail a save over it
            print(f"[publish] could not write the progress marker: {exc}", flush=True)


class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics):
        eval_loss = None
        if state.log_history:
            last_log_entry = state.log_history[-1]
            eval_loss = last_log_entry.get("eval_reward", None)
            print(f"choose eval_loss ({eval_loss}) as eval_reward from: last_log_entry: {last_log_entry}; \n metrics: {metrics}", flush=True)
        else:
            print(f"state.log_history is empty", flush=True)
            
        if eval_loss is not None:
            eval_loss = - eval_loss
            
        return eval_loss
    
    def penalize_eval_loss(self, eval_loss: float):
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool: 
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)  # Make end_time timezone-aware in UTC
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result =  time_diff.total_seconds() < minutes * 60
    if result:
        print(f"*** current time: {now} end_time: {end_time} time_diff: {time_diff}", flush=True)
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, periodic_save_steps: int = -1, steps_per_epoch: int = -1, max_steps: int = -1, insurance_step: Optional[int] = None):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps
        # One extra save early in the run (submission_utils.insurance_save_step), so a crash before the
        # first periodic save still leaves a trained adapter to ship instead of nothing. None = disabled
        # (the previous schedule: first publish at periodic_save_steps).
        self.insurance_step = insurance_step

    def __call__(self, global_step: int) -> dict:

        if self.steps_per_epoch != -1 and global_step % self.steps_per_epoch == 0 and global_step > 1:
            return {"eval": True, "reason": "epoch"}

        if self.periodic_save_steps != -1 and global_step % self.periodic_save_steps == 0 and global_step > 1:
            return {"eval": True, "reason": "periodic"}

        if self.insurance_step is not None and global_step == self.insurance_step:
            return {"eval": True, "reason": "insurance"}

        if self.save_before_remaining_time > 0 and not self.run_eval:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                print(f"***ALERT: The time is about to run out need to eval & save the model", flush=True)
                # the eval time might be higher than the end_time, so we need to let the pod not stop by setting a flag for this
                self.run_eval = True
                return {"eval": True, "reason": "end_time"}
        
        if self.max_steps != -1 and global_step == self.max_steps:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            return {"eval": True, "reason": "max_step"}

        return {"eval": False, "reason": "none"}


def set_generation_config(model_name, model):
    try:
        if model_name in ERROR_GENERATION_CONFIG_MODELS:
            model.generation_config = GenerationConfig(temperature=None, top_p=None)
    except:
        print(f"Error setting generation config for model {model_name}")
        pass


def resize_if_needed(model_name, model, token_nums):
    try:
        if model_name in MIS_MATCH_VOCAB_SIZE_MODELS:
            model.resize_token_embeddings(token_nums)
    except:
        print(f"Error resizing token embeddings for model {model_name}")
        pass


def init_wandb(train_request: Dict):
    # WandB env vars are set via training_env_vars in text_trainer.py run_training().
    # This function is kept for backwards compatibility with train_instruct.py imports.
    return True


class LossSpikeDetectorCallback(TrainerCallback):
    """
    Three-layer training stability callback.

    Layer 1: Transient grad spike mitigation (on_pre_optimizer_step)
      - Detect via z-score on gradient norm (ZClip-style, Kumar et al. 2025)
      - Action: zero grads → skip update step (correct for numerical spikes)

    Layer 2: Overfitting backtracking (on_evaluate)
      - Detect eval_loss regression from best (HF EarlyStoppingCallback pattern)
      - Action (adapted from AdaptiveCycleCosineLRCallback):
          1. Restore best weights from in-memory cache
          2. Reduce LR by backtrack_lr_factor
          3. Continue training (do NOT stop yet)

    Layer 3: No hard stop — training runs until time budget or max_steps.
      Backtracking retries indefinitely; WhenToEvalHandler owns the stop signal.

    EMA weight tracking (Polyak-Ruppert averaging) as cache source for backtracking.
    """
    def __init__(
        self,
        # Layer 1 params
        window_size: int = 50,
        z_threshold: float = 4.0,
        min_history: int = 10,
        # Layer 1.5 params — AdaGC-style adaptive per-param soft clipping
        # (Wang et al. 2025 https://hf.co/papers/2502.11034)
        hard_zero_z: float = 8.0,
        soft_clip_factor: float = 1.5,
        grad_ema_decay: float = 0.95,
        # Layer 2 params (backtracking)
        enable_backtracking: bool = True,
        eval_min_delta: float = 0.0,
        regression_patience: int = 2,
        backtrack_lr_factor: float = 0.5,
        # Layer 2.5 params — severity-based backtrack tiers
        mild_regression_threshold: float = 0.05,
        moderate_regression_threshold: float = 0.15,
        mild_lr_factor: float = 0.85,
        moderate_lr_factor: float = 0.70,
        # Post-backtrack warmup: linearly ramp from 10% → 100% of new LR over N steps
        post_backtrack_warmup_ratio: float = 0.05,
        # EMA weight tracking (Polyak-Ruppert averaging)
        use_ema: bool = True,
        ema_decay_target: float = 0.999,
        ema_decay_min: float = 0.90,
        ema_warmup_steps: int = 70,
        ema_update_every: int = 1,
    ):
        # Layer 1 state
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.min_history = min_history
        self._grad_history = deque(maxlen=window_size)
        self._spike_count = 0

        # Layer 1.5 state — adaptive grad clipping
        self.hard_zero_z = hard_zero_z
        self.soft_clip_factor = soft_clip_factor
        self._grad_ema_decay = grad_ema_decay
        self._param_grad_ema: dict = {}
        self._original_z_threshold = z_threshold

        # Layer 2 state
        self.enable_backtracking = enable_backtracking
        self.eval_min_delta = eval_min_delta
        self.regression_patience = max(1, regression_patience)
        self.backtrack_lr_factor = backtrack_lr_factor
        self._best_eval_loss = float("inf")
        self._best_state = None
        self._best_step = None
        self._bad_eval_count = 0

        # Layer 2.5 state — severity tiers + post-backtrack warmup
        self.mild_regression_threshold = mild_regression_threshold
        self.moderate_regression_threshold = moderate_regression_threshold
        self.mild_lr_factor = mild_lr_factor
        self.moderate_lr_factor = moderate_lr_factor
        self.post_backtrack_warmup_ratio = post_backtrack_warmup_ratio
        self._warmup_steps_remaining: int = 0
        self._warmup_target_lrs: list = []
        self._warmup_total_steps: int = 1

        # Layer 3 state — no hard stop, logging only
        self._backtrack_count = 0

        # Train-loss trend state
        self._train_loss_window: deque = deque(maxlen=15)
        self._trend_z_tighten: bool = False

        # EMA state
        self.use_ema = use_ema
        self.ema_decay_target = ema_decay_target
        self.ema_decay_min = ema_decay_min
        self.ema_warmup_steps = max(1, ema_warmup_steps)
        self.ema_update_every = max(1, ema_update_every)
        self._ema_state = None
        self._ema_updates = 0

    # --- Layer 1 helpers ---

    def _compute_grad_norm(self, model) -> float:
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.detach().data.norm(2).item()) ** 2
        return total_sq ** 0.5

    def _zero_grads(self, model):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def _soft_clip_grads(self, model) -> int:
        """
        AdaGC-style per-parameter soft gradient clipping.
        (Wang et al. 2025 https://hf.co/papers/2502.11034)
        Returns the number of parameters that were clipped.
        """
        m = model.module if hasattr(model, "module") else model
        clipped = 0
        with torch.no_grad():
            for name, p in m.named_parameters():
                if p.grad is None:
                    continue
                g_norm = float(p.grad.detach().norm(2).item())
                if not math.isfinite(g_norm):
                    p.grad.zero_()
                    clipped += 1
                    continue
                if name not in self._param_grad_ema:
                    self._param_grad_ema[name] = g_norm
                else:
                    self._param_grad_ema[name] = (
                        self._grad_ema_decay * self._param_grad_ema[name]
                        + (1.0 - self._grad_ema_decay) * g_norm
                    )
                threshold = self._param_grad_ema[name] * self.soft_clip_factor
                if threshold > 1e-10 and g_norm > threshold:
                    p.grad.mul_(threshold / g_norm)
                    clipped += 1
        return clipped

    def on_pre_optimizer_step(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Layer 1: detect and skip transient numerical grad spikes."""
        model = kwargs.get("model", None)
        if model is None:
            return control

        total_norm = self._compute_grad_norm(model)

        if not np.isfinite(total_norm):
            self._spike_count += 1
            if state.is_local_process_zero:
                print(
                    f"[LossSpikeDetector] NON-FINITE GRAD #{self._spike_count} "
                    f"step={state.global_step}: norm={total_norm} → zeroing grads, skip step",
                    flush=True,
                )
            self._zero_grads(model)
            return control

        self._grad_history.append(total_norm)

        if len(self._grad_history) < self.min_history:
            return control

        arr = np.array(self._grad_history)
        mean, std = arr.mean(), arr.std()

        if std < 1e-8:
            return control

        z_score = (total_norm - mean) / std
        if z_score > self.z_threshold:
            self._spike_count += 1
            if z_score > self.hard_zero_z:
                if state.is_local_process_zero:
                    print(
                        f"[LossSpikeDetector] GRAD HARDSKIP #{self._spike_count} "
                        f"step={state.global_step}: norm={total_norm:.4f} z={z_score:.2f} "
                        f"(mean={mean:.4f} std={std:.4f}) → zeroing ALL grads, skip step",
                        flush=True,
                    )
                self._zero_grads(model)
                self._grad_history.pop()
            else:
                clipped = self._soft_clip_grads(model)
                if state.is_local_process_zero:
                    print(
                        f"[LossSpikeDetector] GRAD SOFTCLIP #{self._spike_count} "
                        f"step={state.global_step}: norm={total_norm:.4f} z={z_score:.2f} "
                        f"(mean={mean:.4f} std={std:.4f}) → soft-clipped {clipped} params, "
                        f"step proceeds",
                        flush=True,
                    )

        return control

    # --- EMA weight tracking ---

    def _current_ema_decay(self) -> float:
        progress = min(1.0, self._ema_updates / self.ema_warmup_steps)
        decay_range = self.ema_decay_target - self.ema_decay_min
        return self.ema_decay_min + decay_range * (1.0 - math.exp(-5.0 * progress))

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Update EMA weights after each optimizer step."""
        if not self.use_ema:
            return control
        if state.global_step == 0:
            return control
        if state.global_step % self.ema_update_every != 0:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        model_module = model.module if hasattr(model, "module") else model

        try:
            if self._ema_state is None:
                self._ema_state = {}
                for name, param in model_module.named_parameters():
                    if param.requires_grad:
                        self._ema_state[name] = param.data.detach().cpu().clone()
                self._ema_updates = 1
                if state.is_local_process_zero:
                    total_bytes = sum(
                        t.numel() * t.element_size() for t in self._ema_state.values()
                    )
                    print(
                        f"[LossSpikeDetector] EMA initialized step={state.global_step} "
                        f"with {len(self._ema_state)} params "
                        f"(~{total_bytes / (1024 ** 2):.1f} MB CPU RAM)",
                        flush=True,
                    )
                return control

            self._ema_updates += 1
            decay = self._current_ema_decay()
            with torch.no_grad():
                for name, param in model_module.named_parameters():
                    if name in self._ema_state and param.requires_grad:
                        ema_tensor = self._ema_state[name]
                        ema_tensor.mul_(decay).add_(
                            param.data.detach().cpu(), alpha=(1.0 - decay)
                        )

            if state.is_local_process_zero and self._ema_updates % 200 == 0:
                print(
                    f"[LossSpikeDetector] EMA step={state.global_step} "
                    f"updates={self._ema_updates} decay={decay:.6f} "
                    f"{'(warmed up)' if self._ema_updates >= self.ema_warmup_steps else '(warming up)'}",
                    flush=True,
                )
        except Exception as e:
            if state.is_local_process_zero:
                print(
                    f"[LossSpikeDetector] EMA update failed at step "
                    f"{state.global_step}: {e} → disabling EMA",
                    flush=True,
                )
            self.use_ema = False
            self._ema_state = None

        return control

    def export_ema_model(self, output_dir: str):
        """Save EMA state as a model checkpoint in the specified output directory."""
        if not self.use_ema or self._ema_state is None:
            print("[LossSpikeDetector] EMA inactive or empty. Cannot export.", flush=True)
            return False

        ema_dir = os.path.join(output_dir, "ema_checkpoint")
        if os.path.exists(ema_dir):
            shutil.rmtree(ema_dir)
        os.makedirs(ema_dir, exist_ok=True)

        print(f"[LossSpikeDetector] Exporting EMA model to {ema_dir}...", flush=True)
        
        # We need a reference checkpoint to copy structure (tokenizer, config, etc.)
        # Use the submission directory if it exists, otherwise the output dir
        ref_dir = None
        for item in reversed(sorted(os.listdir(output_dir))):
            if item.startswith("checkpoint-"):
                ref_dir = os.path.join(output_dir, item)
                break
        
        if ref_dir:
            for item in os.listdir(ref_dir):
                if not item.endswith(".safetensors") and not item.endswith(".bin") and not item.endswith(".json"):
                    item_path = os.path.join(ref_dir, item)
                    if os.path.isfile(item_path):
                        shutil.copy2(item_path, ema_dir)
                    elif os.path.isdir(item_path):
                        shutil.copytree(item_path, os.path.join(ema_dir, item))
            # copy some jsons specifically
            for js in ["config.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "adapter_config.json"]:
                js_path = os.path.join(ref_dir, js)
                if os.path.exists(js_path):
                    shutil.copy2(js_path, ema_dir)

        # Save weights
        try:
            from safetensors.torch import save_file
            save_file(self._ema_state, os.path.join(ema_dir, "model.safetensors"))
            print(f"[LossSpikeDetector] EMA model saved successfully at {ema_dir}", flush=True)
            return True
        except Exception as e:
            print(f"[LossSpikeDetector] EMA export failed: {e}", flush=True)
            return False

    # --- Layer 2 helpers: state save/restore ---

    def _capture_best_state(self, model) -> dict:
        model_module = model.module if hasattr(model, "module") else model
        state_dict = {}
        for name, param in model_module.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data.detach().cpu().clone()
        return state_dict

    def _restore_best_state(self, model) -> bool:
        if self._best_state is None:
            return False
        model_module = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            for name, param in model_module.named_parameters():
                if name in self._best_state:
                    param.data.copy_(self._best_state[name].to(param.device))
        return True

    def _reset_optimizer_state(self, model, optimizer) -> int:
        if optimizer is None or self._best_state is None:
            return 0
        reset_count = 0
        model_module = model.module if hasattr(model, "module") else model
        for name, param in model_module.named_parameters():
            if name not in self._best_state or not param.requires_grad:
                continue
            if param in optimizer.state:
                optimizer.state.pop(param, None)
                reset_count += 1
        return reset_count

    def _reduce_lr(self, optimizer, factor: Optional[float] = None) -> tuple:
        """Multiply all param_group lrs by factor (defaults to backtrack_lr_factor)."""
        factor = factor if factor is not None else self.backtrack_lr_factor
        old_lrs = []
        new_lrs = []
        for pg in optimizer.param_groups:
            old_lr = float(pg["lr"])
            new_lr = old_lr * factor
            pg["lr"] = new_lr
            old_lrs.append(old_lr)
            new_lrs.append(new_lr)
        return old_lrs, new_lrs

    def _sync_scheduler_after_lr_change(self, lr_scheduler, old_lrs, new_lrs) -> tuple:
        if lr_scheduler is None or not old_lrs or not new_lrs:
            return False, "no_scheduler"

        synced_fields = []
        scale_factors = []
        for old_lr, new_lr in zip(old_lrs, new_lrs):
            if old_lr > 0:
                scale_factors.append(new_lr / old_lr)
            else:
                scale_factors.append(self.backtrack_lr_factor)

        base_lrs = getattr(lr_scheduler, "base_lrs", None)
        if isinstance(base_lrs, list) and len(base_lrs) == len(scale_factors):
            lr_scheduler.base_lrs = [
                float(base_lr) * scale
                for base_lr, scale in zip(base_lrs, scale_factors)
            ]
            synced_fields.append("base_lrs")

        min_lrs = getattr(lr_scheduler, "min_lrs", None)
        if isinstance(min_lrs, list) and len(min_lrs) == len(scale_factors):
            lr_scheduler.min_lrs = [
                float(min_lr) * scale
                for min_lr, scale in zip(min_lrs, scale_factors)
            ]
            synced_fields.append("min_lrs")

        if hasattr(lr_scheduler, "_last_lr"):
            lr_scheduler._last_lr = list(new_lrs)
            synced_fields.append("_last_lr")

        return bool(synced_fields), ",".join(synced_fields) if synced_fields else "unsupported_scheduler"

    # --- Train-loss trend detection ---

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """
        Detect sustained train-loss rise using a rolling window.
        If rising >10%, tighten z_threshold by 30% to catch instability earlier.
        """
        if logs is None:
            return control
        train_loss = logs.get("loss")
        if train_loss is None or not math.isfinite(float(train_loss)):
            return control

        self._train_loss_window.append(float(train_loss))
        if len(self._train_loss_window) < 10:
            return control

        arr = np.array(self._train_loss_window)
        mid = len(arr) // 2
        first_half_mean = arr[:mid].mean()
        second_half_mean = arr[mid:].mean()

        rising = second_half_mean > first_half_mean * 1.10

        if rising and not self._trend_z_tighten:
            self._trend_z_tighten = True
            self.z_threshold = max(2.5, self._original_z_threshold * 0.70)
            if state.is_local_process_zero:
                print(
                    f"[LossSpikeDetector] TRAIN_LOSS RISING step={state.global_step}: "
                    f"first_half={first_half_mean:.4f} second_half={second_half_mean:.4f} "
                    f"→ tightened z_threshold to {self.z_threshold:.2f}",
                    flush=True,
                )
        elif not rising and self._trend_z_tighten:
            self._trend_z_tighten = False
            self.z_threshold = self._original_z_threshold
            if state.is_local_process_zero:
                print(
                    f"[LossSpikeDetector] TRAIN_LOSS STABLE step={state.global_step}: "
                    f"first_half={first_half_mean:.4f} second_half={second_half_mean:.4f} "
                    f"→ restored z_threshold to {self.z_threshold:.2f}",
                    flush=True,
                )
        return control


    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Post-backtrack LR warmup restart: ramp from 10% → 100% over warmup_total_steps."""
        if self._warmup_steps_remaining <= 0:
            return control
        optimizer = kwargs.get("optimizer")
        if optimizer is None or not self._warmup_target_lrs:
            return control

        self._warmup_steps_remaining -= 1
        elapsed = self._warmup_total_steps - self._warmup_steps_remaining
        frac = min(1.0, elapsed / self._warmup_total_steps)
        scale = 0.10 + 0.90 * frac

        for pg, target_lr in zip(optimizer.param_groups, self._warmup_target_lrs):
            pg["lr"] = target_lr * scale

        if self._warmup_steps_remaining == 0:
            if state.is_local_process_zero:
                final_lrs = [f"{t:.6f}" for t in self._warmup_target_lrs]
                print(
                    f"[LossSpikeDetector] POST_BACKTRACK_WARMUP complete "
                    f"step={state.global_step} → lrs={final_lrs}",
                    flush=True,
                )
            self._warmup_target_lrs = []
        return control

    # --- Layer 2 + 3: overfitting response ---

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """
        Layer 2: cache best state on improvement, backtrack on regression.
        Layer 3: no hard stop — retries indefinitely until time/steps budget.
        """
        if metrics is None or "eval_loss" not in metrics:
            return control

        eval_loss = metrics["eval_loss"]
        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)
        lr_scheduler = kwargs.get("lr_scheduler", None)

        # New best path
        improvement_threshold = self._best_eval_loss - self.eval_min_delta
        if eval_loss < improvement_threshold:
            prev_best = self._best_eval_loss
            self._best_eval_loss = eval_loss
            self._best_step = state.global_step
            self._bad_eval_count = 0
            if model is not None:
                try:
                    ema_ready = (
                        self.use_ema
                        and self._ema_state is not None
                        and self._ema_updates >= self.ema_warmup_steps
                    )
                    if ema_ready:
                        self._best_state = {k: v.clone() for k, v in self._ema_state.items()}
                        cache_source = f"EMA (updates={self._ema_updates})"
                    else:
                        self._best_state = self._capture_best_state(model)
                        if self.use_ema:
                            cache_source = (
                                f"current (EMA warming up: "
                                f"{self._ema_updates}/{self.ema_warmup_steps})"
                            )
                        else:
                            cache_source = "current (EMA disabled)"

                    if state.is_local_process_zero:
                        prev_best_str = (
                            f"{prev_best:.4f}" if prev_best != float("inf") else "N/A"
                        )
                        print(
                            f"[LossSpikeDetector] NEW BEST eval_loss={eval_loss:.4f} "
                            f"(prev={prev_best_str}) step={state.global_step} "
                            f"→ cached {len(self._best_state)} params from {cache_source}",
                            flush=True,
                        )
                except Exception as e:
                    if state.is_local_process_zero:
                        print(
                            f"[LossSpikeDetector] Failed to cache best state: {e}",
                            flush=True,
                        )
            return control

        regression_threshold = self._best_eval_loss + self.eval_min_delta
        if eval_loss <= regression_threshold:
            self._bad_eval_count = 0
            if state.is_local_process_zero:
                print(
                    f"[LossSpikeDetector] EVAL PLATEAU eval_loss={eval_loss:.4f} "
                    f"best={self._best_eval_loss:.4f} within min_delta={self.eval_min_delta:.4f} "
                    f"→ no backtrack",
                    flush=True,
                )
            return control

        if not self.enable_backtracking:
            return control

        # Bug fix: suppress backtrack during post-backtrack warmup
        if self._warmup_steps_remaining > 0:
            if state.is_local_process_zero:
                print(
                    f"[LossSpikeDetector] EVAL REGRESSION suppressed — "
                    f"post-backtrack warmup still active "
                    f"({self._warmup_steps_remaining} steps remaining), "
                    f"eval_loss={eval_loss:.4f}",
                    flush=True,
                )
            return control

        self._bad_eval_count += 1
        if self._bad_eval_count < self.regression_patience:
            if state.is_local_process_zero:
                print(
                    f"[LossSpikeDetector] EVAL REGRESSION eval_loss={eval_loss:.4f} "
                    f"best={self._best_eval_loss:.4f} "
                    f"({self._bad_eval_count}/{self.regression_patience}) "
                    f"→ waiting for patience before backtrack",
                    flush=True,
                )
            return control

        if self._best_state is None or model is None or optimizer is None:
            if state.is_local_process_zero:
                print(
                    f"[LossSpikeDetector] EVAL REGRESSION eval_loss={eval_loss:.4f} "
                    f"best={self._best_eval_loss:.4f} — backtracking skipped "
                    f"(best_state={'yes' if self._best_state else 'no'}, "
                    f"model={'yes' if model else 'no'}, "
                    f"optimizer={'yes' if optimizer else 'no'})",
                    flush=True,
                )
            return control

        # Severity-based backtrack tiers
        regression_ratio = (eval_loss - self._best_eval_loss) / (self._best_eval_loss + 1e-8)

        if regression_ratio < self.mild_regression_threshold:
            tier = "MILD"
            lr_factor = self.mild_lr_factor
            do_restore = False
            do_reset_opt = False
            count_as_backtrack = False
        elif regression_ratio < self.moderate_regression_threshold:
            tier = "MODERATE"
            lr_factor = self.moderate_lr_factor
            do_restore = True
            do_reset_opt = False
            count_as_backtrack = True
        else:
            tier = "SEVERE"
            lr_factor = self.backtrack_lr_factor
            do_restore = True
            do_reset_opt = True
            count_as_backtrack = True

        if count_as_backtrack:
            self._backtrack_count += 1

        restored = self._restore_best_state(model) if do_restore else False
        reset_states = (
            self._reset_optimizer_state(model, optimizer)
            if do_reset_opt and restored
            else 0
        )
        old_lrs, new_lrs = self._reduce_lr(optimizer, factor=lr_factor)
        scheduler_synced, scheduler_sync_details = self._sync_scheduler_after_lr_change(
            lr_scheduler, old_lrs, new_lrs
        )
        self._bad_eval_count = 0

        if do_restore and new_lrs:
            remaining_steps = max(10, (state.max_steps or 1000) - state.global_step)
            warmup_n = max(5, int(remaining_steps * self.post_backtrack_warmup_ratio))
            self._warmup_steps_remaining = warmup_n
            self._warmup_total_steps = warmup_n
            self._warmup_target_lrs = list(new_lrs)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * 0.10
            warmup_msg = f"warmup={warmup_n} steps (lr×0.1 → target)"
        else:
            warmup_msg = "no warmup (MILD tier)"

        backtrack_label = (
            f"#{self._backtrack_count}"
            if count_as_backtrack
            else "#MILD(no budget used)"
        )
        if state.is_local_process_zero:
            print(
                f"[LossSpikeDetector] BACKTRACK_{tier} {backtrack_label} "
                f"eval_loss={eval_loss:.4f} regressed from best={self._best_eval_loss:.4f} "
                f"({regression_ratio:.1%}) → "
                f"restored={'yes' if restored else 'no'}, "
                f"reset_optimizer_states={reset_states}, "
                f"new_lrs={[f'{lr:.6f}' for lr in new_lrs]}, "
                f"scheduler_synced={'yes' if scheduler_synced else 'no'}({scheduler_sync_details}), "
                f"{warmup_msg}",
                flush=True,
            )
        return control


class LoRAPlusCallback(TrainerCallback):
    """
    Applies LoRA+ asymmetric learning rates on the very first optimizer step.

    Hayou et al. 2024 (LoRA+, https://hf.co/papers/2402.12354):
      Setting lora_B lr > lora_A lr improves feature learning without extra cost.
      Default ratio used here: lora_B = 4 × lora_A  (B=2×base, A=0.5×base).
    """

    def __init__(self, lora_a_lr: float, lora_b_lr: float):
        self.lora_a_lr = lora_a_lr
        self.lora_b_lr = lora_b_lr
        self._applied = False

    def on_step_begin(self, args, state, control, **kwargs):
        if self._applied:
            return control
        self._applied = True

        optimizer = kwargs.get("optimizer")
        model = kwargs.get("model")
        if optimizer is None or model is None:
            return control

        base = model.module if hasattr(model, "module") else model
        param_to_name = {id(p): n for n, p in base.named_parameters()}

        new_groups = []
        for pg in optimizer.param_groups:
            base_cfg = {k: v for k, v in pg.items() if k != "params"}
            a_params, b_params, other_params = [], [], []
            for p in pg["params"]:
                name = param_to_name.get(id(p), "")
                if "lora_A" in name:
                    a_params.append(p)
                elif "lora_B" in name:
                    b_params.append(p)
                else:
                    other_params.append(p)
            if a_params:
                new_groups.append({**base_cfg, "params": a_params, "lr": self.lora_a_lr})
            if b_params:
                new_groups.append({**base_cfg, "params": b_params, "lr": self.lora_b_lr})
            if other_params:
                new_groups.append({**base_cfg, "params": other_params})

        n_a = sum(len(g["params"]) for g in new_groups if g.get("lr") == self.lora_a_lr)
        n_b = sum(len(g["params"]) for g in new_groups if g.get("lr") == self.lora_b_lr)

        if n_a > 0 or n_b > 0:
            optimizer.param_groups = new_groups
            if state.is_local_process_zero:
                print(
                    f"[LoRA+] Asymmetric LR applied (Hayou et al. 2024): "
                    f"lora_A={self.lora_a_lr:.2e} ({n_a} params), "
                    f"lora_B={self.lora_b_lr:.2e} ({n_b} params), "
                    f"ratio={self.lora_b_lr / self.lora_a_lr:.1f}x",
                    flush=True,
                )
        else:
            if state.is_local_process_zero:
                print(
                    "[LoRA+] No lora_A/lora_B params found in optimizer - "
                    "LoRA+ not applied (full-weight or non-standard LoRA naming).",
                    flush=True,
                )
        return control


class SWA_CyclicLRCallback(TrainerCallback):
    """
    Kicks in at the end of training to collect diverse checkpoints for SWA.
    Oscillates LR to jump out of local minima.
    """
    def __init__(self, cycle_start_epoch: float = 2.8, cycle_steps: int = 50, lr_max_factor: float = 1.2, lr_min_factor: float = 0.5):
        self.cycle_start_epoch = cycle_start_epoch
        self.cycle_steps = cycle_steps
        self.lr_max_factor = lr_max_factor
        self.lr_min_factor = lr_min_factor
        self._is_cycling = False
        self._anchor_lr = None

    def on_step_begin(self, args, state, control, **kwargs):
        if state.epoch < self.cycle_start_epoch:
            return control
        
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return control
        
        if not self._is_cycling:
            self._is_cycling = True
            # Get the LR before we start messing with it
            self._anchor_lr = optimizer.param_groups[0]["lr"]
            if state.is_local_process_zero:
                print(f"[SWA_Cyclic] Starting LR cycles at epoch {state.epoch:.2f}, anchor_lr={self._anchor_lr:.2e}", flush=True)
        
        # Cosine oscillation within steps
        cycle_progress = (state.global_step % self.cycle_steps) / self.cycle_steps
        factor = self.lr_min_factor + 0.5 * (self.lr_max_factor - self.lr_min_factor) * (1 + math.cos(2 * math.pi * cycle_progress))
        
        for pg in optimizer.param_groups:
            pg["lr"] = self._anchor_lr * factor
            
        return control