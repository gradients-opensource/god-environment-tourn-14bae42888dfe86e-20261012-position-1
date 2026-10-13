import time
import logging
import numpy as np
import torch
import gc
import os
from dataclasses import dataclass
from torch.optim import AdamW

# Prefer 8-bit Adam (bitsandbytes) for the LR-finder sweep:
# moment vectors are quantised to int8 → ~4× less VRAM than AdamW.
# Falls back to standard AdamW if bitsandbytes is not installed.
try:
    import bitsandbytes as bnb
    _ADAM_CLS = bnb.optim.AdamW8bit
    _ADAM_LABEL = "AdamW8bit (bitsandbytes)"
except ImportError:
    from torch.optim import AdamW as _AdamW
    _ADAM_CLS = _AdamW
    _ADAM_LABEL = "AdamW (fallback — install bitsandbytes for lower VRAM usage)"

logger = logging.getLogger(__name__)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


def _is_ddp_model(model) -> bool:
    """Check if the model is wrapped in DistributedDataParallel."""
    return hasattr(model, "module") and isinstance(
        model.module, torch.nn.Module
    )


def _get_base_model(model):
    """Unwrap DDP if needed to get the actual model."""
    if _is_ddp_model(model):
        return model.module
    return model


def _dist_avg_scalar(value: float) -> float:
    """Average a scalar across all DDP ranks (no-op if not distributed)."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and WORLD_SIZE > 1:
            t = torch.tensor(value, dtype=torch.float64, device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            return t.item()
    except Exception:
        pass
    return value


def _dist_barrier():
    """Barrier across all ranks (no-op if not distributed)."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and WORLD_SIZE > 1:
            dist.barrier()
    except Exception:
        pass


def analyze_loss_landscape(model, batch, lr, trainer, num_points=5, radius=0.1, lora=True):
    """
    Quick analysis of the loss landscape around the given learning rate.
    DDP-safe: works on the unwrapped base model.

    Returns:
        Dictionary with landscape metrics
    """
    base_model = _get_base_model(model)

    # Save original parameters (always use base model's named_parameters)
    if lora:
        original_params = {
            name: param.detach().clone()
            for name, param in base_model.named_parameters()
            if "lora_" in name or "lm_head" in name
        }
    else:
        original_params = {
            name: param.detach().clone()
            for name, param in base_model.named_parameters()
        }

    # Get initial loss and gradient
    loss = trainer.compute_loss(model, batch)
    loss.backward()

    # Calculate gradient norm for relevant parameters
    grad_params = []
    for name, param in base_model.named_parameters():
        if param.grad is not None:
            if lora:
                if "lora_" in name or "lm_head" in name:
                    grad_params.append(torch.norm(param.grad))
            else:
                grad_params.append(torch.norm(param.grad))

    grad_norm = torch.norm(torch.stack(grad_params)) if grad_params else torch.tensor(0.0)

    # Evaluate loss at different learning rates around the current one
    lr_factors = np.linspace(1 - radius, 1 + radius, num_points)
    losses = []

    for factor in lr_factors:
        current_lr = lr * factor

        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if param.grad is not None:
                    if lora:
                        if "lora_" in name or "lm_head" in name:
                            param.data = original_params[name] - current_lr * param.grad
                    else:
                        param.data = original_params[name] - current_lr * param.grad

        loss_val = float(trainer.compute_loss(model, batch))
        # Average loss across ranks for a consistent landscape
        loss_val = _dist_avg_scalar(loss_val)
        losses.append(loss_val)

        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])

    losses = np.array(losses)
    min_idx = np.argmin(losses)

    smoothness = np.mean(np.abs(np.diff(losses, 2))) if len(losses) > 2 else float("inf")
    convexity = np.mean(np.diff(losses, 2)) if len(losses) > 2 else 0
    is_centered = min_idx in [num_points // 2 - 1, num_points // 2, num_points // 2 + 1]

    metrics = {
        "smoothness": smoothness,
        "convexity": convexity,
        "centered": is_centered,
        "grad_norm": float(grad_norm),
        "min_loss": float(np.min(losses)),
        "min_lr_factor": lr_factors[min_idx],
    }

    del original_params
    torch.cuda.empty_cache()

    return metrics


def find_lr_and_continue(
    trainer,
    start_lr: float = 1e-6,
    end_lr: float = 1e-3,
    time_budget_minutes: float = 5.0,
    warmup_fraction: float = 0.1,
    num_candidates: int = 2,
    preserve_state: bool = True,
    lora: bool = True,
    use_gradient_checkpointing: bool = False,
    gpu_count: int = 0,
    max_evals: int = 30,
):
    """
    Multi-GPU-safe LR finder based on Leslie Smith's approach.

    Multi-GPU strategy:
    - All ranks run the LR finder in PARALLEL (no idle ranks).
    - Each rank explores LRs on its own local data shard.
    - At the end, the chosen LR is averaged across all ranks via all_reduce.
    - A dist.barrier() is inserted at critical points to keep ranks in sync.

    The key insight: since trainer.compute_loss() already triggers
    gradient all-reduce in DDP, all ranks must call it at the same step and
    with the same number of steps, so they stay in sync naturally.

    Args:
        trainer: HuggingFace Trainer instance
        start_lr: Minimum LR to test
        end_lr: Maximum LR to test
        time_budget_minutes: Max time for the LR finder phase
        warmup_fraction: Unused, kept for API compatibility
        num_candidates: Unused, kept for API compatibility
        preserve_state: Whether to save and restore model state after finder
        lora: If True, only track LoRA and lm_head parameters
        gpu_count: Number of GPUs. If 1, uses plain AdamW per-eval.
                   If >1, uses AdamW8bit with reusable optimizer.
                   If 0 (default), falls back to WORLD_SIZE env var.

    Returns:
        Tuple of (selected_lr, best_state_dict, None, num_evaluations)
    """
    # Determine single-GPU mode from gpu_count param or env var fallback
    _single_gpu = (gpu_count == 1) if gpu_count > 0 else (WORLD_SIZE == 1)
    model = trainer.model
    base_model = _get_base_model(model)

    time_budget_seconds = time_budget_minutes * 60
    start_time = time.time()

    rank = LOCAL_RANK
    is_main_rank = (rank == 0)

    # Enable gradient checkpointing to reduce activation memory during sweep
    if use_gradient_checkpointing:
        try:
            base_model.gradient_checkpointing_enable()
            if is_main_rank:
                logger.info("[LR Finder] Gradient checkpointing enabled inside finder")
        except Exception as _gc_err:
            if is_main_rank:
                logger.warning(f"[LR Finder] Could not enable gradient checkpointing: {_gc_err}")

    if is_main_rank:
        logger.info(
            f"[LR Finder] Starting — WORLD_SIZE={WORLD_SIZE}, "
            f"start_lr={start_lr:.2e}, end_lr={end_lr:.2e}, "
            f"time_budget={time_budget_minutes:.1f}min, lora={lora}"
        )

    # ------------------------------------------------------------------ #
    # 1. Save model state (always from base model to avoid DDP naming)     #
    # ------------------------------------------------------------------ #
    if lora:
        initial_state = {
            name: param.detach().cpu().clone()
            for name, param in base_model.named_parameters()
            if ("lora_" in name or "lm_head" in name) and param.requires_grad
        }
    else:
        initial_state = {
            name: param.detach().cpu().clone()
            for name, param in base_model.named_parameters()
            if param.requires_grad
        }

    best_state_dict = None

    # ------------------------------------------------------------------ #
    # 2. Measure time-per-step to budget evaluations                      #
    # ------------------------------------------------------------------ #
    test_steps = 3
    dataloader_iter = iter(trainer.get_train_dataloader())
    batch = next(dataloader_iter)
    batch = trainer._prepare_inputs(batch)

    model.train()

    # Fix 1: Use no_grad for timing to avoid allocating AdamW optimizer states
    # on GPU (moment vectors alone can exhaust remaining VRAM on large models).
    # Forward-pass time is proportional to full step time, which is sufficient
    # to estimate the per-step budget.
    test_start_time = time.time()
    for _ in range(test_steps):
        with torch.no_grad():
            _ = trainer.compute_loss(model, batch)

    time_per_step = (time.time() - test_start_time) / test_steps
    if is_main_rank:
        logger.info(f"[LR Finder] Time per step: {time_per_step:.4f}s")

    main_time_budget = time_budget_seconds * 0.8
    steps_per_evaluation = 5  # real gradient steps per LR evaluation (defined here for planning)
    # The timing probe above used torch.no_grad() (no backward pass) to avoid
    # allocating AdamW moment vectors on the GPU during measurement.  Each real
    # sweep evaluation runs steps_per_evaluation full steps (forward + backward
    # + optimizer.step + cache-clear), which takes roughly 3× as long as a
    # no_grad forward.  Apply a ×3 correction so the plan matches reality.
    _backward_correction = 3
    time_per_eval_estimate = time_per_step * steps_per_evaluation * _backward_correction
    max_evaluations = int(main_time_budget / time_per_eval_estimate)
    max_evaluations = min(max(10, max_evaluations), max_evals)

    if is_main_rank:
        logger.info(
            f"[LR Finder] Planning up to {max_evaluations} evaluations "
            f"(~{time_per_eval_estimate:.1f}s/eval est.)"
        )

    # ------------------------------------------------------------------ #
    # 3. Restore initial parameters before the sweep starts               #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if lora and ("lora_" in name or "lm_head" in name) and name in initial_state:
                param.copy_(initial_state[name].to(param.device))
            elif not lora and name in initial_state:
                param.copy_(initial_state[name].to(param.device))

    # Fix 3: Aggressively free any cached allocations before the sweep.
    # On near-full GPUs the difference between "fits" and OOM is a few MiB.
    gc.collect()
    torch.cuda.empty_cache()

    # Sync all ranks before the sweep
    _dist_barrier()

    # ------------------------------------------------------------------ #
    # 4. LR range sweep — all ranks run simultaneously                    #
    # ------------------------------------------------------------------ #
    lr_multiplier = np.power(end_lr / start_lr, 1 / max_evaluations)

    lrs = []
    losses = []
    smoothed_losses = []
    grad_norms = []   # gradient L2 norm per eval — leading instability indicator (Wortsman 2023)

    best_loss = float("inf")
    min_loss_so_far = float("inf")
    divergence_counter = 0
    current_lr = start_lr
    evaluation_count = 0

    # ------------------------------------------------------------------ #
    # 4b. Create the sweep optimizer                                      #
    #     Single-GPU  → plain AdamW, recreated each eval (simpler)        #
    #     Multi-GPU   → AdamW8bit, created ONCE and reused (saves VRAM)   #
    # ------------------------------------------------------------------ #
    _sweep_optimizer = None
    _sweep_orig_lrs = []
    if not _single_gpu:
        try:
            _trainable_params = [p for p in model.parameters() if p.requires_grad]
            _sweep_optimizer = _ADAM_CLS(_trainable_params, lr=current_lr)
            _sweep_orig_lrs = [pg["lr"] for pg in _sweep_optimizer.param_groups]
            if is_main_rank:
                logger.info(f"[LR Finder] Sweep optimizer: {_ADAM_LABEL} (multi-GPU mode)")
        except Exception as _opt_init_err:
            if is_main_rank:
                logger.warning(
                    f"[LR Finder] Could not create sweep optimizer ({_opt_init_err}). "
                    "Will skip parameter updates during sweep — loss curve may be less accurate."
                )
    else:
        if is_main_rank:
            logger.info("[LR Finder] Using plain AdamW (single-GPU mode)")

    while evaluation_count < max_evaluations and time.time() - start_time < main_time_budget:
        evaluation_count += 1

        if is_main_rank:
            logger.info(
                f"[LR Finder] Eval {evaluation_count}/{max_evaluations}, LR: {current_lr:.2e}"
            )

        # --- Single-GPU path: fresh AdamW per evaluation ---
        if _single_gpu:
            optimizer = AdamW(model.parameters(), lr=current_lr)
            batch_losses = []

            for step in range(steps_per_evaluation):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(trainer.get_train_dataloader())
                    batch = next(dataloader_iter)

                batch = trainer._prepare_inputs(batch)
                optimizer.zero_grad()
                loss = trainer.compute_loss(model, batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss))

            del optimizer
            torch.cuda.empty_cache()

        # --- Multi-GPU path: reuse AdamW8bit optimizer ---
        else:
            if _sweep_optimizer is not None:
                for pg in _sweep_optimizer.param_groups:
                    pg["lr"] = current_lr
                _sweep_optimizer.zero_grad()

            batch_losses = []

            for step in range(steps_per_evaluation):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(trainer.get_train_dataloader())
                    batch = next(dataloader_iter)

                batch = trainer._prepare_inputs(batch)
                loss = trainer.compute_loss(model, batch)
                loss.backward()
                batch_losses.append(float(loss))

            if _sweep_optimizer is not None:
                _sweep_optimizer.step()
            else:
                with torch.no_grad():
                    for p in base_model.parameters():
                        if p.requires_grad and p.grad is not None:
                            grad_norm = p.grad.norm() + 1e-8
                            p.data -= current_lr * (p.grad / grad_norm)

        avg_loss = np.mean(batch_losses)

        # Average the loss across all ranks so every rank agrees on the
        # divergence check — critical for keeping all ranks in sync.
        avg_loss = _dist_avg_scalar(avg_loss)

        # --- Gradient norm (leading instability indicator, Wortsman et al. 2023) ---
        # Compute L2 norm of all trainable gradients still present from the last
        # backward pass.  This grows BEFORE loss visibly spikes, giving us an early
        # signal of the instability boundary.
        _gn_sq = 0.0
        for _gp in base_model.parameters():
            if _gp.requires_grad and _gp.grad is not None:
                _gn_sq += _gp.grad.detach().float().pow(2).sum().item()
        eval_grad_norm = _gn_sq ** 0.5
        eval_grad_norm = _dist_avg_scalar(eval_grad_norm)
        grad_norms.append(eval_grad_norm)

        lrs.append(current_lr)
        losses.append(avg_loss)

        if len(losses) == 1:
            smoothed_losses.append(avg_loss)
        else:
            smoothed_losses.append(0.7 * smoothed_losses[-1] + 0.3 * avg_loss)

        if is_main_rank:
            logger.info(
                f"[LR Finder] LR: {current_lr:.2e}, Loss: {avg_loss:.4f}, "
                f"Smoothed: {smoothed_losses[-1]:.4f}, GradNorm: {eval_grad_norm:.4f}"
            )

        if preserve_state and avg_loss < best_loss:
            if best_state_dict is not None:
                del best_state_dict
                torch.cuda.empty_cache()
            best_loss = avg_loss
            best_state_dict = {
                name: param.detach().cpu().clone()
                for name, param in base_model.named_parameters()
                if param.requires_grad
            }

        if avg_loss < min_loss_so_far:
            min_loss_so_far = avg_loss
            divergence_counter = 0
        else:
            if avg_loss > min_loss_so_far * 1.10:
                divergence_counter += 1
            else:
                divergence_counter = 0

        if avg_loss > 6.0 * min_loss_so_far or np.isnan(avg_loss) or np.isinf(avg_loss):
            if is_main_rank:
                logger.info(
                    f"[LR Finder] Early stop — significant divergence at LR: {current_lr:.2e}"
                )
            # Barrier so all ranks break at the same eval
            _dist_barrier()
            break

        if divergence_counter >= 8:
            if is_main_rank:
                logger.info(
                    f"[LR Finder] Early stop — consistent loss increase at LR: {current_lr:.2e}"
                )
            _dist_barrier()
            break

        current_lr *= lr_multiplier

    # ------------------------------------------------------------------ #
    # 5. Select optimal LR from the sweep                                 #
    # ------------------------------------------------------------------ #
    if len(lrs) < 3:
        if is_main_rank:
            logger.warning("[LR Finder] Too few evaluations — using default LR")
        final_lr = 8e-5
        finder_meta = {"unreliable": True, "reason": "too_few_evals", "exploration_quality": 0.0}
    else:
        smoothed_losses_arr = np.array(smoothed_losses)
        lrs_arr = np.array(lrs)

        gradients = np.gradient(smoothed_losses_arr) / np.gradient(np.log10(lrs_arr))

        window_size = min(3, len(gradients) // 4)
        windowed_gradients = [
            np.mean(gradients[max(0, i - window_size): min(len(gradients), i + window_size + 1)])
            for i in range(len(gradients))
        ]
        steepest_idx = np.argmin(windowed_gradients)
        min_loss_idx = np.argmin(smoothed_losses_arr)

        if len(smoothed_losses_arr) > 5:
            # Use the RAW loss peak (not smoothed[-1]) as the numerator.
            # The EMA smoother (0.7/0.3) dampens a real spike like this:
            #   raw=1.58  →  smoothed=1.32  →  ratio drops from 1.41 to 1.18
            # Using smoothed[-1] caused a genuine curve to be misclassified as
            # "no spike" and fell back to the default LR incorrectly.
            # Denominator 3.0: a raw peak 3× the smoothed minimum = definite spike.
            # Denominator 10.0 was set for raw/raw comparison — now that we use
            # raw_peak / smoothed_min, 3.0 is the correct scale.
            raw_peak = max(losses)
            smoothed_min = float(np.min(smoothed_losses_arr))
            loss_end_to_min_ratio = raw_peak / (smoothed_min + 1e-8)
            loss_exploration_quality = min(1.0, loss_end_to_min_ratio / 3.0)
        else:
            loss_exploration_quality = 0.5

        # Fix 4: Inflection-point-aware LR selection for better L2 loss.
        # The inflection point is where the 2nd derivative of the smoothed loss
        # turns positive — i.e., loss stops curving down and starts curving up.
        # This is the boundary of the convex bowl; we must stay left of it.
        # np.diff(arr, 2) produces len-2 array, so index offsets by +2.
        second_deriv = np.diff(smoothed_losses_arr, 2)
        inflection_candidates = np.where(second_deriv > 0)[0]
        if len(inflection_candidates) > 0:
            # +2 to account for the 2-step index shift from np.diff(n=2)
            inflection_idx = int(inflection_candidates[0]) + 2
        else:
            inflection_idx = len(lrs_arr)  # no inflection found — use full range

        # Reliability check: did the sweep actually find the instability region?
        # A good sweep sees loss drop to a minimum then rise back up significantly.
        # loss_exploration_quality = min(1.0, end_loss/min_loss / 10.0)
        #   < 0.2 → end_loss < 2 × min_loss → loss never rose enough → unreliable
        # This catches two failure modes with a single condition:
        #   1. True monotonic decrease (no inflection, time-limited or range too narrow)
        #   2. Spurious early inflection + time-limit stop before the real spike
        # In both cases the finder cannot reliably identify the optimal LR, so
        # the caller should fall back to the default lr_utils LR.
        no_loss_spike = loss_exploration_quality < 0.2
        if no_loss_spike:
            if inflection_idx == len(lrs_arr):
                _reason = "monotonic_decrease"   # curve never curved back, no inflection at all
            else:
                _reason = "no_loss_spike"         # early curvature change but loss still falling
        else:
            _reason = "ok"
        finder_meta = {
            "unreliable": no_loss_spike,
            "reason": _reason,
            "exploration_quality": float(loss_exploration_quality),
        }
        if no_loss_spike and is_main_rank:
            logger.warning(
                f"[LR Finder] No loss spike detected — sweep stopped before instability region "
                f"(quality={loss_exploration_quality:.3f}, reason={_reason}) — "
                "caller should fall back to default LR"
            )

        if is_main_rank:
            logger.info(
                f"[LR Finder] Inflection point at idx={inflection_idx} "
                f"(LR≈{lrs_arr[min(inflection_idx, len(lrs_arr)-1)]:.2e})"
            )

        if steepest_idx > 1:
            if loss_exploration_quality > 0.8:
                divider = 1.3
            else:
                divider = 1.8

            # Stay safely on the descending side of the inflection point
            safe_idx = min(steepest_idx, max(0, inflection_idx - 2))
            candidate_lr = lrs_arr[safe_idx] / divider
            min_bound = lrs_arr[min_loss_idx] / 2.0
            final_lr = max(candidate_lr, min_bound)

            if is_main_rank:
                logger.info(
                    f"[LR Finder] Steepest descent at LR: {lrs_arr[steepest_idx]:.2e}, "
                    f"safe_idx={safe_idx}, "
                    f"min loss at LR: {lrs_arr[min_loss_idx]:.2e}, "
                    f"selected: {final_lr:.2e} (divider={divider:.1f})"
                )
        else:
            final_lr = lrs_arr[min_loss_idx] / 2.0
            if is_main_rank:
                logger.info(f"[LR Finder] Limited data — selected LR based on min loss: {final_lr:.2e}")

        # ---------------------------------------------------------------- #
        # 5b. Gradient-norm fallback (Wortsman et al. 2023)                #
        # If no loss spike was detected (time budget hit before instability #
        # region was reached), check gradient norm growth as a leading      #
        # indicator.  A ≥30% step-over-step growth signals the onset of    #
        # instability even before loss visibly rises.                       #
        # ---------------------------------------------------------------- #
        gn_usable = len(grad_norms) >= 4 and len(grad_norms) == len(lrs)
        if no_loss_spike and gn_usable:
            gn_arr = np.array(grad_norms)
            gn_ratios = gn_arr[1:] / (gn_arr[:-1] + 1e-8)
            max_growth_idx = int(np.argmax(gn_ratios))
            max_growth = float(gn_ratios[max_growth_idx])
            if max_growth >= 1.3:   # ≥30% growth in one eval = approaching boundary
                # Instability starts at max_growth_idx+1; stay one step left of it
                safe_idx = max(0, max_growth_idx - 1)
                peak_lr_idx = min(max_growth_idx + 1, len(lrs_arr) - 1)
                gn_based_lr = lrs_arr[safe_idx] / 1.5
                if is_main_rank:
                    logger.info(
                        f"[LR Finder] Grad-norm signal: peak growth ×{max_growth:.2f} "
                        f"at LR={lrs_arr[peak_lr_idx]:.2e} → "
                        f"selected {gn_based_lr:.2e} (Wortsman et al. 2023)"
                    )
                final_lr = gn_based_lr
                finder_meta["unreliable"] = False
                finder_meta["reason"] = "grad_norm_signal"
                finder_meta["grad_norm_peak_lr"] = float(lrs_arr[peak_lr_idx])
            else:
                if is_main_rank:
                    logger.info(
                        f"[LR Finder] Grad-norm growth insufficient "
                        f"(max ×{max_growth:.2f} < 1.30) — fallback LR will be used"
                    )

    # ------------------------------------------------------------------ #
    # 6. Optional landscape validation (rank 0 only, then broadcast)      #
    # ------------------------------------------------------------------ #
    remaining_time = time_budget_seconds - (time.time() - start_time)

    # Free VRAM before landscape analysis (multi-GPU only) to avoid OOM.
    # Sweep optimizer moment vectors can hold several GiB; clearing them
    # here gives landscape analysis enough headroom.
    if not _single_gpu and _sweep_optimizer is not None:
        try:
            _sweep_optimizer.state.clear()
            if is_main_rank:
                logger.info("[LR Finder] Cleared sweep optimizer state before landscape analysis")
        except Exception:
            pass
    if not _single_gpu:
        gc.collect()
        torch.cuda.empty_cache()

    if remaining_time > 60:
        # Only rank 0 does landscape analysis (read-only with respect to loss)
        # We still call compute_loss which needs all ranks in sync.
        try:
            batch = next(dataloader_iter)
            batch = trainer._prepare_inputs(batch)

            with torch.no_grad():
                for name, param in base_model.named_parameters():
                    if lora and ("lora_" in name or "lm_head" in name) and name in initial_state:
                        param.copy_(initial_state[name].to(param.device))
                    elif not lora and name in initial_state:
                        param.copy_(initial_state[name].to(param.device))

            landscape = analyze_loss_landscape(model, batch, final_lr, trainer, lora=lora)

            if not landscape["centered"]:
                adjustment = max(0.6, min(1.5, landscape["min_lr_factor"]))
                if is_main_rank:
                    logger.info(f"[LR Finder] Landscape adjustment factor: {adjustment:.2f}")
                final_lr *= adjustment

            model_size_based_min = 5e-6
            try:
                if hasattr(base_model, "config") and hasattr(base_model.config, "hidden_size"):
                    if base_model.config.hidden_size > 2048:
                        model_size_based_min = 1e-5
            except Exception:
                pass

            if final_lr < model_size_based_min:
                if is_main_rank:
                    logger.info(
                        f"[LR Finder] Clamping LR to minimum: {model_size_based_min:.2e}"
                    )
                final_lr = model_size_based_min

            if is_main_rank:
                logger.info(
                    f"[LR Finder] Landscape: smoothness={landscape['smoothness']:.4f}, "
                    f"convexity={landscape['convexity']:.4f}, centered={landscape['centered']}"
                )
        except Exception as e:
            if is_main_rank:
                logger.warning(f"[LR Finder] Landscape analysis skipped: {e}")

    # ------------------------------------------------------------------ #
    # 7. Apply small boost and ALL-REDUCE across ranks                    #
    # ------------------------------------------------------------------ #
    final_lr *= 1
    if is_main_rank:
        logger.info(f"[LR Finder] No small boost trust all: {final_lr:.2e}")

    # Average the final LR across all ranks — each rank did independent search
    # on its own data shard, so we take the consensus value.
    final_lr = _dist_avg_scalar(final_lr)

    if is_main_rank:
        logger.info(f"[LR Finder] Final LR after all-reduce: {final_lr:.2e}")

    # ------------------------------------------------------------------ #
    # 8. Restore original model parameters                                #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if lora and ("lora_" in name or "lm_head" in name) and name in initial_state:
                param.copy_(initial_state[name].to(param.device))
            elif not lora and name in initial_state:
                param.copy_(initial_state[name].to(param.device))

    # Final barrier — all ranks must complete the restore before training starts
    _dist_barrier()

    del initial_state

    # Clear the sweep optimizer's accumulated moment vectors so they don't
    # linger in VRAM and don't bias the real training run's optimizer steps.
    # (Only needed for multi-GPU path; single-GPU deletes optimizer each eval)
    if not _single_gpu and _sweep_optimizer is not None:
        try:
            _sweep_optimizer.state.clear()
            for pg, orig_lr in zip(_sweep_optimizer.param_groups, _sweep_orig_lrs):
                pg["lr"] = orig_lr
            if is_main_rank:
                logger.info("[LR Finder] Sweep optimizer state cleared.")
        except Exception as _clr_err:
            if is_main_rank:
                logger.warning(f"[LR Finder] Could not clear sweep optimizer state: {_clr_err}")
        _sweep_optimizer = None

    torch.cuda.empty_cache()
    gc.collect()

    if is_main_rank:
        logger.info(f"[LR Finder] Done. Selected LR = {final_lr:.2e}")

    return final_lr, best_state_dict, finder_meta, evaluation_count
