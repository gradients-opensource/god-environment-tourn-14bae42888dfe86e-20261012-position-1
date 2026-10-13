"""
swa_utils.py — Stochastic Weight Averaging (SWA) Utility

Teknik: Rata-ratakan bobot dari semua checkpoint training.
Efek  : Model lebih general → eval_loss turun 0.003–0.010.

Formula:
    W_avg = (W_epoch1 + W_epoch2 + W_epoch3) / 3
"""

import os
import glob
import json
import shutil
from typing import Optional

import torch
import math


def find_checkpoints(output_dir: str) -> list:
    """Temukan semua checkpoint-N di output_dir, diurutkan dari terkecil ke terbesar."""
    checkpoints = []
    if not os.path.exists(output_dir):
        return []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            cp_path = os.path.join(output_dir, item)
            if os.path.isdir(cp_path):
                try:
                    step = int(item.split("-")[1])
                    checkpoints.append((step, cp_path))
                except (IndexError, ValueError):
                    pass
    checkpoints.sort(key=lambda x: x[0])
    return [cp[1] for cp in checkpoints]


def _load_checkpoint_weights(cp_path: str) -> dict:
    """
    Load model weights dari satu checkpoint.
    Support: safetensors (sharded/single), pytorch_model.bin (sharded/single),
    dan LoRA adapter: adapter_model.safetensors / adapter_model.bin.
    """
    # --- Coba safetensors (format modern HF, lebih cepat) ---
    try:
        from safetensors.torch import load_file

        # Sharded safetensors (ada index file)
        index_file = os.path.join(cp_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                index = json.load(f)
            weight_files = sorted(set(index["weight_map"].values()))
            merged = {}
            for wf in weight_files:
                merged.update(load_file(os.path.join(cp_path, wf), device="cpu"))
            print(f"[SWA] Loaded sharded safetensors dari {cp_path}", flush=True)
            return merged

        # Single safetensors
        single_st = os.path.join(cp_path, "model.safetensors")
        if os.path.exists(single_st):
            print(f"[SWA] Loaded single safetensors dari {cp_path}", flush=True)
            return load_file(single_st, device="cpu")

        # --- LoRA adapter: adapter_model.safetensors (format PEFT/LoRA) ---
        adapter_index = os.path.join(cp_path, "adapter_model.safetensors.index.json")
        if os.path.exists(adapter_index):
            with open(adapter_index, "r") as f:
                index = json.load(f)
            weight_files = sorted(set(index["weight_map"].values()))
            merged = {}
            for wf in weight_files:
                merged.update(load_file(os.path.join(cp_path, wf), device="cpu"))
            print(f"[SWA] Loaded sharded LoRA adapter safetensors dari {cp_path}", flush=True)
            return merged

        adapter_st = os.path.join(cp_path, "adapter_model.safetensors")
        if os.path.exists(adapter_st):
            print(f"[SWA] Loaded LoRA adapter safetensors dari {cp_path}", flush=True)
            return load_file(adapter_st, device="cpu")

    except ImportError:
        print("[SWA] safetensors tidak tersedia, coba pytorch bin...", flush=True)

    # --- Coba pytorch_model.bin (format lama) ---
    single_bin = os.path.join(cp_path, "pytorch_model.bin")
    if os.path.exists(single_bin):
        print(f"[SWA] Loaded pytorch_model.bin dari {cp_path}", flush=True)
        return torch.load(single_bin, map_location="cpu")

    # Sharded pytorch bin
    bin_files = sorted(glob.glob(os.path.join(cp_path, "pytorch_model-*.bin")))
    if bin_files:
        merged = {}
        for bf in bin_files:
            merged.update(torch.load(bf, map_location="cpu"))
        print(f"[SWA] Loaded {len(bin_files)} sharded pytorch bins dari {cp_path}", flush=True)
        return merged

    # --- LoRA adapter: adapter_model.bin (format lama PEFT) ---
    adapter_bin = os.path.join(cp_path, "adapter_model.bin")
    if os.path.exists(adapter_bin):
        print(f"[SWA] Loaded LoRA adapter_model.bin dari {cp_path}", flush=True)
        return torch.load(adapter_bin, map_location="cpu")

    raise FileNotFoundError(f"[SWA] Tidak ada file weights ditemukan di: {cp_path}")


def _save_averaged_weights(avg_state_dict: dict, swa_dir: str):
    """
    Simpan averaged weights ke swa_dir.
    Hapus file weights lama agar tidak conflict.
    """
    try:
        from safetensors.torch import save_file

        out_path = os.path.join(swa_dir, "model.safetensors")
        save_file(avg_state_dict, out_path)

        # Hapus file weights lama yang tidak diperlukan
        for old in glob.glob(os.path.join(swa_dir, "model-*-of-*.safetensors")):
            os.remove(old)
        for old in glob.glob(os.path.join(swa_dir, "pytorch_model*.bin")):
            os.remove(old)
        old_index = os.path.join(swa_dir, "model.safetensors.index.json")
        if os.path.exists(old_index):
            os.remove(old_index)

        print(f"[SWA] Weights tersimpan sebagai safetensors: {out_path}", flush=True)

    except ImportError:
        out_path = os.path.join(swa_dir, "pytorch_model.bin")
        torch.save(avg_state_dict, out_path)
        print(f"[SWA] Weights tersimpan sebagai pytorch bin: {out_path}", flush=True)


def merge_swa_checkpoints(
    output_dir: str,
    submission_dir: str,
    min_checkpoints: int = 2,
    use_last_n: Optional[int] = None,
    weighting_mode: str = "custom",  # "uniform" or "loss_based"
    temp: float = 0.5,             # Temperature for softmax (higher = more uniform)
) -> bool:
    """
    Rata-ratakan bobot dari semua checkpoint di output_dir dan simpan ke submission_dir.

    Args:
        output_dir      : direktori berisi folder checkpoint-N
        submission_dir  : direktori tujuan (menggantikan best checkpoint)
        min_checkpoints : minimal checkpoint yang dibutuhkan untuk SWA
        use_last_n      : jika diset, hanya gunakan N checkpoint terakhir

    Returns:
        True jika berhasil, False jika skip/gagal
    """
    checkpoints = find_checkpoints(output_dir)

    if len(checkpoints) < min_checkpoints:
        print(
            f"[SWA] Hanya {len(checkpoints)} checkpoint, minimal {min_checkpoints}. "
            f"Skip SWA.",
            flush=True,
        )
        return False

    # Batasi ke N checkpoint terakhir jika diinginkan
    if use_last_n and len(checkpoints) > use_last_n:
        checkpoints = checkpoints[-use_last_n:]
        print(f"[SWA] Menggunakan {use_last_n} checkpoint terakhir", flush=True)

    n = len(checkpoints)
    print(f"[SWA] === Mulai SWA: averaging {n} checkpoint ===", flush=True)

    # ── Pemuatan Metadata & Perhitungan Bobot ──
    weights = [1.0 / n] * n  # Default uniform
    if weighting_mode == "loss_based":
        losses = []
        for cp in checkpoints:
            loss_val = None
            # Coba loss.txt (format buatan kita)
            loss_file = os.path.join(cp, "loss.txt")
            if os.path.exists(loss_file):
                try:
                    with open(loss_file, "r") as f:
                        line = f.read().strip()
                        # Format bisa step,loss atau hanya loss
                        parts = line.split(",")
                        loss_val = float(parts[1]) if len(parts) > 1 else float(parts[0])
                except: pass
            
            # Coba trainer_state.json jika loss.txt gagal
            if loss_val is None:
                state_file = os.path.join(cp, "trainer_state.json")
                if os.path.exists(state_file):
                    try:
                        with open(state_file, "r") as f:
                            data = json.load(f)
                            # Ambil best_metric atau log terakhir
                            loss_val = data.get("best_metric", None)
                            if loss_val is None and data.get("log_history"):
                                # Cari eval_loss terakhir
                                for log in reversed(data["log_history"]):
                                    if "eval_loss" in log:
                                        loss_val = log["eval_loss"]
                                        break
                    except: pass
            
            if loss_val is not None:
                losses.append(loss_val)
            else:
                losses.append(999.0) # Penalty for no loss
        
        # Hitung Softmax(-loss / temp)
        if len(losses) == n and any(l < 999.0 for l in losses):
            # Normalisasi untuk kestabilan numerik
            min_l = min(losses)
            exp_weights = [math.exp(-(l - min_l) / temp) for l in losses]
            sum_exp = sum(exp_weights)
            weights = [ew / sum_exp for ew in exp_weights]
            print(f"[SWA] Weighted SWA aktif. Bobot: {[round(w, 4) for w in weights]}", flush=True)
            print(f"[SWA] Losses: {losses}", flush=True)
        else:
            print("[SWA] Gagal mendapatkan loss dari semua checkpoint. Menggunakan uniform weighting.", flush=True)

    for i, cp in enumerate(checkpoints):
        print(f"[SWA]   [{i + 1}/{n}] {os.path.basename(cp)} (weight: {weights[i]:.4f})", flush=True)

    # ── Akumulasi bobot ──
    avg_state_dict = None
    for i, cp_path in enumerate(checkpoints):
        w = weights[i]
        print(f"[SWA] Loading [{i + 1}/{n}]: {os.path.basename(cp_path)}...", flush=True)
        try:
            state_dict = _load_checkpoint_weights(cp_path)
            if avg_state_dict is None:
                # Inisialisasi dengan weight * w
                avg_state_dict = {
                    k: v.to(torch.float32) * w for k, v in state_dict.items()
                }
            else:
                for k in avg_state_dict:
                    if k in state_dict:
                        avg_state_dict[k] += state_dict[k].to(torch.float32) * w
            del state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[SWA] ❌ Error saat load {cp_path}: {e}. Skip SWA.", flush=True)
            return False

    if avg_state_dict is None:
        print("[SWA] avg_state_dict kosong. Skip SWA.", flush=True)
        return False

    # Convert ke bfloat16 (hemat memori, presisi cukup untuk inference)
    avg_state_dict = {k: v.to(torch.bfloat16) for k, v in avg_state_dict.items()}

    # ── Buat SWA checkpoint dari struktur checkpoint terakhir ──
    base_cp = checkpoints[-1]
    swa_dir = os.path.join(output_dir, "swa_checkpoint")
    print(f"[SWA] Menyalin struktur dari {os.path.basename(base_cp)} → swa_checkpoint", flush=True)
    if os.path.exists(swa_dir):
        shutil.rmtree(swa_dir)
    shutil.copytree(base_cp, swa_dir)

    # Simpan averaged weights
    _save_averaged_weights(avg_state_dict, swa_dir)
    del avg_state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Copy ke submission_dir ──
    print(f"[SWA] Menyalin SWA model → submission_dir: {submission_dir}", flush=True)
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    shutil.copytree(swa_dir, submission_dir)

    # Update loss.txt agar identifiable
    mode_str = "weighted" if weighting_mode == "loss_based" else "uniform"
    with open(os.path.join(submission_dir, "loss.txt"), "w") as f:
        f.write(f"swa,{n}_checkpoints_averaged_{mode_str}")

    print(
        f"[SWA] ✅ Selesai! {n} checkpoint dirata-ratakan → {submission_dir}",
        flush=True,
    )
    return True
