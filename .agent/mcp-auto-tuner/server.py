from mcp.server.fastmcp import FastMCP
import subprocess
import json
import re
import os
import wandb
from huggingface_hub import HfApi

mcp = FastMCP("AutoTuner")

@mcp.tool()
def read_config(file_path: str) -> str:
    """Membaca isi dari sebuah file konfigurasi (seperti scripts/dpo_config.py)"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Gagal membaca file: {str(e)}"

@mcp.tool()
def modify_config_regex(file_path: str, pattern: str, replacement: str) -> str:
    """
    Memodifikasi file konfigurasi menggunakan metode regex (regular expression).
    Sangat berguna untuk mengubah hyperparameter tanpa merusak struktur kode Python (AST).
    Sediakan path absolut file, pola regex yang dicari, dan teks penggantinya secara tepat.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content, num_subs = re.subn(pattern, replacement, content)
        
        if num_subs == 0:
            return f"Tidak ada kecocokan yang ditemukan untuk pola '{pattern}' pada file {file_path}. Tidak ada perubahan."
            
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return f"Berhasil memperbarui file {file_path}. Membuat {num_subs} perubahan penggantian."
    except Exception as e:
        return f"Gagal memodifikasi file: {str(e)}"

@mcp.tool()
def run_training_trial(train_script_cmd: str, max_steps: int = 100) -> str:
    """
    Menjalankan jalannya percobaan (trial) skrip pelatihan secara singkat.
    Akan otomatis menambahkan flag --max_steps pada baris perintah untuk mencegah pelatihan berdurasi penuh.
    Merekam teks output (stdout/stderr) untuk mendeteksi eror OOM atau model yang konvergensinya rusak (meledak).
    """
    cmd = f"{train_script_cmd} --max_steps {max_steps}"
    try:
        result = subprocess.run(
            cmd, 
            shell=True,
            capture_output=True,
            text=True
        )
        return json.dumps({
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:]
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def read_latest_eval_loss(log_dir: str) -> dict:
    """
    Membaca riwayat file trainer_state.json terbaru dari folder hasil (output text) Huggingface/TRL 
    guna mengekstraksi nilai akhir metrik 'eval_loss' dan 'training loss'.
    """
    state_file = None
    if os.path.exists(log_dir):
        for root, dirs, files in os.walk(log_dir):
            if 'trainer_state.json' in files:
                current_file = os.path.join(root, 'trainer_state.json')
                if not state_file or os.path.getmtime(current_file) > os.path.getmtime(state_file):
                    state_file = current_file
                
    if not state_file:
        return {"error": f"File trainer_state.json tidak ditemukan sama sekali di dalam {log_dir}"}
        
    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
            log_history = data.get("log_history", [])
            result = {"file_read": state_file}
            for log in reversed(log_history):
                if "eval_loss" in log:
                    result["latest_eval_loss"] = log["eval_loss"]
                    result["eval_step"] = log.get("step")
                    break
            for log in reversed(log_history):
                if "loss" in log:
                    result["latest_train_loss"] = log["loss"]
                    result["train_loss_step"] = log.get("step")
                    break
            return result
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def check_wandb_run(entity: str, project: str, run_id: str = None) -> str:
    """
    Menarik (Fetch) metrik pelaporan terakhir dari Weights & Biases (W&B) lewat API resmi.
    Berguna jika hasil training diunggah ke *cloud* W&B dan bukan hanya di lokal.
    Memerlukan WANDB_API_KEY disetel di sistem Pengguna.
    Jika run_id dikosongi, fungsi akan menarik metrik *Run* paling baru dari keseluruhan proyek.
    """
    try:
        api = wandb.Api()
        if run_id:
            run = api.run(f"{entity}/{project}/{run_id}")
            metrics = {k: v for k, v in run.summary.items() if not k.startswith("_")}
            return json.dumps({"name": run.name, "metrics": metrics, "state": run.state}, indent=2)
        else:
            runs = api.runs(f"{entity}/{project}")
            if not runs:
                return f"Tidak ada Run eksperimen sama sekali di proyek W&B {entity}/{project}"
            last_run = runs[0]
            metrics = {k: v for k, v in last_run.summary.items() if not k.startswith("_")}
            return json.dumps({"name": last_run.name, "metrics": metrics, "state": last_run.state}, indent=2)
    except Exception as e:
        return f"Gagal membaca dari layanan cloud WandB API: {str(e)}"

@mcp.tool()
def upload_to_huggingface(output_model_dir: str, repo_id: str, hf_token: str) -> str:
    """
    Otomatis mengunggah seluruh direktori output/ckpt LORA dan konfigurasinya ke repositori Huggingface Hub secara utuh.
    WAJIB dipanggil jika auto-tuning sudah menemukan hiperparameter emas dan model Full Training sukses diselesaikan.
    """
    try:
        api = HfApi(token=hf_token)
        url = api.upload_folder(
            folder_path=output_model_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        return f"SUKSES! Berkas model berhasil diunggah dengan sempurna ke Cloud HuggingFace: {url}"
    except Exception as e:
        return f"Gagal menyinkronkan/mengunggah ke jaringan HuggingFace Hub: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport='stdio')
