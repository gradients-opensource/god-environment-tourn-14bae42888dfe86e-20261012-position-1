import argparse
import sys
import json
import requests
from huggingface_hub import HfApi, model_info

def check_model_integrity(model_id):
    print(f"--- Menjalankan Verifikasi untuk: {model_id} ---")
    api = HfApi()
    
    # 1. Cek Visibilitas
    try:
        info = model_info(model_id)
        if info.private:
            print("❌ ERROR: Model bersifat PRIVATE. Tournament tidak akan bisa mengaksesnya.")
            return False
        else:
            print("✅ SUCCESS: Model bersifat PUBLIC.")
    except Exception as e:
        print(f"❌ ERROR: Tidak dapat mengakses model. Apakah namanya benar? ({e})")
        return False

    # 2. Cek File config.json dan model_type
    config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        response = requests.get(config_url)
        if response.status_code == 200:
            config = response.json()
            if "model_type" in config:
                print(f"✅ SUCCESS: File config.json ditemukan dengan model_type: '{config['model_type']}'")
            else:
                print("❌ ERROR: File config.json ditemukan TAPI TIDAK MEMILIKI 'model_type'.")
                print("   Ini akan menyebabkan error 'Unrecognized model' di Tournament.")
                return False
        else:
            # Mungkin ini adapter? Cek adapter_config.json
            adapter_url = f"https://huggingface.co/{model_id}/raw/main/adapter_config.json"
            adapter_resp = requests.get(adapter_url)
            if adapter_resp.status_code == 200:
                print("ℹ️  INFO: config.json tidak ditemukan, tapi adapter_config.json ditemukan (Mode PEFT/LoRA).")
            else:
                print("❌ ERROR: File config.json tidak ditemukan.")
                return False
    except Exception as e:
        print(f"❌ ERROR: Gagal memvalidasi config.json: {e}")
        return False

    # 3. Cek File Tokenizer
    tokenizer_url = f"https://huggingface.co/{model_id}/raw/main/tokenizer_config.json"
    try:
        if requests.get(tokenizer_url).status_code == 200:
            print("✅ SUCCESS: tokenizer_config.json ditemukan.")
        else:
            print("⚠️  WARNING: tokenizer_config.json tidak ditemukan. Beberapa sistem eval mungkin gagal.")
    except:
        pass

    print("\n🎉 KESIMPULAN: Model ini SIAP untuk dievaluasi.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifikasi Integritas Model Hugging Face")
    parser.add_argument("model_id", help="Nama repository (contoh: user/repo)")
    args = parser.parse_args()
    
    success = check_model_integrity(args.model_id)
    if not success:
        sys.exit(1)
