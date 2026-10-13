FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1

# System dependencies not in axolotl base image.
# Trimmed 2026-09-16 (Fase 0 Docker-slim, <1800s build cap): dropped dev/monitoring
# tools (vim/tmux/iotop/nvtop/bmon/nano/zsh/htop) that training never uses. Kept
# redis-server (entrypoint.sh daemonizes it), wget (asset fetch), zip (output packaging).
RUN apt-get update && apt-get install -y \
    zip \
    wget \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Default dir
RUN mkdir -p /workspace /cache /workspace/scripts/datasets /app/checkpoints

# axo_py venv for axolotl tokenization — reuse base image's torch/cuda via system-site-packages
RUN python -m venv --system-site-packages /workspace/axo_py
RUN bash -c "source /workspace/axo_py/bin/activate && \
    pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    uv pip install --no-build-isolation \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --index-strategy unsafe-best-match \
        axolotl==0.9.1 && \
    uv pip install \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        --index-strategy unsafe-best-match \
        torchvision && \
    pip install requests==2.32.3 && \
    deactivate"

# Install main training dependencies (copy requirements file saja, bukan seluruh scripts)
COPY scripts/training_requirements.txt /workspace/scripts/training_requirements.txt
# NOTE (Fase 0 Docker-slim, 2026-09-16): vllm==0.8.3 REMOVED. It was the dominant
# build-time cost — it requires torch 2.6.0 and forced a multi-GB torch reinstall +
# xformers/flash-attn recompile ON TOP of the axolotl base's torch 2.5.1, pushing the
# cold build past the validator's 1800s cap (Sep-14 forfeit). vllm is used ONLY by
# TRL's GRPO fast-generation path; this image hardcodes ENV_SKIP_GRPO=1 (SFT-only), no
# Python module `import vllm`s directly, so it is dead weight here. If a GRPO/RL image
# is ever needed, add it back in a separate build. Keeping torch at the base's 2.5.1
# avoids the swap entirely.
RUN pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    uv pip install -r /workspace/scripts/training_requirements.txt --system && \
    pip install hf_transfer==0.1.9 && \
    pip install tenacity==9.1.2 && \
    pip install tiktoken==0.9.0 && \
    pip install "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0" && \
    uv pip install open_spiel --system

# --- InterCode: NO eval-filesystem bake ---
# We deliberately do NOT clone princeton-nlp/intercode or materialize the eval's /testbed,
# /system, /workspace filesystems here. Per the validator's 2026-09 rule, replicating the
# eval environment to ground training data reveals the test set and is DQ-able. Intercode SFT
# rows are built at runtime from the validator's WHITELISTED passthrough dataset only
# (scripts/envs/intercode_dataset.py → eval-exact execute_bash/submit tool-call rows).

# Copy scripts LAST — script changes don't invalidate the cache layers above.
# NB: NO training dataset is bundled in this image. Per the env-tournament rules ("do not
# bundle your own dataset in the Docker image"), ALL SFT data is generated at RUNTIME inside
# the training window (scripts/envs/generate_trajectories.py) or loaded from the validator's
# whitelisted MINER_DATASETS. scripts/envs/prebaked/ and scripts/datasets/ are excluded via
# .dockerignore so no data file can leak into the image.
COPY scripts /workspace/scripts

# Make entrypoint script executable
RUN chmod +x /workspace/scripts/entrypoint.sh

WORKDIR /workspace/scripts

# EnvTask training mode. ENV_SKIP_GRPO=1 → SFT-only (imitation warm-start is the final model,
# no GRPO RL phase). Chosen for tight wall-clock budgets (e.g. 1.5h) where RL can't converge.
# Set to 0 (or remove this line) to restore the SFT warm-start → GRPO RL pipeline.
ENV ENV_SKIP_GRPO=1

ENTRYPOINT ["./entrypoint.sh"]
