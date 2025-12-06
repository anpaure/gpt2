import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import wandb

from model import GPT, GPTConfig, TokenDataset


def main():
    # ---- mixed-precision / TF32 prefs ----
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    config = GPTConfig()

    block_size = config.n_ctx        # 1024
    batch_size = 128                 # you can probably push this on H200
    max_steps = 30000

    # base LR for AdamW; Muon gets a higher one
    adam_lr = 3e-4
    muon_lr_factor = 4.0
    muon_lr = muon_lr_factor * adam_lr

    eval_interval = 2500
    max_eval_steps = 10
    ckpt_interval = 5000

    # trapezoidal LR schedule hyperparams (fractions of total steps)
    warmup_ratio = 0.01              # 1% warmup
    warmdown_ratio = 0.20            # last 20% linearly decay

    tokens_path = "data/processed/openwebtext_gpt2_5B.bin"

    # ---- device ----
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    print(f"Using device: {device}")

    # ---- wandb ----
    run = wandb.init(
        project="gpt2-small",
        config={
            "model": "custom_gpt2_small",
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "d_model": config.d_model,
            "d_ff": config.d_ff,
            "n_ctx": config.n_ctx,
            "vocab_size": config.vocab_size,
            "batch_size": batch_size,
            "adam_lr": adam_lr,
            "muon_lr": muon_lr,
            "muon_lr_factor": muon_lr_factor,
            "max_steps": max_steps,
            "eval_interval": eval_interval,
            "max_eval_steps": max_eval_steps,
            "device": str(device),
            "dtype": "bfloat16",
            "tokens_path": tokens_path,
            "warmup_ratio": warmup_ratio,
            "warmdown_ratio": warmdown_ratio,
        },
    )

    # ---- per-run checkpoint directory ----
    run_id = run.id or time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join("checkpoints", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- load preprocessed tokens ----
    assert os.path.exists(tokens_path), f"tokens file not found: {tokens_path}"

    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    print("Loaded tokens:", tokens.shape, tokens.dtype)

    # 90/10 split by token index
    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    train_dataset = TokenDataset(train_tokens, block_size)
    val_dataset = TokenDataset(val_tokens, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
    )

    # ---- model ----
    model = GPT(config).to(device)

    # cast heavy matmul parts to bf16
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.Linear)):
            m.to(dtype=torch.bfloat16)

    # ---- Muon + AdamW param split (before compile) ----
    muon_params = []
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_matrix = (p.ndim == 2)
        is_embedding_or_head = (
            name.startswith("transformer.wte")
            or name.startswith("transformer.wpe")
            or name.startswith("lm_head")
        )

        # All 2D hidden weights (attn/MLP/etc.) → Muon
        if is_matrix and not is_embedding_or_head:
            muon_params.append(p)
        else:
            adam_params.append(p)

    print(f"Muon params: {len(muon_params)}, AdamW params: {len(adam_params)}")

    # torch.compile for CUDA
    if device.type == "cuda":
        model = torch.compile(model, dynamic=False)

    model.train()

    # ---- optimizers ----
    optimizer_adam = AdamW(
        adam_params,
        lr=adam_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    optimizer_muon = Muon(
        muon_params,
        lr=muon_lr,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        adjust_lr_fn="match_rms_adamw",
    )

    optimizers = [optimizer_adam, optimizer_muon]

    # ---- trapezoidal LR ----
    warmup_steps = max(1, int(warmup_ratio * max_steps)) if warmup_ratio > 0 else 0
    warmdown_steps = max(1, int(warmdown_ratio * max_steps)) if warmdown_ratio > 0 else 0

    def lr_multiplier(step_idx: int) -> float:
        # [0, warmup_steps): linear warmup 0 → 1
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)

        plateau_start = warmup_steps
        warmdown_start = max_steps - warmdown_steps

        # [warmup_steps, warmdown_start): flat
        if step_idx < warmdown_start:
            return 1.0

        # [warmdown_start, max_steps): linear decay 1 → 0
        if warmdown_steps > 0 and step_idx < max_steps:
            remaining = max_steps - step_idx
            return max(0.0, float(remaining) / float(warmdown_steps))

        return 0.0

    scheduler_adam = LambdaLR(optimizer_adam, lr_lambda=lr_multiplier)
    scheduler_muon = LambdaLR(optimizer_muon, lr_lambda=lr_multiplier)
    schedulers = [scheduler_adam, scheduler_muon]

    step = 0
    start_time = time.time()

    # ---- training loop ----
    while step < max_steps:
        for x, y in train_loader:
            step += 1

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            for opt in optimizers:
                opt.step()

            for sched in schedulers:
                sched.step()

            # ---- train logging ----
            if step % 10 == 0:
                lr_adam = scheduler_adam.get_last_lr()[0]
                lr_muon = scheduler_muon.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(
                    f"Step: {step}, loss: {loss.item():.4f}, "
                    f"lr_adam: {lr_adam:.6e}, lr_muon: {lr_muon:.6e}"
                )
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr_adam": lr_adam,
                        "train/lr_muon": lr_muon,
                        "train/step": step,
                        "train/time_elapsed_s": elapsed,
                    },
                    step=step,
                )

            # ---- checkpoint ----
            if step % ckpt_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"gpt2_small_step_{step}.pt")
                try:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": config.__dict__,
                            "step": step,
                        },
                        ckpt_path,
                    )
                    print(f"Saved checkpoint to {ckpt_path}")
                    wandb.save(ckpt_path)
                except Exception as e:
                    print(f"[WARN] Failed to save checkpoint at step {step}: {e}")

            # ---- eval ----
            if step % eval_interval == 0:
                model.eval()
                losses = []
                eval_step = 0
                with torch.no_grad():
                    for x_eval, y_eval in val_loader:
                        eval_step += 1
                        x_eval = x_eval.to(device, non_blocking=True)
                        y_eval = y_eval.to(device, non_blocking=True)
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                        ):
                            _, eval_loss = model(x_eval, y_eval)
                        losses.append(eval_loss.item())
                        if eval_step >= max_eval_steps:
                            break

                val_loss = sum(losses) / len(losses)
                elapsed = time.time() - start_time
                model.train()

                print(
                    f"Eval step: {step}, val_loss {val_loss:.4f}, "
                    f"time elapsed: {elapsed:.1f}s"
                )
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/step": step,
                        "val/time_elapsed_s": elapsed,
                    },
                    step=step,
                )

                if device.type == "mps":
                    print(
                        f"VRAM usage, current allocated: "
                        f"{torch.mps.current_allocated_memory() / (1024 ** 2):.2f}MB, "
                        f"driver allocated: "
                        f"{torch.mps.driver_allocated_memory() / (1024 ** 2):.2f}MB"
                    )

            if step >= max_steps:
                break

    # ---- final checkpoint ----
    final_ckpt_path = os.path.join(ckpt_dir, "gpt2_small_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "step": step,
        },
        final_ckpt_path,
    )
    print("Saved final checkpoint to", final_ckpt_path)
    wandb.save(final_ckpt_path)
    wandb.finish()


if __name__ == "__main__":
    main()
