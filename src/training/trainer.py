"""
Training loop for 3D brain tumor segmentation.
Logs to TensorBoard (always) and optionally W&B.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from monai.data import DataLoader, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

CLASS_NAMES = ["NCR/NET", "Edema", "Enhancing Tumor"]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        run_name: str | None = None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.run_name = run_name or f"{cfg['model']}_{cfg['loss']}"

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg.get("weight_decay", 1e-5),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg["max_epochs"],
            eta_min=1e-6,
        )
        # BF16 doesn't suffer from the underflow that requires loss scaling,
        # but GradScaler is a no-op on BF16 — kept for FP16 fallback compatibility.
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = GradScaler("cuda", enabled=(self.amp_dtype == torch.float16))

        self.inferer = SlidingWindowInferer(
            roi_size=cfg["patch_size"],
            sw_batch_size=cfg.get("sw_batch_size", 2),
            overlap=cfg.get("sw_overlap", 0.5),
            mode="gaussian",
        )

        out_channels = cfg["out_channels"]
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=True,
        )
        self.post_pred  = Compose([AsDiscrete(argmax=True, to_onehot=out_channels)])
        self.post_label = Compose([AsDiscrete(to_onehot=out_channels)])

        # cuDNN benchmark caches the fastest conv algorithm at startup; that algorithm
        # requires a fixed workspace size that may become unavailable after many epochs
        # of memory fragmentation → CUDNN_STATUS_EXECUTION_FAILED_CUDART.
        torch.backends.cudnn.benchmark = False

        self.output_dir = Path(cfg.get("output_dir", "outputs")) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_dsc = -1.0
        self.global_step  = 0

        # TensorBoard — always on, logs to outputs/<run_name>/tb_logs/
        tb_dir = self.output_dir / "tb_logs"
        self.writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"  TensorBoard logs → {tb_dir}", flush=True)
        print(f"  Run:  tensorboard --logdir {Path(cfg.get('output_dir', 'outputs'))}", flush=True)

        # W&B — optional
        self.use_wandb = cfg.get("use_wandb", False)
        if self.use_wandb:
            import wandb
            wandb.init(
                project=cfg.get("wandb_project", "brats-segmentation"),
                name=self.run_name,
                config={**cfg, "run_name": self.run_name},
                reinit=True,
            )
            self._wandb_watch_done = False

    # ------------------------------------------------------------------
    def train(self) -> float:
        """Run full training. Returns best val DSC."""
        max_epochs   = self.cfg["max_epochs"]
        val_interval = self.cfg.get("val_interval", 5)

        print(f"\n{'='*60}", flush=True)
        print(f"  Run:   {self.run_name}", flush=True)
        print(f"  Model: {self.cfg['model']}  |  Loss: {self.cfg['loss']}", flush=True)
        print(f"  LR:    {self.cfg['lr']}  |  Epochs: {max_epochs}", flush=True)
        print(f"{'='*60}", flush=True)

        # Log hyperparameters to TensorBoard
        self.writer.add_text("config/model",    self.cfg["model"],    0)
        self.writer.add_text("config/loss",     self.cfg["loss"],     0)
        self.writer.add_text("config/lr",       str(self.cfg["lr"]),  0)
        self.writer.add_text("config/patch_size",
                             str(self.cfg["patch_size"]), 0)

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            epoch_secs = time.time() - t0
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # TensorBoard — epoch-level training metrics
            self.writer.add_scalar("train/epoch_loss", train_loss,  epoch)
            self.writer.add_scalar("train/lr",         current_lr,  epoch)

            if self.use_wandb:
                import wandb
                wandb.log({
                    "train/epoch_loss": train_loss,
                    "train/lr":         current_lr,
                    "epoch":            epoch,
                })

            print(f"  Ep {epoch:03d}/{max_epochs}  loss={train_loss:.4f}  "
                  f"lr={current_lr:.2e}  time={epoch_secs/60:.1f}min", flush=True)

            if epoch % val_interval == 0:
                val_dsc, per_class = self._validate(epoch)
                if val_dsc > self.best_val_dsc:
                    self.best_val_dsc = val_dsc
                    self._save_checkpoint("best_model.pth")
                    print(f"  => New best val DSC: {val_dsc:.4f} — checkpoint saved", flush=True)

        self._save_checkpoint("last_model.pth")
        self.writer.add_hparams(
            hparam_dict={
                "model":      self.cfg["model"],
                "loss":       self.cfg["loss"],
                "lr":         self.cfg["lr"],
                "patch_size": str(self.cfg["patch_size"]),
                "batch_size": self.cfg["batch_size"],
            },
            metric_dict={"hparam/best_val_dsc": self.best_val_dsc},
        )
        self.writer.close()
        print(f"\nDone. Best val DSC: {self.best_val_dsc:.4f}", flush=True)

        if self.use_wandb:
            import wandb
            wandb.run.summary["best_val_dsc"] = self.best_val_dsc
            wandb.finish()

        return self.best_val_dsc

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        torch.cuda.empty_cache()
        running_loss = 0.0
        n_steps      = 0
        log_interval  = self.cfg.get("log_interval", 10)
        grad_accum    = max(1, self.cfg.get("grad_accum_steps", 1))

        # W&B model watch (once)
        if self.use_wandb and not self._wandb_watch_done:
            import wandb
            wandb.watch(self.model, log="gradients", log_freq=50)
            self._wandb_watch_done = True

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}", leave=False, ncols=80)
        for micro_step, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast("cuda", dtype=self.amp_dtype):
                preds = self.model(images)
                # Divide loss so gradients sum to the correct scale after accum
                loss  = self.loss_fn(preds, labels) / grad_accum

            self.scaler.scale(loss).backward()

            is_last_batch    = (micro_step + 1) == len(self.train_loader)
            do_optimizer_step = ((micro_step + 1) % grad_accum == 0) or is_last_batch

            if do_optimizer_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                n_steps      += 1
                self.global_step += 1

            loss_val = loss.item() * grad_accum  # rescale for logging
            running_loss += loss_val
            mem_gb = torch.cuda.memory_reserved(self.device) / 1e9
            pbar.set_postfix(loss=f"{loss_val:.4f}", vram=f"{mem_gb:.1f}GB")

            # TensorBoard — step-level loss (log on optimizer steps only)
            if do_optimizer_step and n_steps % log_interval == 0:
                self.writer.add_scalar("train/step_loss", loss_val, self.global_step)

            if self.use_wandb and do_optimizer_step and n_steps % log_interval == 0:
                import wandb
                wandb.log({"train/step_loss": loss_val})

        return running_loss / max(micro_step + 1, 1)

    # ------------------------------------------------------------------
    def _validate(self, epoch: int) -> tuple[float, dict]:
        self.model.eval()
        self.dice_metric.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="  Val", leave=False, ncols=80):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                with autocast("cuda", dtype=self.amp_dtype):
                    preds = self.inferer(images, self.model)

                preds_post  = [self.post_pred(p)  for p in decollate_batch(preds)]
                labels_post = [self.post_label(l) for l in decollate_batch(labels)]
                self.dice_metric(y_pred=preds_post, y=labels_post)

        metric, _  = self.dice_metric.aggregate()
        mean_dsc   = metric.nanmean().item()
        per_class  = {name: metric[i].item() for i, name in enumerate(CLASS_NAMES)}

        # Console
        parts = "  |  ".join(f"{n}: {v:.4f}" for n, v in per_class.items())
        print(f"  Val  Ep {epoch:03d} | {parts}  |  Mean: {mean_dsc:.4f}", flush=True)

        # TensorBoard — val metrics
        self.writer.add_scalar("val/mean_dsc", mean_dsc, epoch)
        for name, val in per_class.items():
            self.writer.add_scalar(f"val/dsc_{name}", val, epoch)

        torch.cuda.empty_cache()

        if self.use_wandb:
            import wandb
            log_dict = {"epoch": epoch, "val/mean_dsc": mean_dsc}
            for name, val in per_class.items():
                log_dict[f"val/dsc_{name}"] = val
            wandb.log(log_dict)

        return mean_dsc, per_class

    # ------------------------------------------------------------------
    def _save_checkpoint(self, filename: str) -> None:
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_dsc":         self.best_val_dsc,
                "cfg":                  self.cfg,
                "run_name":             self.run_name,
            },
            self.output_dir / filename,
        )
