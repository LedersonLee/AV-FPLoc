#!/usr/bin/env python3
"""Train and evaluate a fresh UnLoc head on HSSD RGB observations.

The dependency-free metric helpers in this module define the evaluation
contract used by both Lightning callbacks and post-training reports.  Heavy
dependencies are intentionally imported only by the training entry point so
the contract can be tested on the host without a CUDA container.
"""

from __future__ import annotations

import json
from pathlib import Path
import argparse
import os
import sys

import numpy as np


QUALITATIVE_LOG_EVERY_N_EPOCHS = 5


def initialize_loader_worker(_worker_id: int) -> None:
    """Avoid multiplying OpenCV/BLAS thread pools in every loader worker."""
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def tee_training_output(run_root: Path) -> None:
    """Persist console output alongside checkpoints, also under detached Docker."""
    run_root.mkdir(parents=True, exist_ok=True)
    stream = (run_root / "train.log").open("a", buffering=1)

    class Tee:
        def __init__(self, original):
            self.original = original

        def write(self, text):
            stream.write(text)
            return self.original.write(text)

        def flush(self):
            stream.flush()
            self.original.flush()

        def __getattr__(self, name):
            return getattr(self.original, name)

    sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)


def resolve_adapter_path(value: str, *, hssd_root: Path) -> Path:
    """Resolve canonical HSSD-relative paths; retain legacy absolute paths."""

    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(hssd_root).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"adapter path outside HSSD root: {value}")
    return resolved


def load_adapter_frame_rows(adapter_root: Path, split: str) -> list[dict[str, object]]:
    """Load the one-to-one HSSD frame index without importing Torch.

    Every accepted RGB has exactly one metadata row and a row in its scene's
    ``depth40.npy``.  Keeping the loader dependency-free makes it possible to
    validate the exported dataset before entering a CUDA container.
    """

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    root = Path(adapter_root)
    hssd_root = root.parent.parent
    manifest_path = root / "manifests" / f"{split}_scenes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError(f"adapter split manifest lacks scenes: {manifest_path}")
    rows: list[dict[str, object]] = []
    for scene in scenes:
        if not isinstance(scene, dict) or not isinstance(scene.get("scene_id"), str):
            raise ValueError("adapter scene manifest has an invalid scene record")
        metadata_path = resolve_adapter_path(str(scene.get("sample_metadata_path", "")), hssd_root=hssd_root)
        metadata_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line]
        depth40_path = resolve_adapter_path(str(scene.get("depth40_path") or metadata_path.parent / "depth40.npy"), hssd_root=hssd_root)
        if not depth40_path.is_file():
            raise FileNotFoundError(f"{scene['scene_id']}: missing depth40 target file: {depth40_path}")
        expected = int(scene.get("accepted_frames", -1))
        if expected < 1 or len(metadata_rows) != expected:
            raise ValueError(f"{scene['scene_id']}: adapter frame count does not match its scene manifest")
        for row in metadata_rows:
            if not isinstance(row, dict):
                raise ValueError(f"{scene['scene_id']}: invalid frame metadata row")
            image_path = resolve_adapter_path(str(row.get("image_path", "")), hssd_root=hssd_root)
            rows.append({**row, "image_path": str(image_path), "scene_id": str(scene["scene_id"]), "depth40_path": str(depth40_path)})
    if not rows:
        raise ValueError(f"adapter split contains no frames: {split}")
    return rows


def assert_fresh_unloc_head(checkpoint: str | None, *, init_unloc_checkpoint: str | None) -> None:
    """Reject loading any existing UnLoc-head checkpoint for this experiment."""

    if checkpoint is not None or init_unloc_checkpoint is not None:
        raise ValueError("this run requires a fresh UnLoc head; prior UnLoc checkpoints are not permitted")


def angular_error_degrees(predicted_deg: np.ndarray, ground_truth_deg: np.ndarray) -> np.ndarray:
    """Return the shortest unsigned angular distance in degrees, in [0, 180]."""

    predicted = np.asarray(predicted_deg, dtype=np.float64)
    ground_truth = np.asarray(ground_truth_deg, dtype=np.float64)
    if predicted.shape != ground_truth.shape:
        raise ValueError("predicted and ground-truth yaw arrays must have identical shapes")
    return np.abs((predicted - ground_truth + 180.0) % 360.0 - 180.0)


def localization_recall_metrics(predicted_pose_xzyaw: np.ndarray, ground_truth_pose_xzyaw: np.ndarray) -> dict[str, float]:
    """Measure independent XY/XZ recall and a strict position-plus-yaw recall.

    Pose rows are ``[world_x_m, world_z_m, logical_yaw_deg]``.  The first
    three recalls deliberately mirror the common position-only benchmark
    convention; ``recall_at_1m_30deg`` is the orientation-sensitive companion
    metric required for this HSSD experiment.
    """

    predicted = np.asarray(predicted_pose_xzyaw, dtype=np.float64)
    ground_truth = np.asarray(ground_truth_pose_xzyaw, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1] != 3 or ground_truth.shape != predicted.shape or not len(predicted):
        raise ValueError("predicted and ground-truth poses must both have non-empty shape (N, 3)")
    if not np.isfinite(predicted).all() or not np.isfinite(ground_truth).all():
        raise ValueError("pose metrics require finite coordinates and yaws")
    position_error_m = np.linalg.norm(predicted[:, :2] - ground_truth[:, :2], axis=1)
    yaw_error_deg = angular_error_degrees(predicted[:, 2], ground_truth[:, 2])
    return {
        "recall_at_0.1m": float(np.mean(position_error_m <= 0.1)),
        "recall_at_0.5m": float(np.mean(position_error_m <= 0.5)),
        "recall_at_1m": float(np.mean(position_error_m <= 1.0)),
        "recall_at_1m_30deg": float(np.mean((position_error_m <= 1.0) & (yaw_error_deg <= 30.0))),
        "median_position_error_m": float(np.median(position_error_m)),
        "median_yaw_error_deg": float(np.median(yaw_error_deg)),
    }


def should_log_qualitative_panels(epoch: int, *, every_n_epochs: int = QUALITATIVE_LOG_EVERY_N_EPOCHS) -> bool:
    """Log a fixed validation panel at epoch zero and every configured interval."""

    if epoch < 0 or every_n_epochs <= 0:
        raise ValueError("epoch must be non-negative and logging interval positive")
    return epoch % every_n_epochs == 0


def _build_runtime_classes():
    """Import CUDA-only dependencies only when the actual training CLI starts."""

    import cv2
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from modules.depth_net_pl import UnLocDepthModule

    class HSSDFrameDataset(Dataset):
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self.rows = rows
            self._targets: dict[str, np.ndarray] = {}

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, object]:
            row = self.rows[index]
            image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (480, 640):
                raise FileNotFoundError(f"invalid HSSD RGB frame: {row['image_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image = (image - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            target_path = str(row["depth40_path"])
            if target_path not in self._targets:
                self._targets[target_path] = np.load(target_path, mmap_mode="r")
            depth = np.asarray(self._targets[target_path][int(row["target_index"])], dtype=np.float32).copy()
            if depth.shape != (40,) or not np.isfinite(depth).all() or np.any(depth <= 0):
                raise ValueError(f"invalid depth40 target for {row['scene_id']}/{row['sample_index']}")
            return {
                "ref_img": np.transpose(image, (2, 0, 1)).astype(np.float32),
                "ref_mask": np.ones((480, 640), dtype=np.uint8),
                "ref_depth": depth,
            }

    class HSSDUnLocModule(UnLocDepthModule):
        """Original UnLoc objective plus explicit HSSD calibration telemetry."""

        def train(self, mode=True):
            super().train(mode)
            self.encoder.depth_feature.da.eval()
            return self

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/ray_nll"}}

        def _step(self, batch, stage: str):
            location, scale, _, _ = self.encoder(batch["ref_img"], batch["ref_mask"])
            target = batch["ref_depth"]
            nll = self.laplace_nll_loss(target, location, scale)
            loss = nll
            if self.shape_loss_weight is not None:
                loss = loss + self.shape_loss_weight * (1 - F.cosine_similarity(location, target, dim=-1).mean())
            error = torch.abs(location.detach().float() - target)
            ratio = location.detach().float() / torch.clamp(target, min=1e-3)
            uncertainty = scale.detach().float()
            correlation = F.cosine_similarity(location.detach().float(), target, dim=-1).mean()
            values = {
                f"{stage}/ray_nll": nll,
                f"{stage}/loss": loss,
                f"{stage}/ray_mae_m": error.mean(),
                f"{stage}/ray_median_ae_m": error.median(),
                f"{stage}/depth_ratio_p10": torch.quantile(ratio, 0.10),
                f"{stage}/depth_ratio_p50": torch.quantile(ratio, 0.50),
                f"{stage}/depth_ratio_p90": torch.quantile(ratio, 0.90),
                f"{stage}/uncertainty_p50_m": torch.quantile(uncertainty, 0.50),
                f"{stage}/uncertainty_p90_m": torch.quantile(uncertainty, 0.90),
                f"{stage}/coverage_1scale": (error <= uncertainty).float().mean(),
                f"{stage}/coverage_2scale": (error <= 2 * uncertainty).float().mean(),
                f"{stage}/ray_correlation": correlation,
            }
            self.log_dict(values, on_step=stage == "train", on_epoch=True, prog_bar=stage == "val", batch_size=target.shape[0])
            return loss

        def training_step(self, batch, batch_idx):
            return self._step(batch, "train")

        def validation_step(self, batch, batch_idx):
            self._step(batch, "val")

        def on_after_backward(self):
            squared = [parameter.grad.detach().norm(2).square() for parameter in self.parameters() if parameter.grad is not None]
            if squared:
                self.log("train/gradient_norm", torch.sqrt(torch.stack(squared).sum()), on_step=True, on_epoch=False)

    return HSSDFrameDataset, HSSDUnLocModule


def _fixed_panel_callback(val_dataset, *, every_n_epochs: int):
    import matplotlib.pyplot as plt
    import torch
    from lightning.pytorch.callbacks import Callback

    class FixedPanelCallback(Callback):
        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            if trainer.sanity_checking or not should_log_qualitative_panels(trainer.current_epoch, every_n_epochs=every_n_epochs):
                return
            indices = np.linspace(0, len(val_dataset) - 1, min(4, len(val_dataset)), dtype=int)
            images = []
            pl_module.eval()
            with torch.inference_mode():
                for index in indices:
                    sample = val_dataset[int(index)]
                    rgb = torch.from_numpy(sample["ref_img"])[None].to(pl_module.device)
                    mask = torch.from_numpy(sample["ref_mask"])[None].to(pl_module.device)
                    prediction, scale, _, _ = pl_module.encoder(rgb, mask)
                    figure, axis = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
                    columns = np.arange(40)
                    axis.plot(columns, sample["ref_depth"], label="structural GT", color="#0b5fff", linewidth=2)
                    axis.plot(columns, prediction.squeeze(0).cpu(), label="predicted", color="#b45309", linewidth=2)
                    axis.fill_between(columns, (prediction - scale).squeeze(0).cpu(), (prediction + scale).squeeze(0).cpu(), color="#f59e0b", alpha=.22, label="Laplace scale")
                    axis.set(xlabel="UnLoc RGB feature column (left → right)", ylabel="camera-forward depth (m)")
                    axis.legend(fontsize=8)
                    figure.canvas.draw()
                    images.append(np.asarray(figure.canvas.buffer_rgba()).copy())
                    plt.close(figure)
            logger = trainer.logger
            if hasattr(logger, "log_image"):
                logger.log_image(key="qualitative/depth_profiles", images=images, step=trainer.global_step)
    return FixedPanelCallback()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fresh UnLoc head on exported HSSD RGB frames.")
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--unloc-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--floorplan-root", type=Path)
    parser.add_argument("--maps-root", type=Path)
    parser.add_argument("--localization-per-scene", type=int, default=16)
    parser.add_argument("--qualitative-every-n-epochs", type=int, default=QUALITATIVE_LOG_EVERY_N_EPOCHS)
    parser.add_argument("--skip-final-localization", action="store_true", help="Only for short runtime checks.")
    parser.add_argument("--max-train-scenes", type=int, default=None)
    parser.add_argument("--max-val-scenes", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--shape-loss-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-unloc-checkpoint", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert_fresh_unloc_head(None, init_unloc_checkpoint=args.init_unloc_checkpoint)
    if args.max_epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs, batch size, and worker count must be valid")

    # The upstream model resolves Depth Anything weights relative to this cwd.
    args.adapter_root, args.run_root, args.unloc_root = args.adapter_root.resolve(), args.run_root.resolve(), args.unloc_root.resolve()
    tee_training_output(args.run_root)
    hssd_root = args.adapter_root.parent.parent
    args.floorplan_root = (args.floorplan_root or hssd_root / "UnLoc" / "structural_targets").resolve()
    args.maps_root = (args.maps_root or hssd_root / "2DFP").resolve()
    if not args.unloc_root.is_dir():
        raise FileNotFoundError(args.unloc_root)
    os.chdir(args.unloc_root)
    sys.path.insert(0, str(args.unloc_root))
    import lightning as Lightning
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
    from torch.utils.data import DataLoader

    torch.set_float32_matmul_precision("medium")
    torch.set_num_threads(args.cpu_threads)
    import cv2
    cv2.setNumThreads(1)

    train_rows, val_rows = load_adapter_frame_rows(args.adapter_root, "train"), load_adapter_frame_rows(args.adapter_root, "val")
    def limit_scenes(rows: list[dict[str, object]], count: int | None) -> list[dict[str, object]]:
        if count is None:
            return rows
        chosen = sorted({str(row["scene_id"]) for row in rows})[:count]
        return [row for row in rows if str(row["scene_id"]) in chosen]
    train_rows, val_rows = limit_scenes(train_rows, args.max_train_scenes), limit_scenes(val_rows, args.max_val_scenes)
    HSSDFrameDataset, HSSDUnLocModule = _build_runtime_classes()
    train_dataset, val_dataset = HSSDFrameDataset(train_rows), HSSDFrameDataset(val_rows)
    Lightning.seed_everything(args.seed, workers=True)
    run_root = args.run_root.resolve(); run_root.mkdir(parents=True, exist_ok=True)
    contract = {
        "adapter_root": str(Path(args.adapter_root).resolve()), "unloc_root": str(args.unloc_root),
        "fresh_unloc_head": True, "frozen_depth_anything_backbone": True,
        "train_frames": len(train_dataset), "val_frames": len(val_dataset),
        "max_epochs": args.max_epochs, "qualitative_every_n_epochs": QUALITATIVE_LOG_EVERY_N_EPOCHS,
        "train_scenes": len({r["scene_id"] for r in train_rows}),
        "val_scenes": len({r["scene_id"] for r in val_rows}),
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "cpu_threads": args.cpu_threads, "prefetch_factor": args.prefetch_factor,
        "learning_rate": args.lr, "loss": "laplace_nll" if args.shape_loss_weight is None else "laplace_nll_plus_shape",
        "shape_loss_weight": args.shape_loss_weight, "seed": args.seed,
        "localization_per_scene": args.localization_per_scene,
        "localization_ray_order": "rgb_left_to_right_no_flip", "precision": "16-mixed",
        "checkpoint_monitor": "val/ray_nll", "early_stopping_patience": 12,
        "lr_plateau_patience": 4, "lr_plateau_factor": 0.5,
        "floorplan_root": str(args.floorplan_root), "maps_root": str(args.maps_root),
        "final_full_validation_localization": not args.skip_final_localization,
    }
    contract["qualitative_every_n_epochs"] = args.qualitative_every_n_epochs
    (run_root / "training_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    model = HSSDUnLocModule(shape_loss_weight=args.shape_loss_weight, lr=args.lr)
    if any(parameter.requires_grad for parameter in model.encoder.depth_feature.da.parameters()):
        raise RuntimeError("Depth Anything backbone must remain frozen")
    offline = args.wandb_mode == "offline"
    os.environ["WANDB_MODE"] = args.wandb_mode
    logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
                         save_dir=str(run_root), offline=offline, log_model=False)
    logger.log_hyperparams(contract)
    logger.experiment.define_metric("epoch_completed")
    logger.experiment.define_metric("val_localization_subset/*", step_metric="epoch_completed")
    (run_root / "wandb_run.json").write_text(json.dumps({
        "id": logger.experiment.id, "url": logger.experiment.url,
        "project": args.wandb_project, "entity": args.wandb_entity,
    }, indent=2) + "\n")
    from .monitor import build_localization_callback
    monitor = build_localization_callback(
        val_dataset, floorplan_root=args.floorplan_root, maps_root=args.maps_root,
        run_root=run_root, every_n_epochs=args.qualitative_every_n_epochs,
        per_scene=args.localization_per_scene, seed=args.seed, batch_size=args.batch_size,
        final_full_validation=not args.skip_final_localization,
    )
    callbacks = [
        monitor,
        ModelCheckpoint(dirpath=run_root / "checkpoints", monitor="val/ray_nll", mode="min", save_top_k=3,
                        save_last=True, filename="epoch{epoch:03d}-step{step}", auto_insert_metric_name=False),
        ModelCheckpoint(dirpath=run_root / "checkpoints", monitor="val_localization_subset/recall_at_1m_30deg",
                        mode="max", save_top_k=1, every_n_epochs=args.qualitative_every_n_epochs,
                        filename="localization-epoch{epoch:03d}-step{step}", auto_insert_metric_name=False),
        EarlyStopping(monitor="val/ray_nll", mode="min", patience=12, min_delta=0.001),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = Lightning.Trainer(max_epochs=args.max_epochs, accelerator="gpu", devices=1, precision="16-mixed",
                                logger=logger, callbacks=callbacks, default_root_dir=str(run_root), log_every_n_steps=25,
                                enable_progress_bar=False,
                                limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches)
    trainer.fit(model,
        train_dataloaders=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                     pin_memory=True, persistent_workers=args.num_workers > 0,
                                     worker_init_fn=initialize_loader_worker,
                                     **({"prefetch_factor": args.prefetch_factor} if args.num_workers else {})),
        val_dataloaders=DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True, persistent_workers=args.num_workers > 0,
                                   worker_init_fn=initialize_loader_worker,
                                   **({"prefetch_factor": args.prefetch_factor} if args.num_workers else {})),
    )


if __name__ == "__main__":
    main()
