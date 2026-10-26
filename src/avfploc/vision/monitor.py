#!/usr/bin/env python3
"""Fixed validation localization and RGB/likelihood/depth panels for training."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def fixed_validation_indices(rows: list[dict], *, per_scene: int = 16, seed: int = 42) -> list[int]:
    """Freeze a scene/yaw-balanced subset without using model performance."""
    if per_scene < 1:
        raise ValueError("per_scene must be positive")
    groups: dict[str, dict[float, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        groups[str(row["scene_id"])][float(row["logical_yaw_deg"]) % 360].append(index)
    selected: list[int] = []
    for scene, headings in sorted(groups.items()):
        scene_seed = int.from_bytes(hashlib.sha256(scene.encode()).digest()[:4], "little") ^ seed
        rng = np.random.default_rng(scene_seed)
        buckets = [rng.permutation(indices).tolist() for _, indices in sorted(headings.items())]
        picked = []
        for offset in range(max(map(len, buckets))):
            for bucket in buckets:
                if offset < len(bucket):
                    picked.append(bucket[offset])
                    if len(picked) == per_scene:
                        break
            if len(picked) == per_scene:
                break
        selected.extend(sorted(picked))
    return selected


def hssd_candidate_windows(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the evaluator's no-flip, 11-ray yaw windows at free candidates."""
    from avfploc.geometry.camera import hssd_ray_bins

    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 36 or not np.isfinite(values).all():
        raise ValueError("expected finite [row, col, 36] DeSDF")
    free = np.any(values > 0, axis=2)
    if not free.any():
        raise ValueError("DeSDF has no free candidates")
    indices = np.stack([hssd_ray_bins(yaw) for yaw in range(36)])
    return values[free][:, indices], free, np.argwhere(free)


def build_localization_callback(
    val_dataset, *, floorplan_root: Path, maps_root: Path, run_root: Path,
    every_n_epochs: int = 5, per_scene: int = 16, seed: int = 42, batch_size: int = 8,
    final_full_validation: bool = True,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from lightning.pytorch.callbacks import Callback
    from avfploc.visualization.likelihood import _likelihood_map
    from avfploc.vision.train import localization_recall_metrics
    from utils.localization_utils import get_ray_from_depth_uncertainty
    from avfploc.geometry.camera import unloc_feature_column_bearings_deg

    fixed = fixed_validation_indices(val_dataset.rows, per_scene=per_scene, seed=seed)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in fixed:
        grouped[str(val_dataset.rows[index]["scene_id"])].append(index)
    for scene in grouped:
        if not (floorplan_root / scene / "desdf.npy").is_file():
            raise FileNotFoundError(f"validation DeSDF missing: {scene}")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "validation_localization_subset.json").write_text(json.dumps({
        "split": "val", "seed": seed, "per_scene": per_scene,
        "frame_count": len(fixed), "scene_count": len(grouped),
        "samples": [{"scene_id": val_dataset.rows[i]["scene_id"],
                     "sample_index": val_dataset.rows[i]["sample_index"]} for i in fixed],
    }, indent=2) + "\n")

    class LocalizationCallback(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            self._epoch_started = self._window_started = time.monotonic()
            self._window_samples = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self._window_samples += int(batch["ref_depth"].shape[0])
            if (batch_idx + 1) % 50 == 0:
                now = time.monotonic()
                rate = self._window_samples / (now - self._window_started)
                metrics = {"performance/train_images_per_second": rate,
                           "performance/gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30}
                trainer.logger.log_metrics(metrics, step=trainer.global_step)
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
                print(json.dumps({"event": "training_progress", "epoch": trainer.current_epoch + 1,
                      "batch": batch_idx + 1, "batches": trainer.num_training_batches,
                      "global_step": trainer.global_step, "images_per_second": round(rate, 2),
                      "loss": None if loss is None else float(loss.detach())}), flush=True)
                self._window_samples, self._window_started = 0, now

        def on_train_epoch_end(self, trainer, pl_module):
            trainer.logger.log_metrics({"performance/epoch_wall_seconds": time.monotonic() - self._epoch_started},
                                       step=trainer.global_step)

        def _evaluate(self, trainer, module, epoch: int, *, indices_by_scene=None, prefix="val_localization_subset", panels=True):
            started = time.monotonic()
            module.eval()
            selected = grouped if indices_by_scene is None else indices_by_scene
            folder = run_root / "validation_localization" / (f"epoch_{epoch:03d}" if panels else prefix)
            folder.mkdir(parents=True, exist_ok=True)
            predicted, ground_truth, records, images, captions = [], [], [], [], []
            scene_results = {}
            with torch.inference_mode():
                for scene, indices in selected.items():
                    predictions, scales, targets = [], [], []
                    for start in range(0, len(indices), batch_size):
                        samples = [val_dataset[index] for index in indices[start:start + batch_size]]
                        rgb = torch.from_numpy(np.stack([s["ref_img"] for s in samples])).to(module.device)
                        masks = torch.from_numpy(np.stack([s["ref_mask"] for s in samples])).to(module.device)
                        with trainer.precision_plugin.forward_context():
                            mu, scale = module.encoder(rgb, masks)[:2]
                        predictions.extend(mu.float().cpu().numpy())
                        scales.extend(scale.float().cpu().numpy())
                        targets.extend(s["ref_depth"] for s in samples)
                    payload = np.load(floorplan_root / scene / "desdf.npy", allow_pickle=True).item()
                    windows, free, cells = hssd_candidate_windows(payload["desdf"])
                    tensor = torch.as_tensor(windows, device=module.device)
                    spacing = float(payload["desdf_spacing_m"])
                    origin_x, origin_z = float(payload["origin_x_m"]), float(payload["origin_z_min_m"])
                    scene_pred, scene_gt = [], []
                    scene_records = []
                    for offset, index in enumerate(indices):
                        row = val_dataset.rows[index]
                        mu, scale = np.asarray(predictions[offset]), np.asarray(scales[offset])
                        if not np.isfinite(mu).all() or not np.isfinite(scale).all() or (scale <= 0).any():
                            raise ValueError(f"invalid depth/scale in validation: {scene}/{row['sample_index']}")
                        rays, ray_scale = get_ray_from_depth_uncertainty(mu, scale, V=11, F_W=3 / 8)
                        ray_tensor = torch.as_tensor(rays, dtype=torch.float32, device=module.device)
                        scale_tensor = torch.as_tensor(ray_scale, dtype=torch.float32, device=module.device)
                        scores = -(torch.abs(tensor - ray_tensor) / scale_tensor).sum(dim=-1) / 40.0
                        spatial_score, yaw = scores.max(dim=-1)
                        winner = int(spatial_score.argmax().item())
                        cell = cells[winner]
                        pose = [origin_x + int(cell[1]) * spacing, origin_z + int(cell[0]) * spacing,
                                int(yaw[winner].item()) * 10.0]
                        actual = [float(v) for v in row["pose_xzyaw"]]
                        scene_pred.append(pose)
                        scene_gt.append(actual)
                        record = {"scene_id": scene, "sample_index": int(row["sample_index"]),
                                  "gt_pose_xzyaw": actual, "map_pose_xzyaw": pose,
                                  "position_error_m": float(np.linalg.norm(np.asarray(pose[:2]) - actual[:2])),
                                  "yaw_error_deg": float(abs((pose[2] - actual[2] + 180) % 360 - 180))}
                        scene_records.append(record)
                        if panels and offset == 0:
                            likelihood = np.zeros(free.shape, dtype=np.float32)
                            likelihood[free] = torch.exp(spatial_score - spatial_score.max()).cpu().numpy()
                            metadata = json.loads((maps_root / scene / "floorplan_metadata.json").read_text())
                            metadata["desdf_spacing_m"] = spacing
                            base = f"{scene}_Q{int(row['sample_index']):05d}"
                            map_path, panel_path = folder / f"{base}_likelihood.png", folder / f"{base}_panel.png"
                            _likelihood_map(maps_root / scene / "map.png", metadata, likelihood, free, actual, pose, map_path)
                            fig = plt.figure(figsize=(11, 8), dpi=110, constrained_layout=True)
                            grid = fig.add_gridspec(2, 2)
                            rgb_axis, map_axis = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
                            rgb_axis.imshow(Image.open(str(row["image_path"])).convert("RGB"))
                            rgb_axis.set_title(f"{scene} / Q{int(row['sample_index']):05d} · epoch {epoch}", fontsize=9)
                            rgb_axis.axis("off")
                            map_axis.imshow(Image.open(map_path))
                            map_axis.set_title(f"GT blue / MAP green · {record['position_error_m']:.2f} m, {record['yaw_error_deg']:.0f}°", fontsize=9)
                            map_axis.axis("off")
                            axis = fig.add_subplot(grid[1, :])
                            bearing = unloc_feature_column_bearings_deg()
                            axis.plot(bearing, targets[offset], color="#0b5fff", label="structural GT")
                            axis.plot(bearing, mu, color="#b45309", label="predicted")
                            axis.fill_between(bearing, mu - scale, mu + scale, color="#f59e0b", alpha=.22, label="± Laplace scale")
                            axis.set(xlabel="relative bearing (°; RGB left → right)", ylabel="camera-forward depth (m)")
                            axis.grid(alpha=.2)
                            axis.legend(fontsize=8)
                            fig.savefig(panel_path)
                            plt.close(fig)
                            np.savez_compressed(folder / f"{base}.npz", likelihood=likelihood, free=free,
                                                predicted_depth40=mu, predicted_scale40=scale,
                                                gt_depth40=targets[offset], gt_pose_xzyaw=actual, map_pose_xzyaw=pose)
                            images.append(str(panel_path))
                            captions.append(f"{scene} Q{int(row['sample_index']):05d} epoch={epoch}")
                    scene_results[scene] = localization_recall_metrics(np.asarray(scene_pred), np.asarray(scene_gt))
                    predicted.extend(scene_pred)
                    ground_truth.extend(scene_gt)
                    records.extend(scene_records)
                    del tensor
            metrics = localization_recall_metrics(np.asarray(predicted), np.asarray(ground_truth))
            metrics.update(frame_count=len(records), scene_count=len(selected), evaluation_seconds=time.monotonic() - started)
            epoch_metric = {"epoch_completed": epoch} if panels else {f"{prefix}/checkpoint_epoch_completed": epoch}
            trainer.logger.log_metrics({**epoch_metric, **{f"{prefix}/{key}": value for key, value in metrics.items()}}, step=trainer.global_step)
            # Callback metrics also feed the orientation-sensitive checkpoint.
            for key, value in metrics.items():
                trainer.callback_metrics[f"{prefix}/{key}"] = torch.tensor(float(value), device=module.device)
            if panels and hasattr(trainer.logger, "log_image"):
                trainer.logger.log_image(key="qualitative/likelihood_rgb_depth", images=images,
                                         caption=captions, step=trainer.global_step)
            (folder / "metrics.json").write_text(json.dumps({"epoch_completed": epoch, "split": "val",
                "scope": "fixed_subset" if panels else "full_validation", "overall": metrics,
                "scenes": scene_results}, indent=2) + "\n")
            (folder / "poses.json").write_text(json.dumps(records) + "\n")
            print(f"HSSD {prefix} epoch={epoch}: n={len(records)} R@1m={metrics['recall_at_1m']:.4f} "
                  f"R@1m,30deg={metrics['recall_at_1m_30deg']:.4f} ({metrics['evaluation_seconds']:.1f}s)", flush=True)

        def on_fit_start(self, trainer, pl_module):
            self._evaluate(trainer, pl_module, 0)
            pl_module.train()
            pl_module.encoder.depth_feature.da.eval()

        def on_validation_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch + 1
            if not trainer.sanity_checking and epoch % every_n_epochs == 0:
                self._evaluate(trainer, pl_module, epoch)

        def on_fit_end(self, trainer, pl_module):
            if not final_full_validation:
                return
            # Lightning may move the module to CPU during strategy teardown
            # before on_fit_end. The final localization pass must still use GPU.
            pl_module.to(trainer.strategy.root_device)
            full = defaultdict(list)
            for index, row in enumerate(val_dataset.rows):
                full[str(row["scene_id"])].append(index)
            best = next((callback.best_model_path for callback in trainer.checkpoint_callbacks
                         if callback.monitor == "val/ray_nll"), "")
            if best:
                checkpoint = torch.load(best, map_location="cpu", weights_only=False)
                checkpoint_epoch = int(checkpoint["epoch"]) + 1
                pl_module.load_state_dict(checkpoint["state_dict"])
                del checkpoint
                self._evaluate(trainer, pl_module, checkpoint_epoch,
                               indices_by_scene=full, prefix="val_full_best_nll", panels=False)
                trainer.logger.experiment.summary["best_nll_checkpoint"] = best

    return LocalizationCallback()
