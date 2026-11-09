> 역사적 연구 기록. 원본: `candidate_sampling_20261107_2337/historical_mining_README.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# HSSD frozen-epoch20 training hard negatives

This is training-data preparation, not held-out performance evaluation. Only
the frozen 116 train scenes / 61,502 RGB frames are read for mining. The 19 val
and 17 test scenes remain untouched. The checkpoint SHA is pinned in protocol.json.
The training run remains stopped; mining performs inference only.

## Selection

- Full per-scene visual localization uses the existing HSSD no-flip 11-ray,
  36-world-yaw, 0.1m DeSDF comparison. Yaw zero points +X, positive yaw goes +Z;
  raster +X is right and +Z is down.
- Save depth40 and scale40, max-over-yaw XY log likelihood, argmax yaw per cell,
  candidate world coordinates and nearest-GT-cell 36-yaw scores. Full joint
  likelihood can be reconstructed from saved predictions and unchanged DeSDF.
- Select up to four strongest spatial modes strictly >1m from GT, with 1m NMS.
  Natural ambiguity is not rejected: it is precisely what these negatives target.
- Add up to two orientation-only negatives at the exact captured XYZ, with yaw
  error >=60 degrees and 60-degree angular NMS. Their ranking is evaluated at
  the nearest DeSDF cell, but RIR rendering never snaps the captured XYZ.
- Add one seeded, same-scene random spatial negative >2m from GT and >=1m from
  chosen hard modes. Small scenes may have fewer candidates; shortages are
  recorded rather than silently labeling nearby positives as negatives.
- Spatial candidate centres and all six microphones (5cm radius) are checked
  against the original 1cm map. Camera/source height is the original RGB Y.
- GT is used for exclusion/negative labels only, not in visual likelihood.
  These are in-sample training-model negatives, not out-of-fold predictions.

## Storage and reuse

Local inference outputs:
`${HSSD_SSD_ROOT}/UnLoc/train_likelihood/epoch020/`

Shared pair links, immutable schedules, protocol, readiness and render statuses:
`${HSSD_ROOT}/RIR/manifests/hard_negatives_epoch020_v1/`

New raw six-channel FP arrays:
`HSSD/RIR/FP/train/<scene>/negative_records/nNNNNN/base_acoustic_yaw_BBB.npy`

Observed RIRs are reused from positive queries. Exact scene/XYZ/base matches
reuse existing positive FP records too. New FP records are de-duplicated across
all queries in each scene. Pair metadata specifies channel rolls for both
observed and FP arrays. World yaw -> acoustic angle is a=(-yaw)%360, base=a%60,
roll=-(a-base)/60. Ten-degree candidate headings are never rounded to 45 degrees.

Raw RIRs are currently generated on File2, not duplicated to SSD during the job.
Baseline-specific visual likelihood lives under UnLoc; shared acoustic data
lives under RIR. Common geometry/semantic maps remain at the HSSD root.

## Queue and monitoring

Node2 waits for BOTH positive jobs (111-scene main + 5-scene height supplement)
AND all 116 mined-scene schedules. It checks completion, failures, split hashes
and schedule checksums before starting FP-only negatives. The same exclusive
NFS lock prevents positive/negative CPU overlap.

Node2 uses at most 76 logical CPUs, 38 workers x 2 acoustic threads, nice 15,
104GiB RAM. GPU visibility is disabled, Docker uses runc with no GPU device
requests, Habitat GPU device=-1 and software EGL. Other users' jobs are untouched.

The local watchdog is a user systemd service polling every 600 seconds; the
node queue is a user systemd service polling every 60 seconds. Both have bounded
restart-on-failure policies and user lingering enabled, so SSH logout does not
terminate supervision. This replaces the original detached, record-only monitor.
Latest combined status is `UnLoc/train_likelihood/epoch020/pipeline_status.json`
on SSD and `RIR/manifests/hard_negatives_epoch020_v1/pipeline_status.json` on
File2. Events are recorded on stage/progress-bucket changes. Automatic chat
notifications are not available in this session; monitoring writes files and
systemd logs rather than claiming to wake Codex.

## Recovery policy (2026-10-18)

- Completed visual scenes require matching depth, likelihood, schedule and pair
  hashes before reuse. Per-scene DataLoaders have `persistent_workers=False`;
  NPZ archives are explicitly closed and open file descriptors are logged.
- Each native RIR task now runs in a fresh subprocess, assigned two CPU cores.
  Across 38 concurrent tasks, the limit remains 76 CPUs and no GPU devices.
  This avoids inheriting/reusing native simulator thread-pool state across scenes.
- A worker with no log/output progress for 600 seconds or runtime >1800 seconds
  is terminated as an isolated process group and retried, at most three attempts.
  Already valid arrays are SHA-checked and skipped, never regenerated.
- If every expected array and metadata record is valid but native teardown has
  stalled, only that child is stopped; the task is recovered from verified data.
- The node watchdog may restart ONLY the three explicitly named owned containers,
  with matching image/script identity, CPU cap and GPU-disabled runtime. A render
  stage without progress for 30 minutes, or a safely interrupted container, has
  at most three automatic container restarts and a ten-minute cooldown.
- The local watchdog restarts ONLY the pinned epoch20 miner for recognized
  loader/descriptor/connection failures or non-OOM interruptions, at most three
  times. It also ensures that the node queue service is running.
- Checksum/geometry/split mismatches, disk I/O errors, full disks, OOM and unknown
  failed records are not blindly retried or silently replaced. They require
  attention and cannot trigger downstream negative rendering.
- Original positive main scope stays 111 scenes; the five height-supplement
  scenes remain separate. Full positive counts are checked before negatives.
- Services: local `hssd-pipeline-monitor.service`, node2
  `hssd-acoustic-queue.service`. Controller locks prevent duplicate supervisors.

## Local + node2 generation (2026-10-18)

The full immutable negative inventory contains 23,012 tasks / 326,355 physical
six-channel FP arrays, linking 430,514 training pairs / 61,502 RGB queries.

- `distributed/plan.json` assigns global task IDs deterministically:
  `id % 48 < 10` belongs to Local, all other IDs to node2. Local owns 4,800 tasks /
  68,005 arrays with 10 workers / 20 CPUs; node2 owns 18,212 tasks / 258,350 arrays
  with 38 workers / 76 CPUs. Both remain GPU-free and nice 15.
- Tasks have disjoint output paths. NFS per-task exclusive locks and per-shard
  supervisor locks add duplicate-writer protection. Cross-host NFS lock exclusion
  was tested successfully. The shared stage lock uses a readable/writable FD,
  as required by NFS shared-lock emulation; positives still require an exclusive
  stage lock. The old unsharded writer was stopped before either shard began.
- Previously generated outputs are verified and reused irrespective of which
  host originally produced them. Raw storage paths do not change.
- `runs/negative_shards/local/status.json` and `.../node2/status.json` report each
  shard. `runs/node2_negative/status.json` is the combined pipeline status.
- The node2 coordinator waits for BOTH shards, then verifies every expected
  negative and positive array, full SHA256, finite nonzero six-channel samples,
  geometry/pose/base/source/acoustic contracts, output inventory, all pair links,
  train-only membership, and channel rolls. No duplicate query/negative links
  or unexpected committed negative files may be silently accepted.
- Final completion requires `distributed/verification.json`: expected 326,355
  negative arrays + 62,386 positive arrays = 388,741 unique physical arrays,
  and 430,514 pair links. Failure cannot be labeled complete.
- The Local watchdog now also recovers its explicitly owned CPU RIR shard,
  subject to the same bounded retry/integrity policy.

### Renderer equivalence

Node2 remains the baseline; its physics/settings were not changed. Image IDs
are different (`71d611...` node2 versus `c2907a...` Local), but all 32 ordered
filesystem layers, Habitat 0.2.2, `libRLRAudioPropagation.so`, and Habitat native
bindings match exactly by SHA256. The inspected default User field is empty
string versus absent/null, equivalent to the same default user; actual jobs
explicitly run as dataset UID/GID 1000:1000. CPU/software-EGL and acoustic
launch settings match. A full image retransmission was therefore unnecessary.
See `distributed/renderer_equivalence.json` and `distributed/local_launch.json`.
Different CPUs/Monte Carlo traces are not required to yield bit-identical newly
rendered waveforms; each committed waveform has its own integrity hash.

The current HSSD stage prepares contrastive supervision only. It does not train
an encoder or implement an HSSD InfoNCE training loop. The repository has a
separate legacy Replica InfoNCE experiment, not connected to these new manifests.
