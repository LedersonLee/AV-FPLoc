# Physical Material Audit

## Installed implementation and assignment

The installed SoundSpaces/RLR image and library hashes are recorded in the [manifest](reproducibility_manifest.json). We inspected the installed `RLRAudioPropagation.h` and Habitat `AudioSensor.cpp`, rather than assuming the latest upstream implementation matches this binary.

`enableMaterials=false` loads a joined nonsemantic mesh with the default acoustic material. It does **not** disable absorption or scattering. The observed default is absorption0.1, scattering0.5, transmission0. The stock semantic-loading branch consumes `setAudioMaterialsJSON`; passing a JSON while using the nonsemantic branch does not establish HSSD surface assignment.

This experiment uses the same installed RLR C API directly. Mesh indices are grouped under `audit_surface` and `fixed_background`. The public header matches category strings against material `labels` substrings; the human-readable `name` is ignored. All1,980 controlled acoustic OBJ exports were checked against the requested coefficients. A removed single-wall scene has no varying-surface faces, so its export correctly contains only the fixed background material.

## Frequency coefficients

Actual `RLRA_WriteIRMetrics` output reports four log-spaced centers:

| Band | Center Hz |
|---|---:|
| 1 | 83.910034 |
| 2 | 369.251007 |
| 3 | 1624.911255 |
| 4 | 7150.516602 |

The material JSON uses alternating frequency/value pairs. The exact interpolation kernel between anchors is not documented in the public header, and engine source is unavailable. The full experiment places coefficients at these reported centers to minimize reliance on an assumed interpolation rule. Acoustic mesh export verifies assignment, but does not expose internal path times or intensity histograms.

Uniform absorption uses α at all four centers. Low-frequency absorption uses `[α+δ, α+δ, α−δ, α−δ]`; high-frequency absorption reverses that sequence, with δ=.8min(α,1−α). Thus the center-wise mean is identical, while the incident-spectrum-weighted absorbed energy need not be. The tested α values are .05/.30/.70/.99 and scattering values are0/.5/1. Transmission is always0.

Floor/ceiling or distant background geometry retains the bundled default material, including its damping, density and material-medium speed. These are not additional interventions. The material speed field near1484m/s applies to transmission through the material; it is not the ambient air reference c=343m/s.

## Absorption, reflection and scattering

The public header describes coefficient values as pressure fractions, whereas our single-wall response is closer to an energy-fraction relationship. At s=0, the observed early-energy ratio for α=.70 versus.05 is.347 at2m and.309 at6m. The energy-fraction prediction is.316; treating the residual coefficient as pressure and squaring it predicts.100.

This is **consistent evidence**, not a direct measurement of total incident/reflected flux or proof of the internal formula. With transmission0, absorbed and reflected energy share the incident budget; reflection is therefore not an independent slider. Scattering redistributes surviving reflection energy. An increase at one receiver does not imply an increase in total reflected energy.

## Geometry and sensor contract

The source is at height1.5m at the center of a0.05m six-microphone ring, with θ=π+iπ/3. The renderer's0.1m listener capture radius is a different parameter and remains fixed. Sample0 is unchanged across conditions.

All runs use48kHz, unitScale1, globalVolume1, default maximum IR length4s,20,000 indirect rays,512 source rays, depth50, two threads, diffraction on, transmission off and temporal coherence off. Channels are sequential mono renders, not verified phase-synchronous microphones. Three fresh contexts per condition are not asserted to be independent RNG seeds.

Single-wall geometry uses the prior large vertical panel and a distant floor. The room is8×8×3m; the corridor is2.4×18×3m. In room/corridor conditions, only vertical surfaces receive the variable material. Front-panel removal leaves other reflectors. Moving the front panel by1m leaves corner gaps; this is a panel intervention, not a sealed-room resize.

No detected peak is assigned a wall ID. Independent scalar Fermat minimization and108 raw energy/onset checks verify target-panel geometry predictions and measurement arithmetic. They do not prove acoustic path identity.

## HSSD feasibility and limits

Two Train native stage GLBs were rendered at four previously recorded poses. Original GLB and FP hashes remain unchanged. A vertical-face proxy, `|normal_y|<.25`, receives the variable material; other stage faces retain the default.

| Scene | Stage faces | Variable vertical-proxy faces |
|---|---:|---:|
| 102344250 | 4677 | 2438 |
| 102816615 | 3263 | 1683 |

Stage and object instances are stored separately, but the proxy is not exact architectural-wall segmentation. Furniture and opening objects are excluded from this stage-only audit; their instance counts are archived. RGB and FP assets were not modified or regenerated.

Limited native surface assignment is **verified**. Full furnished HSSD semantic-material mapping remains **unverified**. It would require validated triangle/category assignment, preservation of opening/furniture transforms, and comparison with the original joined acoustic mesh. The controlled-scene results are not presented as HSSD results.

## Evidence level

- **Verified:** parameter assignment, fixed-geometry material interventions, systematic absorption/profile changes, and target-wall ambiguity when other reflectors remain.
- **Consistent:** fixed floor/ceiling responses help explain weaker absorption sensitivity in rooms; measured energy ratios are closer to energy-budget than pressure-squared scaling.
- **Unresolved:** exact first-order path identity, internal intensity-to-pressure filtering/grouping, physical transfer of peak timing, and scene-general reliability. Experiment1B's renderer-dependent timing finding remains applicable.

Primary references: [RLR distribution](https://github.com/facebookresearch/rlr-audio-propagation) and [SoundSpaces2 implementation](https://github.com/facebookresearch/sound-spaces/blob/main/SoundSpaces2.md). Installed-version source, exports and recorded hashes are the authoritative evidence for this run.
