# Fixed grid-phase sensitivity experiment

This follows the completed dense stride x fusion experiment. It fixes two configurations chosen during development, not on independent data:

| Configuration | Original | Horizontal | Vertical | Both |
|---|---|---|---|---|
| 256 / stride 192 / union | (0,0) | (96,0) | (0,96) | (96,96) |
| 256 / stride 64 / majority | (0,0) | (32,0) | (0,32) | (32,32) |

Offsets change window starting positions only; source rasters, georeferencing, prompt protocol, stride, tile size, ROI context and evaluation mask remain fixed. Four deterministic phases are not independent statistical repeats. This is a post-selection sensitivity check on the development ROI, not evidence of generalization or statistical significance.

## Linux commands

From `~/xy/sam-experiments` with `sam-exp` active:

```bash
python -m experiments.sweep_offsets --fusion-dir outputs/dense_fusions --output-dir outputs/grid_offsets --plan-only
```

Inspect the estimated scratch storage, pending window counts, and free disk. Two zero-offset results are reused from the completed fusion sweep; six shifted configurations need actual SAM inference. The 256/64 configuration is expensive, so expect more work than cached refusion. New scratch-map estimate excludes saved tiles and result rasters.

```bash
python -m experiments.sweep_offsets --fusion-dir outputs/dense_fusions --output-dir outputs/grid_offsets
```

No GT is passed to inference processes. GT pixel evaluation starts after all six new configurations complete. Input and existing result hashes are validated. Existing complete configurations can be reused by rerunning the command; incomplete individual runs are rejected rather than overwritten. Preserve/relocate the explicit partial run, or use a new sweep output directory. No recursive deletion is performed.

## Outputs and interpretation

- `comparison.csv`: eight rows with phase, offsets, windows, inference wall time and all metrics.
- `summary.txt`: per-phase metrics and mean/min/max/range summaries.
- `offset_sweep.json`: settings, provenance, status, artifact hashes, metrics and across-phase ranges.
- `index.html` and four SVGs: four-phase metric comparisons for the two configurations, with BsiNet reference; axes auto-scaled, metric units 0–1.
- `cases.json` and PNG panels: four fixed-seed (41) samples from original-image-origin 256-pixel cells intersecting ROI, chosen without GT or error ranking. Samples may include outside-ROI portions, which are darkened. Identical locations show RGB, GT, BsiNet, original-phase union and original-phase majority, plus false-positive/false-negative maps (blue FP, red FN). They illustrate original results, not all shifted variants. Empty/small ROIs can yield fewer samples.

Compare both changes from original phase and the across-phase range. If the reported original gain is smaller than phase variation, the original single-grid gain needs qualification. Do not select the best phase using this ROI and describe it as independent-test improvement. A mean over these four deterministic phases is descriptive, not an uncertainty estimate or a new ensemble method. Inference wall times include initialization/I/O; original reused timings are historical.

## Window boundary convention

`run_tiles` now accepts `--offset-x` and `--offset-y` in [0,stride). The historical zero-origin grid is unchanged. Nonzero offsets add one leading window starting at offset-stride on that axis, clip it to image extent, then continue with the shifted lattice. Border windows can be smaller; input pixels are never shifted/resampled. This keeps complete image coverage while changing interior grid phase. Border coverage multiplicities can differ, so results include phase and edge effects together. The pre-existing ROI context stays fixed; no claim that all differences are caused solely by semantic context is made.

The original dense stride scanner now explicitly rejects nonzero-origin runs for reuse, so new offset experiments cannot accidentally mix into the historical zero-grid comparison.
