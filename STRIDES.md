# Dense stride experiment

This extends the existing center-fusion scale experiment to seven strides per size, holding the prompt protocol, model, image, BsiNet prediction, coverage, development ROI and context fixed.

| Tile size | Strides |
|---|---|
| 256 | 256, 224, 192, 160, 128, 96, 64 |
| 512 | 512, 448, 384, 320, 256, 192, 128 |

The overlap per axis is 0%, 12.5%, 25%, 37.5%, 50%, 62.5%, 75%. This is a prespecified parameter grid, not a claim these are optimal. Keep fusion fixed at center to isolate stride effects. Union/intersection and local-context changes are separate experiments. The fixed-origin window convention and final truncated border windows remain unchanged.

## Linux execution

From `~/xy/sam-experiments`, after pulling the commit, with `sam-exp` active:

```bash
python -m experiments.sweep_strides --template-run outputs/t256_s192 --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --output-dir outputs/dense_strides --plan-only
```

This inherits actual paths and settings from the existing completed center-fusion run. It scans its sibling directories for compatible completed original window runs, including differently named old runs. It does not infer or evaluate. It prints the planned windows, reused/new counts, minimum scratch-map disk requirement, and free space. Cached tile TIFFs and output rasters require additional space beyond the scratch estimate; high overlap creates substantially more windows. Check available space before starting.

Start the experiment with the same command without `--plan-only`:

```bash
python -m experiments.sweep_strides --template-run outputs/t256_s192 --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --output-dir outputs/dense_strides
```

If all six earlier 256/512 configurations are compatible and present, eight configurations need new inference. New results go in `outputs/dense_strides/runs/tSIZE_sSTRIDE`. Each new configuration runs in a separate Python process to release model memory between configurations. No GT is passed to inference; GT pixel evaluation begins only after all configurations finish. GT file hashes can be read earlier for provenance.

## Reuse, resume and limitations

Reuse requires matching paths, the original fixed prompt settings, complete status, no smoke limit, center fusion, device/model/RGB/context/tolerance/CRS policy, and existing input hashes. It validates window logs and tile existence. Source legacy runs are read-only: existing summary files are not overwritten. Legacy manifests did not seal historical image and coverage bytes, so matching current paths cannot prove those files were unchanged historically; the sweep records this limitation and current hashes. New window manifests now also hash image and coverage. Old and new code revisions remain available in each run manifest, rather than being asserted identical.

Re-run the same command to resume at **configuration boundaries**. Complete configurations are reused. Changed input hashes or cached summary artifacts are rejected. A partial window run is not resumed or deleted automatically: the error identifies it. Preserve/relocate that explicit partial directory before retrying, or use a new sweep output directory. Do not remove the sweep state or alter source files to bypass a rejection. Output directories that do not contain a matching sweep state are rejected.

The tool re-evaluates all configurations on the fixed ROI and verifies identical baseline metrics, full ROI inclusion, coverage, output grids and input hashes. It writes:

- `sweep.json`: settings, status, jobs, provenance, metrics and artifact hashes.
- `comparison.csv`: all 14 configurations, IoU, boundary F1, instance F1, PQ and other metrics, processed windows, original run wall time, sum of recorded window times.
- `summary.txt`: short baseline and per-configuration results.
- `index.html` and six standalone SVG plots: overlap vs IoU, boundary F1, instance F1, PQ, windows and wall time. Accuracy plots include the BsiNet reference.

Open `index.html` in a browser and keep the SVGs alongside it. Axes are automatically scaled and metric values use 0–1 units. Timing includes initialization, data I/O and hash work and is not pure GPU time; legacy timings come from historical runs and are not controlled repeated benchmarks. Keep timing conclusions descriptive. There is no automatic best-setting recommendation, significance claim or final test selection. This remains a repeatedly inspected development ROI with unverified training independence and fixed BsiNet IDs.
