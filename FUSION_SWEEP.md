# Dense stride x cached fusion comparison

Next experiment after the 14-configuration dense stride sweep: keep every window prediction fixed, compare center, union, intersection, majority. There is no new SAM inference, no manual rule selection, and no change to stride, context, prompts or weights. Every method retains the original BsiNet instance IDs; this is binary foreground fusion, not probability blending or instance discovery.

From the Linux project root, in `sam-exp`, after pulling the code:

```bash
python -m experiments.sweep_fusions --sweep-dir outputs/dense_strides --output-dir outputs/dense_fusions --plan-only
```

The plan verifies the completed dense sweep, input/GT hashes, saved source raster/log hashes, and cache existence and sizes. It seals current count/votes/center cache snapshots for the next stage. It prints pending configuration count and an uncompressed output raster estimate. The default full experiment has 14 configurations x 4 fusion rules = 56 result rows. Four derived label and count raster pairs are stored per configuration, so disk usage increases even without inference. Actual compression varies; sidecars need extra space.

```bash
python -m experiments.sweep_fusions --sweep-dir outputs/dense_strides --output-dir outputs/dense_fusions
```

Paths, model-independent parameters and GT come from the completed dense sweep. Each configuration uses `refuse_tiles` to validate cache invariants, counts, window-log totals, and reproduction of the original center raster before deriving all four rules. Evaluation uses the same fixed ROI and boundary tolerance. It verifies identical BsiNet metrics across all runs and exact center metric reproduction against the dense sweep. It reports all settings, rather than selecting a best method using GT. Legacy cache hashes are current snapshots, not inference-time seals.

Outputs:

- `comparison.csv`: 56 rows with size, stride, overlap, fusion, windows, original inference wall time, separate fusion time, all metrics, and signed changes from BsiNet and corresponding center result. Metric changes are absolute 0–1 units; multiply by 100 for percentage points.
- `index.html` and eight SVG charts: four fusion curves per tile size for IoU, boundary F1, instance F1, PQ; dashed BsiNet reference. Axes auto-scale and use 0–1 metric units.
- `summary.txt`: baseline plus all 56 results, intentionally without automatic winner selection.
- `fusion_sweep.json`: status, provenance, cache/output hashes and results.
- `tSIZE_sSTRIDE/METHOD/`: derived raster, count, manifest and evaluation summary.

Re-run the exact command to reuse completed configurations. Original dense runs and legacy sources are read-only. Changed sources/caches/recorded derived artifacts are rejected. Partial configurations are identified and never overwritten: preserve/relocate only the explicit partial directory before retrying, or use a new output root. There is no automatic recursive deletion. GT remains evaluation only. A completed plan can be resumed; fusion is CPU/I/O work and does not require the SAM checkpoint to be loaded, but original checkpoint/input hashes are checked.

Interpretation: compare the curve shapes within a rule and the differences between rules at each identical stride. Union favors foreground retention, intersection requires unanimity over valid covering windows, majority requires strictly more than half (ties background), center takes the most interior observation. At no overlap these rules should coincide on observed pixels. This remains development analysis with unverified training independence; timings are descriptive, and no significance or generalization claim is made.
