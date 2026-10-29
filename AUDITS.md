# Development instance and spatial split audits

These commands do not train, infer, change predictions, choose new model settings, or create an independent test split. Run in the Linux project root using `sam-exp`. Existing output directories are rejected.

## Instance failures in the existing development ROI

```bash
python -m experiments.audit_instances --boundary-dir outputs/boundary2 --gt "$HOME/xy/data/qgis_label_result3/instance.tif" --output-dir outputs/instance_audit
```

Send `outputs/instance_audit/summary.txt` for the next analysis. `instances.csv` gives per-method, per-prediction and per-GT records, best IoU/partner, actual matched partner, global pixel bounding box, possible ROI/crop truncation, and merge/split evidence. `report.json` records source hashes, matching rules and totals. It audits all six existing methods, uses the exact existing greedy IoU >=0.5 one-to-one matching, and checks its totals against the saved boundary experiment. It does not tune a matching threshold to improve results.

Unmatched reasons are disjoint: no foreground overlap, best IoU below 0.5, or eligible overlap lost through one-to-one competition. Merge/split flags are separate overlapping heuristics, including matched instances: a prediction overlapping at least 10% of each of multiple GTs suggests merging; multiple predictions each overlapping at least 10% of one GT suggest fragmentation. These do not prove the physical cause and can miss weak overlaps. A split flag does not certify that the model can repair it. Current fixed BsiNet IDs cannot discover or split instances.

The cut flag detects the same ID outside the scoring mask within the cached crop, or touching a crop edge. It is conservative: absence does not certify a whole parcel. All metrics and flags describe ROI-clipped IDs. No independent-test claim follows from this audit.

## Pending spatial split audit

Actual checkpoint training provenance and new validation/test regions are currently unavailable. The tracked example intentionally has null paths and a false provenance attestation:

```bash
python -m experiments.audit_regions --config experiments/independent_regions.example.json --output-dir outputs/split_audit
```

This reports `pending_inputs` and lists what is missing, without opening training rasters or inventing a test mask. Leave the tracked example unchanged. When real sources are available, prepare a separate config with absolute Linux paths (or paths relative to that config):

- `reference`: raster defining the common pixel grid.
- `checkpoint`: actual evaluated BsiNet checkpoint, not the SAM checkpoint.
- `training_provenance`: explanation of records linking that checkpoint to its actual training sources.
- `all_training_footprints_declared`: explicit user attestation that training coverage is complete; do not set true for guessed candidate training folders.
- `train_mask`, `validation_mask`, `test_mask`: full-reference-grid binary 0/1 masks, not instance label TIFFs. Positive pixels must be valid. All train sources must be accounted for on the declared grid; cross-image or geographic provenance cannot be inferred by this tool.
- `buffer_pixels`: predeclared Chebyshev pixel buffer, default 256. A pair within distance <= buffer violates the check.

With complete inputs the tool checks exact dimensions, affine grid and CRS (no CRS mismatch override), nonempty binary masks, direct overlaps, and train/validation, train/test, validation/test separation. It uses block reads with halos and records input/checkpoint hashes. It does not verify annotation completeness, annotation quality, geographic generalization, or provenance truth. `spatial_checks_passed` only means the declared masks pass these geometric checks; `training_independence_confirmed` remains false. Masks that are disconnected from actual training coverage can pass geometric checks while still leaking training data.

Use validation for method decisions, reserve an untouched test region for final evaluation, and keep the repeatedly inspected development ROI out of independent-test claims. If actual old checkpoint provenance cannot be recovered, a new documented training run and fixed split may be necessary; this requires real data and is not performed automatically.
