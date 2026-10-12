"""Conservative per-instance prompt ablation; inference never reads ground truth."""
import numpy as np
from scipy import ndimage


def interior_point(mask):
    # Padding makes image-border objects obey the same distance convention.
    distances = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(distances), mask.shape)
    return np.array([x, y], dtype=np.float32), float(distances[y, x])


def generate_points(rgb, region, mode, args):
    if mode == 'box':
        return [None]
    if mode == 'interior':
        return [interior_point(region)[0]]
    from skimage.segmentation import slic
    # SLIC is computed within each predicted component, never on a GT mask.
    segments = slic(rgb, n_segments=max(1, int(np.ceil(region.sum() / args.superpixel_area))),
                    compactness=args.compactness, mask=region, start_label=1, channel_axis=-1)
    ranked = []
    for label_id in np.unique(segments[region]):
        if label_id == 0:
            continue
        point, radius = interior_point((segments == label_id) & region)
        if radius >= args.min_radius:
            ranked.append((radius, point))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return [interior_point(region)[0]]
    return [point for _, point in ranked[:args.max_points]]


def refine_image(rgb, original, predictor, args):
    output = np.zeros(original.shape, dtype=np.uint32)
    records = []
    next_id = 1
    predictor.set_image(rgb)
    structure = np.ones((3, 3), dtype=bool)

    def assign(mask):
        nonlocal next_id
        if not mask.any():
            return
        if next_id > np.iinfo(np.uint32).max:
            raise ValueError('Instance ID overflow')
        output[mask] = next_id
        next_id += 1

    for original_id in np.unique(original[original > 0]):
        components, count = ndimage.label(original == original_id, structure=structure)
        for component_id in range(1, count + 1):
            region = components == component_id
            area = int(region.sum())
            if area < args.min_area:
                assign(region)
                records.append({'source_id': int(original_id), 'component': component_id,
                                'reason': 'small_component_preserved', 'prompt_count': 0})
                continue
            ys, xs = np.where(region)
            box = np.array([max(0, xs.min() - args.box_pad), max(0, ys.min() - args.box_pad),
                            min(rgb.shape[1] - 1, xs.max() + args.box_pad),
                            min(rgb.shape[0] - 1, ys.max() + args.box_pad)], dtype=np.float32)
            if args.support_pad:
                support = ndimage.binary_dilation(region, structure=structure, iterations=args.support_pad)
            else:
                support = region.copy()
            # Never overwrite another BsiNet object, including another component of this ID.
            support &= (original == 0) | region
            support &= output == 0
            points = generate_points(rgb, region, args.mode, args)
            candidates = []
            for point in points:
                masks, sam_scores, logits = predictor.predict(
                    point_coords=None if point is None else point[None],
                    point_labels=None if point is None else np.ones(1, dtype=np.int32),
                    box=box, multimask_output=True, return_logits=True)
                # Full-resolution logits are used for the official SAM stability convention.
                for raw, score in zip(masks, sam_scores):
                    if not np.isfinite(score) or not np.all(np.isfinite(raw)):
                        continue
                    binary = raw > predictor.model.mask_threshold
                    low = raw > predictor.model.mask_threshold - args.stability_offset
                    high = raw > predictor.model.mask_threshold + args.stability_offset
                    stability = float(high.sum() / low.sum()) if low.any() else 0.0
                    if score < args.min_score or stability < args.min_stability:
                        continue
                    if point is not None:
                        x, y = point.astype(int)
                        if not binary[y, x]:
                            continue
                        cc, _ = ndimage.label(binary, structure=structure)
                        binary = cc == cc[y, x]
                    raw_area = int(binary.sum())
                    if not raw_area or (binary & support).sum() / raw_area < args.min_support:
                        continue
                    mask = binary & support
                    if mask.sum() < args.min_area or not (mask & region).any():
                        continue
                    candidates.append((stability, float(score), mask))
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            occupied = np.zeros(region.shape, dtype=bool)
            accepted = []
            for stability, score, candidate in candidates:
                overlap = float((candidate & occupied).sum() / candidate.sum())
                if overlap > args.max_overlap:
                    continue
                candidate = candidate & ~occupied
                # Clipping can create islands: retain only the largest connected parcel.
                cc, n = ndimage.label(candidate, structure=structure)
                if n:
                    sizes = np.bincount(cc.ravel())
                    sizes[0] = 0
                    candidate = cc == sizes.argmax()
                if candidate.sum() < args.min_area or not (candidate & region).any():
                    continue
                occupied |= candidate
                accepted.append((candidate, stability, score))
            coverage = float((occupied & region).sum() / area)
            if coverage < args.min_coverage:
                assign(region)
                reason = 'low_coverage_original_preserved'
                accepted = []
                residual_area = area
            else:
                for mask, _, _ in accepted:
                    assign(mask)
                # Preserve uncovered original pixels; SAM must not silently delete parcels.
                residual = region & ~occupied
                residual_area = int(residual.sum())
                cc, n = ndimage.label(residual, structure=structure)
                for label_id in range(1, n + 1):
                    assign(cc == label_id)
                reason = 'accepted_with_residual_preservation'
            records.append({'source_id': int(original_id), 'component': component_id,
                            'reason': reason, 'prompt_count': len(points),
                            'box_xyxy': box.tolist(),
                            'points_xy': [None if point is None else point.tolist() for point in points],
                            'candidate_count': len(candidates), 'accepted_count': len(accepted),
                            'coverage': coverage, 'residual_pixels': residual_area,
                            'stability_scores': [stability for _, stability, _ in accepted],
                            'sam_scores': [score for _, _, score in accepted]})
    return output, records
