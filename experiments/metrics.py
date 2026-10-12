"""Pixel counts and BsiNet-compatible greedy instance matching (IoU >= 0.5)."""
import numpy as np


def divide(a, b):
    return float(a / b) if b else 0.0


def pixel_counts(prediction, target):
    p, t = prediction > 0, target > 0
    return {'pixel_tp': int(np.sum(p & t)), 'pixel_fp': int(np.sum(p & ~t)),
            'pixel_fn': int(np.sum(~p & t))}


def instance_counts(prediction, target):
    pred_ids, pred_counts = np.unique(prediction[prediction > 0], return_counts=True)
    gt_ids, gt_counts = np.unique(target[target > 0], return_counts=True)
    pred_areas = dict(zip(pred_ids.tolist(), pred_counts.tolist()))
    gt_areas = dict(zip(gt_ids.tolist(), gt_counts.tolist()))
    overlap = (prediction > 0) & (target > 0)
    pairs, counts = np.unique(np.stack([prediction[overlap], target[overlap]], axis=1),
                              axis=0, return_counts=True)
    candidates = []
    for (p, g), count in zip(pairs, counts):
        iou = divide(count, pred_areas[int(p)] + gt_areas[int(g)] - count)
        if iou >= 0.5:
            candidates.append((iou, int(p), int(g)))
    used_p, used_g, matched = set(), set(), []
    for iou, p, g in sorted(candidates, reverse=True):
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
            matched.append(iou)
    return {'instance_tp': len(matched), 'instance_fp': len(pred_ids) - len(matched),
            'instance_fn': len(gt_ids) - len(matched), 'matched_iou_sum': float(sum(matched))}


def label_boundary(labels):
    boundary = np.zeros(labels.shape, dtype=bool)
    difference = labels[1:] != labels[:-1]
    boundary[1:] |= difference & (labels[1:] > 0)
    boundary[:-1] |= difference & (labels[:-1] > 0)
    difference = labels[:, 1:] != labels[:, :-1]
    boundary[:, 1:] |= difference & (labels[:, 1:] > 0)
    boundary[:, :-1] |= difference & (labels[:, :-1] > 0)
    return boundary


def boundary_counts(prediction, target, tolerance):
    from scipy.ndimage import maximum_filter
    p, t = label_boundary(prediction), label_boundary(target)
    size = 2 * tolerance + 1
    return {'boundary_pred': int(p.sum()), 'boundary_gt': int(t.sum()),
            'boundary_matched_pred': int(np.sum(p & maximum_filter(t, size=size, mode='constant'))),
            'boundary_matched_gt': int(np.sum(t & maximum_filter(p, size=size, mode='constant')))}


def scores(c):
    tp, fp, fn = (c[f'pixel_{key}'] for key in ('tp', 'fp', 'fn'))
    itp, ifp, ifn = (c[f'instance_{key}'] for key in ('tp', 'fp', 'fn'))
    bp = divide(c['boundary_matched_pred'], c['boundary_pred'])
    br = divide(c['boundary_matched_gt'], c['boundary_gt'])
    return {'mask_precision': divide(tp, tp + fp), 'mask_recall': divide(tp, tp + fn),
            'mask_iou': divide(tp, tp + fp + fn), 'mask_dice': divide(2 * tp, 2 * tp + fp + fn),
            'instance_precision': divide(itp, itp + ifp), 'instance_recall': divide(itp, itp + ifn),
            'instance_f1': divide(2 * itp, 2 * itp + ifp + ifn),
            'instance_mean_iou': divide(c['matched_iou_sum'], itp),
            'pq': divide(c['matched_iou_sum'], itp + 0.5 * ifp + 0.5 * ifn),
            'boundary_precision': bp, 'boundary_recall': br, 'boundary_f1': divide(2 * bp * br, bp + br)}
