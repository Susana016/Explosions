import numpy as np
from typing import List, Tuple

def event_level_f1(prob: np.ndarray, times: np.ndarray, gt_events: List[Tuple[float,float]],
                   threshold: float = 0.5, tol: float = 0.5) -> Tuple[float,int,int,int]:
    """Compute a simple event-level F1:
       - A predicted event is any contiguous region where prob >= threshold.
       - It is a TP if its peak time is within tol seconds of any GT event start.
       - Otherwise it's FP. Any unmatched GT is FN.
    Returns: (F1, TP, FP, FN).
    """
    # find predicted peaks
    pred_idxs = np.where(prob >= threshold)[0]
    if pred_idxs.size == 0:
        TP = 0
        FP = 0
        FN = len(gt_events)
        f1 = 0.0 if (TP+FP+FN)>0 else 1.0
        return f1, TP, FP, FN

    # group contiguous indices
    groups = []
    start = pred_idxs[0]
    prev = pred_idxs[0]
    for i in pred_idxs[1:]:
        if i == prev + 1:
            prev = i
        else:
            groups.append((start, prev))
            start = i
            prev = i
    groups.append((start, prev))

    peaks = [int((a+b)//2) for (a,b) in groups]
    peak_times = times[peaks]

    # match to GT
    gt_starts = np.array([s for (s,_) in gt_events], dtype=float)
    matched_gt = np.zeros(len(gt_events), dtype=bool)
    TP = 0
    for pt in peak_times:
        if gt_starts.size == 0:
            break
        d = np.abs(gt_starts - pt)
        j = int(np.argmin(d))
        if d[j] <= tol and not matched_gt[j]:
            matched_gt[j] = True
            TP += 1
        else:
            # leave counting FP for later
            pass

    FP = len(peak_times) - TP
    FN = len(gt_events) - TP
    denom = (2*TP + FP + FN)
    f1 = (2*TP) / denom if denom > 0 else 1.0
    return f1, TP, FP, FN
