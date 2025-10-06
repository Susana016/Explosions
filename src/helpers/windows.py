from typing import List, Tuple, Dict
import numpy as np
import torch
import pandas as pd

def build_timegrid(num_frames: int, fps: int) -> np.ndarray:
    """Return center time (sec) for each sampled frame index."""
    return np.arange(num_frames, dtype=np.float32) / float(fps)

def slice_mel_by_time(mel: torch.Tensor, sr: int, hop_s: float, t_start: float, t_end: float) -> torch.Tensor:
    """Slice mel [1,M,T] by time range (seconds) and pad/truncate to exact length."""
    # hop_s is the time per hop (sec). Index range:
    i0 = int(np.floor(t_start / hop_s))
    i1 = int(np.ceil(t_end   / hop_s))
    i0 = max(i0, 0)
    i1 = min(i1, mel.shape[-1])
    sl = mel[..., i0:i1]
    return sl

def label_windows(window_centers: List[float], ann_df: pd.DataFrame, horizon_sec: float = 1.0) -> torch.Tensor:
    """Binary label = 1 if an 'explosion' event starts within horizon after window end.
    Expected columns in ann_df: video_id, start_sec, end_sec, event_type
    """
    starts = ann_df[ann_df['event_type']=='explosion']['start_sec'].values.astype(float)
    labels = []
    for t_end in window_centers:
        # if any start in (t_end, t_end + horizon], label 1
        labels.append(1 if np.any((starts > t_end) & (starts <= t_end + horizon_sec)) else 0)
    return torch.tensor(labels, dtype=torch.long)

def make_windows(frames: List[torch.Tensor], mel: torch.Tensor, fps: int,
                 win_sec: float = 2.0, stride_sec: float = 0.5,
                 mel_hop_s: float = 0.010) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
    """Create aligned windows:
      - video: take the last k frame-diffs within each window (computed upstream)
      - audio: slice mel for [t0, t1] matching the window
    Return lists: [vid_tensor], [mel_tensor], [window_end_time_sec]
    """
    num_frames = len(frames)
    if num_frames == 0:
        return [], [], []
    stride_f = max(int(round(stride_sec * fps)), 1)
    win_f = max(int(round(win_sec * fps)), 1)

    vids, auds, centers = [], [], []
    for end_idx in range(win_f, num_frames, stride_f):
        start_idx = end_idx - win_f
        t0 = start_idx / float(fps)
        t1 = end_idx   / float(fps)
        # video stack: use last frame-diff stack prepared by caller (shape [C,H,W])
        v_stack = frames[end_idx-1]  # assume caller passed diff stacks per frame index
        a_slice = slice_mel_by_time(mel, sr=1, hop_s=mel_hop_s, t_start=t0, t_end=t1)  # sr not used
        # enforce fixed time width by padding or truncating along time dim
        need_T = int(round((t1 - t0) / mel_hop_s))
        cur_T = a_slice.shape[-1]
        if cur_T < need_T:
            pad = need_T - cur_T
            a_slice = torch.nn.functional.pad(a_slice, (0,pad), mode='constant', value=a_slice.min().item() if cur_T>0 else 0.0)
        elif cur_T > need_T:
            a_slice = a_slice[..., :need_T]
        vids.append(v_stack)
        auds.append(np.expand_dims(a_slice, 0) if a_slice.ndim==2 else a_slice)
        centers.append(t1)
    return vids, auds, centers
