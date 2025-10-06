import argparse, glob, os, json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from helpers.decode_video import sample_frames, frame_diff_stack, get_video_info
from helpers.audio import extract_audio_array, log_mel_spectrogram
from helpers.windows import make_windows, label_windows

def load_annotations(ann_dir: str, video_id: str) -> pd.DataFrame:
    csv_path = os.path.join(ann_dir, f"{video_id}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotation CSV not found for {video_id}: {csv_path}")
    df = pd.read_csv(csv_path)
    assert {'video_id','start_sec','end_sec','event_type'} <= set(df.columns)
    return df[df['video_id'] == video_id].copy()

def infer_video_id(path: str) -> str:
    base = os.path.basename(path)
    vid = os.path.splitext(base)[0]
    return vid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', type=str, required=True, help='Glob like path/to/*.mp4')
    ap.add_argument('--ann_dir', type=str, required=True, help='Directory with per-video CSV annotations')
    ap.add_argument('--out_file', type=str, default='data/dataset_windows.pt')
    ap.add_argument('--fps', type=int, default=4)
    ap.add_argument('--resize', type=int, default=224)
    ap.add_argument('--win_sec', type=float, default=2.0)
    ap.add_argument('--stride_sec', type=float, default=0.5)
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--n_mels', type=int, default=64)
    ap.add_argument('--mel_win', type=float, default=0.025)
    ap.add_argument('--mel_hop', type=float, default=0.010)
    ap.add_argument('--horizon_sec', type=float, default=1.0)
    args = ap.parse_args()

    # mel window/hop provided on the CLI are in seconds (float). Convert to samples
    # for librosa which expects integer sample counts for win_length and hop_length.
    mel_hop_samples = max(1, int(round(args.mel_hop * args.sr)))
    mel_win_samples = max(1, int(round(args.mel_win * args.sr)))

    files = sorted(glob.glob(args.videos))
    if not files:
        raise SystemExit("No videos matched the glob pattern.")

    all_vids, all_mels, all_labels = [], [], []
    for vid_path in tqdm(files, desc='Processing videos'):
        vid_id = infer_video_id(vid_path)
        ann = load_annotations(args.ann_dir, vid_id)

        # frames and frame-diff stacks per sampled frame index
        frames = sample_frames(vid_path, fps=args.fps, resize=args.resize)
        diff_stacks = []
        for i in range(len(frames)):
            if i < 1:
                # first item: replicate image channels to match diff shape later
                diff_stacks.append(frames[i].repeat(3,1,1))
            else:
                diff_stacks.append(torch.cat([frames[i]-frames[i-1]]*3, dim=0))  # [3,H,W]

        # audio → log-mel
        audio = extract_audio_array(vid_path, sr=args.sr)
        mel = log_mel_spectrogram(
            audio,
            sr=args.sr,
            n_mels=args.n_mels,
            win_length=mel_win_samples,
            hop_length=mel_hop_samples,
        )
        mel = torch.from_numpy(mel).unsqueeze(0)  # Convert to torch and add batch dim: (64, T) -> (1, 64, T)

        # windows
        vids, mels, centers = make_windows(diff_stacks, mel, fps=args.fps,
                                           win_sec=args.win_sec, stride_sec=args.stride_sec,
                                           mel_hop_s=args.mel_hop)
        if not vids:
            continue

        # labels
        labels = label_windows(centers, ann_df=ann, horizon_sec=args.horizon_sec)

        all_vids.append(torch.stack(vids))           # [N, 3, H, W]
        all_mels.append(torch.stack(mels).squeeze(1))
        all_labels.append(labels)

    V = torch.cat(all_vids, dim=0) if all_vids else torch.empty(0)
    A = torch.cat(all_mels, dim=0) if all_mels else torch.empty(0)
    Y = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    torch.save({'vids': V, 'mels': A, 'labels': Y}, args.out_file)
    print(f"Saved: {args.out_file} | vids={tuple(V.shape)} mels={tuple(A.shape)} labels={tuple(Y.shape)}")

if __name__ == '__main__':
    main()
