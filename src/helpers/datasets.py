from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

class WindowTensorDataset(Dataset):
    """Simple dataset backed by a precomputed .pt file containing:
       {
         'vids': FloatTensor [N, C, H, W],
         'mels': FloatTensor [N, 1, M, T],
         'labels': LongTensor [N],
       }
    """
    def __init__(self, pt_path: str):
        super().__init__()
        blob = torch.load(pt_path, map_location='cpu')
        self.vids = blob['vids'].float()
        self.mels = blob['mels'].float()
        # Ensure mels have a channel dimension: expected [N, 1, M, T]
        if self.mels.ndim == 3:
            # currently [N, M, T] -> add channel dim at index 1
            self.mels = self.mels.unsqueeze(1)
        self.labels = blob['labels'].long()

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, idx: int):
        return self.vids[idx], self.mels[idx], self.labels[idx]
