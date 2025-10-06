import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models.explex_net import ExplexNet
from helpers.datasets import WindowTensorDataset

def train_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for vids, mels, labels in tqdm(loader, desc='train', leave=False):
        vids, mels, labels = vids.to(device), mels.to(device), labels.float().to(device)
        optimizer.zero_grad()
        logits = model(vids, mels).squeeze(1)       # [B]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)

@torch.inference_mode()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    ys, ps = [], []
    for vids, mels, labels in tqdm(loader, desc='valid', leave=False):
        vids, mels, labels = vids.to(device), mels.to(device), labels.float().to(device)
        logits = model(vids, mels).squeeze(1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        ys.extend(labels.cpu().numpy().tolist())
        ps.extend(probs.tolist())
    try:
        auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else 0.5
    except Exception:
        auc = 0.5
    return total_loss / len(loader.dataset), auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_pt', type=str, default='data/dataset_windows.pt')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--val_split', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_path', type=str, default='weights/best.pt')
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    ds = WindowTensorDataset(args.dataset_pt)
    n_val = max(int(len(ds) * args.val_split), 1)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # infer video input channels from dataset (expect vids shape [N, C, H, W])
    sample_vid, _, _ = ds[0]
    vid_in_ch = sample_vid.shape[0]
    print(f"Using device={device} | video input channels={vid_in_ch}")

    model = ExplexNet(vid_in_ch=vid_in_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_auc = eval_epoch(model, val_loader, device, criterion)
        print(f"epoch {ep} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_auc {va_auc:.3f}")
        if va_auc > best_auc:
            best_auc = va_auc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"✓ Saved new best to {args.save_path}")

if __name__ == '__main__':
    main()
