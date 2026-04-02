import argparse
import json
import os
import random
from dataclasses import dataclass

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.model import ResNetTransferModel


@dataclass
class Sample:
    label: int
    path: str


class CaltechDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        image = self.transform(image)
        return image, sample.label


def build_samples(train_entries, data_root):
    samples = []
    for label, rel_path in train_entries:
        full_path = rel_path
        if not os.path.exists(full_path):
            full_path = os.path.join(data_root, rel_path)
        if os.path.exists(full_path):
            samples.append(Sample(label=int(label), path=full_path))
    return samples


def stratified_split(samples, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    train_split = []
    val_split = []

    for _, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        if len(items) == 1:
            n_val = 0
        val_split.extend(items[:n_val])
        train_split.extend(items[n_val:])

    rng.shuffle(train_split)
    rng.shuffle(val_split)
    return train_split, val_split


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Quick trainer for workshop7 webapp model")
    parser.add_argument("--data", default="train_val.json", help="Path to train_val.json")
    parser.add_argument("--output", default="weights/model.pth", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=6000,
        help="Cap train samples for speed (0 disables cap)")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint or not")
    parser.add_argument("--checkpoint", type=str, default='weights/model.pth', help="Checkpoint file path for resuming")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.data, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    train_entries = dataset_json["train"]
    categories = dataset_json["categories"]
    num_classes = len(categories)

    data_root = os.path.dirname(os.path.abspath(args.data))
    samples = build_samples(train_entries, data_root)
    if not samples:
        raise RuntimeError("No valid training images found from train_val.json paths")

    train_samples, val_samples = stratified_split(samples, val_ratio=args.val_ratio, seed=args.seed)

    if args.max_train_samples > 0 and len(train_samples) > args.max_train_samples:
        train_samples = random.sample(train_samples, args.max_train_samples)

    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | Classes: {num_classes}")

    train_tf = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            #Changes added here: color jitter
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            #Changes added here: random rotation
            #transforms.RandomRotation(degrees=15),
            #Changes added here: gaussian blur
            #transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CaltechDataset(train_samples, train_tf)
    val_ds = CaltechDataset(val_samples, eval_tf)

    num_workers = 2 if os.cpu_count() and os.cpu_count() > 2 else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = ResNetTransferModel(num_classes=num_classes, embedding_size=128, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    best_val_acc = -1.0
    best_state = None
    
    if args.resume:
        checkpoint_file = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint_file["model_state_dict"])
        best_val_acc = checkpoint_file.get("best_val_acc", best_val_acc)
        optimizer.load_state_dict(checkpoint_file.get("optimizer_state_dict", optimizer.state_dict()))
        ran_epochs = checkpoint_file.get("epochs", 0)
        print(f"Resumed from checkpoint: {args.checkpoint}")
        epoch_start = ran_epochs + 1
        epoch_end = ran_epochs + args.epochs
    else:
        epoch_start = 1
        epoch_end = args.epochs + 1

    for epoch in range(epoch_start, epoch_end):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, y in progress:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y).sum().item()
            running_total += x.size(0)
            progress.set_postfix(
                loss=f"{running_loss / max(running_total, 1):.4f}",
                acc=f"{running_correct / max(running_total, 1):.4f}",
            )

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            # TODO: Save the checkpoint of the best validation accuracy in addition to the last checkpoint
        
        scheduler.step()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "num_classes": num_classes,
        "categories": categories,
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, args.output)
    print(f"Saved checkpoint to: {args.output}")


if __name__ == "__main__":
    main()