import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import numpy as np
np.bool = bool  # Bản vá lỗi np.bool
import argparse
import mxnet as mx
from mxnet import recordio
from face_recognition.arcface.model import iresnet_inference
import albumentations as A
import logging
import pickle
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from scipy import interpolate
import random

# Kiểm tra PyTorch version để dùng torch.compile
TORCH_VERSION = torch.__version__.split('.')[0]
USE_TORCH_COMPILE = int(TORCH_VERSION) >= 2

# Cấu hình thiết bị (RTX 4090)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()  # Xóa bộ nhớ GPU trước khi chạy
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Cấu hình bộ nhớ PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)

# Hàm mất mát Label Smoothing Cross Entropy
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()

# Lớp ArcFace
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        input_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(input_norm, weight_norm).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        phi = theta + self.m
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1), 1)
        output = (torch.cos(theta) * (1 - one_hot) + torch.cos(phi) * one_hot) * self.s
        return output

# Dataset cho CASIA-WebFace với các file .rec/.idx
class FaceDataset(Dataset):
    def __init__(self, rec_path, idx_path, transform=None, cache_path="./casia_cache.pkl"):
        self.transform = transform
        logging.info(f"Opening record file: {rec_path}")
        logging.info(f"Opening index file: {idx_path}")
        self.imgrec = recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')

        if cache_path and os.path.exists(cache_path):
            logging.info(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                self.valid_keys = cache['valid_keys']
                self.label_mapping = cache['label_mapping']
        else:
            all_keys = list(self.imgrec.keys)
            logging.info(f"Found {len(all_keys)} records in the dataset")
            logging.info("Filtering invalid records (this may take a while)...")
            self.valid_keys = []
            self.label_mapping = {}
            unique_labels = set()

            for key in tqdm(all_keys, desc="Validating records"):
                try:
                    s = self.imgrec.read_idx(key)
                    header, img_bytes = recordio.unpack(s)
                    if len(img_bytes) == 0:
                        continue
                    img = mx.image.imdecode(img_bytes)
                    if img is None:
                        continue
                    raw_label = int(header.label[0] if isinstance(header.label, (list, np.ndarray)) else header.label)
                    unique_labels.add(raw_label)
                    self.valid_keys.append(key)
                except Exception as e:
                    logging.warning(f"Error processing key {key}: {str(e)}")
                    continue

            logging.info("Building label mapping...")
            self.label_mapping = {raw: idx for idx, raw in enumerate(sorted(unique_labels))}
            logging.info(f"Built label mapping for {len(self.label_mapping)} unique labels.")
            logging.info(f"After filtering, dataset contains {len(self.valid_keys)} valid records")

            if cache_path:
                with open(cache_path, 'wb') as f:
                    pickle.dump({'valid_keys': self.valid_keys, 'label_mapping': self.label_mapping}, f)
                logging.info(f"Saved dataset cache to {cache_path}")

    def __len__(self):
        return len(self.valid_keys)

    def __getitem__(self, idx):
        try:
            key = self.valid_keys[idx % len(self.valid_keys)]
            s = self.imgrec.read_idx(key)
            header, img_bytes = recordio.unpack(s)
            raw_label = int(header.label[0] if isinstance(header.label, (list, np.ndarray)) else header.label)
            label = self.label_mapping.get(raw_label, 0)
            img = mx.image.imdecode(img_bytes).asnumpy()

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            if self.transform:
                img = self.transform(image=img)['image']

            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            return preprocess(img), label
        except Exception as e:
            logging.warning(f"Failed to load image for key {key}: {str(e)}, returning dummy")
            dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
            if self.transform:
                dummy_img = self.transform(image=dummy_img)['image']
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            return preprocess(dummy_img), 0

# Augmentation nhẹ
def get_augmentation_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    ])

# Training Monitor (đã sửa lỗi)
class TrainingMonitor:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, model, optimizer, train_loss, val_loss, scheduler):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(self.save_dir, "best_model.pth"))
            logging.info(f"New best model saved: Loss = {val_loss:.4f}")
        # Lấy learning rate từ optimizer.param_groups
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

# Hàm đánh giá validation
def evaluate(model, arcface_layer, criterion, val_loader):
    model.eval()
    val_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                features = model(images)
                logits = arcface_layer(features, labels)
                loss = criterion(logits, labels)
            val_loss += loss.item()
            total += 1
    return val_loss / (total + 1e-8)

# Hàm huấn luyện với gradient accumulation
def train_model(model, arcface_layer, train_loader, val_loader, num_epochs, save_dir, initial_lr=0.001, weight_decay=1e-4, patience=5):
    model = model.to(device)
    arcface_layer = arcface_layer.to(device)

    if USE_TORCH_COMPILE:
        logging.info("Compiling model with torch.compile for RTX 4090...")
        model = torch.compile(model)
        arcface_layer = torch.compile(arcface_layer)

    optimizer = optim.SGD(list(model.parameters()) + list(arcface_layer.parameters()), lr=initial_lr, momentum=0.9, weight_decay=weight_decay)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    scaler = GradScaler()
    monitor = TrainingMonitor(save_dir)

    best_val_loss = float('inf')
    early_stopping_counter = 0
    accumulation_steps = 4  # Tăng batch size logic lên 4 * 128 = 512

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            if images.size(0) < 2:
                continue

            optimizer.zero_grad()
            with autocast():
                features = model(images)
                logits = arcface_layer(features, labels)
                loss = criterion(logits, labels) / accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("NaN/Inf loss, skipping batch")
                continue

            scaler.scale(loss).backward()
            if (batch_count + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                torch.nn.utils.clip_grad_norm_(arcface_layer.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            batch_count += 1

        avg_train_loss = running_loss / (batch_count + 1e-8)
        avg_val_loss = evaluate(model, arcface_layer, criterion, val_loader)
        scheduler.step(avg_val_loss)
        monitor.on_epoch_end(epoch, model, optimizer, avg_train_loss, avg_val_loss, scheduler)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

        torch.cuda.empty_cache()  # Giải phóng bộ nhớ sau mỗi epoch

    logging.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
    return model

# Hàm trích xuất features
def extract_features(model, loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            with autocast():
                feats = model(images)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    torch.cuda.empty_cache()  # Giải phóng bộ nhớ sau khi trích xuất
    return np.concatenate(features), np.concatenate(labels)

# Hàm tạo pairs từ test set
def generate_pairs(labels, num_pairs=6000):
    logging.info(f"Generating {num_pairs} pairs from test set...")
    pairs = []
    unique_labels = np.unique(labels)
    
    # Same pairs
    for _ in range(num_pairs // 2):
        label = random.choice(unique_labels)
        idxs = np.where(labels == label)[0]
        if len(idxs) < 2:
            continue
        idx1, idx2 = random.sample(list(idxs), 2)
        pairs.append((idx1, idx2, 1))
    
    # Different pairs
    for _ in range(num_pairs // 2):
        label1, label2 = random.sample(list(unique_labels), 2)
        idx1 = random.choice(np.where(labels == label1)[0])
        idx2 = random.choice(np.where(labels == label2)[0])
        pairs.append((idx1, idx2, 0))
    
    logging.info(f"Generated {len(pairs)} pairs successfully.")
    return pairs

# Hàm tính metrics
def compute_metrics(embeddings, labels, pairs, threshold_range=np.arange(0, 4, 0.01)):
    distances = []
    true_labels = []
    
    for idx1, idx2, is_same in pairs:
        emb1, emb2 = embeddings[idx1], embeddings[idx2]
        dist = np.linalg.norm(emb1 - emb2)
        distances.append(dist)
        true_labels.append(is_same)
    
    distances = np.array(distances)
    true_labels = np.array(true_labels)
    
    fpr, tpr, thresholds = roc_curve(true_labels, -distances)
    roc_auc = auc(fpr, tpr)
    
    fnr = 1 - tpr
    eer_threshold = threshold_range[np.argmin(np.abs(fpr - fnr))]
    eer = fpr[np.argmin(np.abs(fpr - fnr))]
    
    accuracies = []
    for thresh in threshold_range:
        predictions = distances < thresh
        acc = np.mean(predictions == true_labels)
        accuracies.append(acc)
    best_accuracy = max(accuracies)
    best_threshold = threshold_range[np.argmax(accuracies)]
    
    return {
        "EER": eer,
        "AUC": roc_auc,
        "Best Accuracy": best_accuracy,
        "Best Threshold": best_threshold,
        "FAR@0.001": interpolate.interp1d(fpr, tpr, bounds_error=False)(0.001) if min(fpr) <= 0.001 else None,
        "FAR@0.01": interpolate.interp1d(fpr, tpr, bounds_error=False)(0.01) if min(fpr) <= 0.01 else None
    }

# Hàm so sánh
def evaluate_and_compare(pretrained_model, finetuned_model, test_loader):
    logging.info("Extracting features from pretrained model...")
    pretrained_features, _ = extract_features(pretrained_model, test_loader)
    
    logging.info("Extracting features from fine-tuned model...")
    finetuned_features, test_labels = extract_features(finetuned_model, test_loader)
    
    pairs = generate_pairs(test_labels)
    
    logging.info("Computing metrics for pretrained model...")
    pretrained_metrics = compute_metrics(pretrained_features, test_labels, pairs)
    
    logging.info("Computing metrics for fine-tuned model...")
    finetuned_metrics = compute_metrics(finetuned_features, test_labels, pairs)
    
    print("\n=== Metrics Comparison ===")
    print(f"{'Metric':<20} {'Pretrained':<15} {'Fine-tuned':<15}")
    print("-" * 50)
    for metric in pretrained_metrics:
        print(f"{metric:<20} {pretrained_metrics[metric]:<15.4f} {finetuned_metrics[metric]:<15.4f}")

# Hàm chính
def main():
    parser = argparse.ArgumentParser(description="Fine-tune ArcFace on CASIA-WebFace with RTX 4090")
    parser.add_argument("--rec-path", type=str, required=True, help="Path to train.rec (CASIA-WebFace)")
    parser.add_argument("--idx-path", type=str, required=True, help="Path to train.idx (CASIA-WebFace)")
    parser.add_argument("--pretrained-path", type=str, required=True, help="Path to pretrained weights")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (RTX 4090 can handle 128)")
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    args = parser.parse_args()

    # Khởi tạo dataset
    logging.info("Initializing dataset...")
    transform = get_augmentation_transforms()
    full_dataset = FaceDataset(args.rec_path, args.idx_path, transform=transform, cache_path="./casia_cache.pkl")
    num_classes = len(full_dataset.label_mapping)

    # Chia dataset: 70% train, 20% val, 10% test
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    logging.info(f"Dataset split: {train_size} train, {val_size} val, {test_size} test samples")

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=False, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=False, prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=False, prefetch_factor=2
    )

    # Tải mô hình pretrained
    logging.info(f"Loading pretrained model from {args.pretrained_path}")
    pretrained_model = iresnet_inference("r100", args.pretrained_path, device, dropout=0.5)
    arcface_layer = ArcFace(in_features=512, out_features=num_classes, s=30.0, m=0.5)

    # Fine-tune
    logging.info("Starting training...")
    finetuned_model = train_model(pretrained_model, arcface_layer, train_loader, val_loader, args.num_epochs, args.save_dir,
                                  initial_lr=args.learning_rate, weight_decay=args.weight_decay, patience=args.patience)

    # Đánh giá và so sánh
    evaluate_and_compare(pretrained_model, finetuned_model, test_loader)

if __name__ == "__main__":
    main()