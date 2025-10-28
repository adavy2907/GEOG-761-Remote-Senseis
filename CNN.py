# ============================================================
# Landslide Detection Training + Evaluation Script (Updated with ROC)
# ============================================================
import os
import random
from glob import glob
from tqdm import tqdm
import torchviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, auc
)

# ============================================================
# SETTINGS
# ============================================================
RUN_MODE = "evaluate"   # "train" or "evaluate"
DATA_DIR = "arrays/content/arrays"
MODEL_PATH = "best_model.pth"
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4

# ============================================================
# Dataset
# ============================================================
class LandslideDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        arr = np.load(self.filepaths[idx]).astype(np.float32)
        arr = np.clip(arr, 0, None)
        for b in range(arr.shape[-1]):
            p2, p98 = np.percentile(arr[..., b], [2, 98])
            if p98 - p2 > 0:
                arr[..., b] = (arr[..., b] - p2) / (p98 - p2)
            else:
                arr[..., b] /= (arr[..., b].max() + 1e-6)

        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        label = torch.tensor(self.labels[idx]).float()
        if self.transform:
            tensor, label = self.transform(tensor, label)
        return tensor, label

# ============================================================
# Simple Augmentation
# ============================================================
class SimpleAugment:
    def __call__(self, x, y):
        if random.random() < 0.5:
            x = TF.hflip(x) if random.random() < 0.5 else TF.vflip(x)
        if random.random() < 0.3:
            k = random.choice([1, 2, 3])
            x = torch.rot90(x, k, [1, 2])
        return x, y

# ============================================================
# Model (outputs probabilities)
# ============================================================
class SmallCNN(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()  # outputs probability
        )

    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(1)

# ============================================================
# Weighted BCELoss for class imbalance
# ============================================================
def weighted_bce_loss(probs, targets, pos_weight):
    weights = torch.ones_like(targets)
    weights[targets == 1] = pos_weight
    loss = nn.BCELoss(weight=weights)(probs, targets)
    return loss

# ============================================================
# Training / Validation helpers
# ============================================================
def train_one_epoch(model, loader, opt, pos_weight, device):
    model.train()
    total_loss = 0
    preds, trues = [], []

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()

        probs = model(x)
        loss = weighted_bce_loss(probs, y, pos_weight)
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        preds.extend(probs.detach().cpu().numpy().tolist())
        trues.extend(y.detach().cpu().numpy().tolist())

    return total_loss / len(loader.dataset), preds, trues

def validate(model, loader, pos_weight, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = model(x)
            loss = weighted_bce_loss(probs, y, pos_weight)
            total_loss += loss.item() * x.size(0)
            preds.extend(probs.cpu().numpy().tolist())
            trues.extend(y.cpu().numpy().tolist())

    return total_loss / len(loader.dataset), preds, trues

# ============================================================
# Confusion Matrix Plot
# ============================================================
# def plot_confusion_matrix(y_true, y_pred_labels):
#     cm = confusion_matrix(y_true, y_pred_labels)
#     labels = ["Non-Landslide", "Landslide"]

#     # Compute row percentages
#     row_sums = cm.sum(axis=1, keepdims=True)
#     pct = np.round(cm / (row_sums + 1e-9) * 100, 1)

#     # Combine counts and percentages
#     annot = np.array([f"{count}\n({pc}%)" for count, pc in zip(cm.flatten(), pct.flatten())])
#     annot = annot.reshape(cm.shape)

#     fig, ax = plt.subplots(figsize=(6, 6))
#     im = ax.imshow(cm, cmap="RdBu", aspect="equal")

#     # Add text annotations manually
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, annot[i, j],
#                     ha="center", va="center", fontsize=11, color="white")

#     # Axis labels and ticks
#     ax.set_xticks(np.arange(len(labels)))
#     ax.set_yticks(np.arange(len(labels)))
#     ax.set_xticklabels(labels)
#     ax.set_yticklabels(labels)
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("Actual")
#     ax.set_title("Confusion Matrix")

#     # Add colorbar
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label("Number of Samples")

#     # ✅ Critical fix: ensure full matrix is drawn
#     ax.set_xlim(-0.5, cm.shape[1]-0.5)
#     ax.set_ylim(cm.shape[0]-0.5, -0.5)

#     plt.tight_layout()
#     plt.show()

def plot_confusion_matrix(y_true, y_pred_labels):
    cm = confusion_matrix(y_true, y_pred_labels)
    labels = ["Non-Landslide", "Landslide"]

    # Compute row percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.round(cm / (row_sums + 1e-9) * 100, 1)

    # Combine counts and percentages
    annot = np.array([f"{count}\n({pc}%)" for count, pc in zip(cm.flatten(), pct.flatten())])
    annot = annot.reshape(cm.shape)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    # Axis labels and ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix\n(Landslide = Positive Class)",
                 fontsize=14, fontweight="bold", pad=12)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of Samples", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # ✅ Ensure full matrix and labels render properly
    ax.set_xlim(-0.5, cm.shape[1] - 0.5)
    ax.set_ylim(cm.shape[0] - 0.5, -0.5)

    plt.tight_layout()
    plt.show()



# -------------------------
# Weighted Average Metrics Plot
# -------------------------
def plot_weighted_metrics(weighted, accuracy):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [accuracy, weighted["precision"], weighted["recall"], weighted["f1-score"]]
    
    # Same colour style as your previous image
    colors = ["#0000FF", "#4CAF50", "#FFCA28", "#FF5252"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(metrics, values, color=colors, alpha=0.9)

    # Add text labels above bars
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            val + 0.02, 
            f"{val:.3f}", 
            ha="center", 
            va="bottom", 
            fontsize=11, 
            fontweight="bold"
        )

    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Weighted Average Performance Metrics - Test Set", fontweight="bold")
    plt.tight_layout()
    plt.show()

# ============================================================
# ROC Curve Plot
# ============================================================
def plot_roc(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Load dataset paths
    landslide_files = glob(f"{DATA_DIR}/*/landslide/*.npy", recursive=True)
    nonlandslide_files = glob(f"{DATA_DIR}/*/nonlandslide/*.npy", recursive=True)
    files = landslide_files + nonlandslide_files
    labels = [1]*len(landslide_files) + [0]*len(nonlandslide_files)

    # Split datasets
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files, labels, test_size=0.3, stratify=labels, random_state=42)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    print(np.unique(test_labels, return_counts=True))

    # Create datasets/loaders
    train_ds = LandslideDataset(train_files, train_labels, transform=SimpleAugment())
    val_ds = LandslideDataset(val_files, val_labels)
    test_ds = LandslideDataset(test_files, test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    in_channels = np.load(files[0]).shape[-1]
    model = SmallCNN(in_ch=in_channels).to(device)

    # Calculate pos_weight for imbalance
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    pos_weight = num_neg / (num_pos + 1e-6)
    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    # Optimizer and scheduler
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    if RUN_MODE == "train":
        best_val_f1 = 0
        for epoch in range(1, EPOCHS+1):
            print(f"\nEpoch {epoch}")
            train_loss, _, _ = train_one_epoch(model, train_loader, opt, pos_weight_tensor, device)
            val_loss, val_preds, val_trues = validate(model, val_loader, pos_weight_tensor, device)
            val_pred_labels = (np.array(val_preds) >= 0.5).astype(int)
            val_f1 = f1_score(val_trues, val_pred_labels)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            scheduler.step(val_loss)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), MODEL_PATH)
                print("✅ Saved best model.")

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------
    print("\n=== EVALUATION ON TEST SET ===")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    test_loss, test_preds, test_trues = validate(model, test_loader, pos_weight_tensor, device)
    test_pred_labels = (np.array(test_preds) >= 0.5).astype(int)

    test_f1 = f1_score(test_trues, test_pred_labels)
    test_acc = accuracy_score(test_trues, test_pred_labels)
    test_auc = roc_auc_score(test_trues, test_preds) if len(set(test_trues)) > 1 else np.nan
    print(f"Test Loss: {test_loss:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f} | Acc: {test_acc:.4f}")

    # Save CSV
    results_df = pd.DataFrame({
        "file": test_files,
        "true_label": test_trues,
        "predicted_prob": test_preds,
        "predicted_label": test_pred_labels
    })
    results_df.to_csv("test_predictions.csv", index=False)
    print("✅ Saved test_predictions.csv")

    # Confusion matrix
    plot_confusion_matrix(test_trues, test_pred_labels)

    # ROC curve
    plot_roc(test_trues, test_preds)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_trues, test_pred_labels,
                                target_names=["Non-Landslide", "Landslide"], digits=3))

    # Show some FP/FN examples
    fp = results_df[(results_df["true_label"] == 0) & (results_df["predicted_label"] == 1)]
    fn = results_df[(results_df["true_label"] == 1) & (results_df["predicted_label"] == 0)]

    print(f"\nFalse Positives: {len(fp)}")
    print(fp['file'].head().tolist())
    print(f"\nFalse Negatives: {len(fn)}")
    print(fn['file'].head().tolist())

    # -------------------------
    # Compute metrics
    # -------------------------
    acc = accuracy_score(test_trues, test_pred_labels)
    report = classification_report(
        test_trues,
        test_pred_labels,
        target_names=["Non-Landslide", "Landslide"],
        output_dict=True
    )
    weighted = report["weighted avg"]
    plot_weighted_metrics(weighted, acc)

    # Print summary
    summary(model, input_size=(1, in_channels, 128, 128))
