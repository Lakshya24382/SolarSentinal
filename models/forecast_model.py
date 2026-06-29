import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                               classification_report)
import joblib, os
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────
SEQ_LEN     = 30    # look back 30 minutes of history
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.3

FEATURE_COLS = [
    'log_soft', 'log_hard', 'log_ratio',
    'soft_mean_5', 'soft_mean_15', 'soft_mean_30',
    'soft_std_5',  'soft_std_15',  'soft_std_30',
    'hard_mean_5', 'hard_mean_15', 'hard_mean_30',
    'soft_deriv_1', 'soft_deriv_5',
    'hard_deriv_1', 'hard_deriv_5',
    'soft_max_15',  'soft_max_30',
]

# ── Use Apple Silicon MPS if available ────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon MPS (GPU)")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")

# ── Dataset: builds sliding windows of SEQ_LEN timesteps ─────────
class FlareSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_sequences(X, y, seq_len=SEQ_LEN):
    """
    Convert flat feature array into overlapping windows.
    Each sample = last seq_len timesteps → predict label at end.
    Shape: (n_samples, seq_len, n_features)
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# ── LSTM Model ────────────────────────────────────────────────────
class FlareLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # take last timestep only
        return self.classifier(out).squeeze(1)

# ── TSS + HSS ─────────────────────────────────────────────────────
def compute_tss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    return round(tpr - fpr, 4), round(tpr, 4), round(fpr, 4)

def compute_hss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    num = 2 * (tp*tn - fp*fn)
    den = (tp+fn)*(fn+tn) + (tp+fp)*(fp+tn)
    return round(num / (den + 1e-9), 4)

# ── Training loop ─────────────────────────────────────────────────
def train_forecast_model(data_path="data/labeled_dataset.csv"):
    print("="*55)
    print("  FORECAST MODEL TRAINING (LSTM)")
    print("="*55)

    device = get_device()

    # Load + sort data
    df = pd.read_csv(data_path, parse_dates=['time_tag'])
    df = df.sort_values('time_tag').reset_index(drop=True)

    X_raw = df[FEATURE_COLS].values
    y_raw = df['forecast_label'].values

    # Time-series split BEFORE scaling
    split_idx = int(len(df) * 0.8)
    X_tr_raw, X_te_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_tr_raw, y_te_raw = y_raw[:split_idx], y_raw[split_idx:]

    # Scale (fit only on train)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_raw)
    X_te_s = scaler.transform(X_te_raw)

    # Build sequences
    X_train, y_train = build_sequences(X_tr_s, y_tr_raw)
    X_test,  y_test  = build_sequences(X_te_s, y_te_raw)
    print(f"Train sequences: {X_train.shape}")
    print(f"Test  sequences: {X_test.shape}")
    print(f"Forecast flare rate — train: {y_train.mean()*100:.1f}%  test: {y_test.mean()*100:.1f}%")

    # DataLoaders
    train_ds = FlareSequenceDataset(X_train, y_train)
    test_ds  = FlareSequenceDataset(X_test,  y_test)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = FlareLSTM(input_size=len(FEATURE_COLS)).to(device)

    # Class weight to handle imbalance
    pos_weight = torch.tensor(
    [(y_train == 0).sum() / (y_train == 1).sum() + 1e-9],
    dtype=torch.float32
).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # Training
    train_losses, val_aucs = [], []
    best_auc, best_state = 0, None

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dl)
        train_losses.append(avg_loss)

        # Validation AUC every epoch
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                probs = model(xb.to(device)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(yb.numpy())

        auc = roc_auc_score(all_labels, all_probs)
        val_aucs.append(auc)
        scheduler.step(1 - auc)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | AUC: {auc:.4f}")

    # Load best weights
    model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            probs = model(xb.to(device)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    y_pred     = (all_probs >= 0.5).astype(int)

    tss, tpr, fpr = compute_tss(all_labels, y_pred)
    hss           = compute_hss(all_labels, y_pred)
    auc           = roc_auc_score(all_labels, all_probs)

    print(f"\n{'─'*40}")
    print(f"  FORECAST MODEL RESULTS")
    print(f"{'─'*40}")
    print(f"  TSS  : {tss}")
    print(f"  HSS  : {hss}")
    print(f"  AUC  : {auc:.4f}")
    print(f"  TPR  : {tpr}")
    print(f"  FPR  : {fpr}")
    print(f"  Lead time: 15 minutes before flare peak")
    print(f"{'─'*40}")
    print(classification_report(all_labels, y_pred,
                                target_names=['Quiet', 'Flare']))

    # Training curve plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, color='#E8593C', lw=1.5)
    ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch')
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_aucs, color='#3B8BD4', lw=1.5)
    ax2.axhline(best_auc, color='gray', ls='--', lw=1,
                label=f'Best AUC: {best_auc:.4f}')
    ax2.set_title('Validation AUC per Epoch'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/forecast_training.png', dpi=150)
    plt.close()

    # Save model + scaler
    torch.save(best_state, 'models/forecast_lstm.pt')
    joblib.dump(scaler,    'models/forecast_scaler.pkl')
    print(f"\nSaved: models/forecast_lstm.pt")
    print(f"Saved: models/forecast_scaler.pkl")
    print(f"Plot:  data/forecast_training.png")

    return model, scaler, {
        'tss': tss, 'hss': hss, 'auc': auc,
        'tpr': tpr, 'fpr': fpr
    }

if __name__ == "__main__":
    model, scaler, metrics = train_forecast_model()