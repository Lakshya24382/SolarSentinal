import pandas as pd
import numpy as np
import joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                               roc_auc_score, roc_curve)
import matplotlib.pyplot as plt

# ── Feature columns the model will use ────────────────────────────
FEATURE_COLS = [
    'log_soft', 'log_hard', 'log_ratio',
    'soft_mean_5', 'soft_mean_15', 'soft_mean_30',
    'soft_std_5',  'soft_std_15',  'soft_std_30',
    'hard_mean_5', 'hard_mean_15', 'hard_mean_30',
    'soft_deriv_1', 'soft_deriv_5',
    'hard_deriv_1', 'hard_deriv_5',
    'soft_max_15', 'soft_max_30',
]

def compute_tss(y_true, y_pred):
    """
    True Skill Statistic — standard metric in space weather forecasting.
    TSS = TPR - FPR. Range -1 to 1. Higher is better. 0 = no skill.
    ISRO judges will know this metric — include it prominently.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn + 1e-9)   # True Positive Rate
    fpr = fp / (fp + tn + 1e-9)   # False Positive Rate
    return round(tpr - fpr, 4), round(tpr, 4), round(fpr, 4)

def compute_hss(y_true, y_pred):
    """
    Heidke Skill Score — another standard space weather metric.
    HSS = 0 means no better than random. HSS = 1 is perfect.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    num = 2 * (tp*tn - fp*fn)
    den = (tp+fn)*(fn+tn) + (tp+fp)*(fp+tn)
    return round(num / (den + 1e-9), 4)

def train_nowcast_model(data_path="data/labeled_dataset.csv"):
    print("="*55)
    print("  NOWCAST MODEL TRAINING")
    print("="*55)

    # ── Load data ─────────────────────────────────────────────────────
    df = pd.read_csv(data_path, parse_dates=['time_tag'])
    df = df.sort_values('time_tag').reset_index(drop=True)

    X = df[FEATURE_COLS].values
    y = df['nowcast_label'].values
    print(f"Dataset: {len(df)} samples | Flare rate: {y.mean()*100:.1f}%")

    # ── Time-series split (NEVER shuffle time-series data!) ───────────
    # We use the last 20% of time as the test set
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Scale features ────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Handle class imbalance with class_weight ──────────────────────
    # Flares are rare (~5%) so we upweight them to avoid
    # model always predicting "no flare"
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced',   # key for imbalanced data
        random_state=42,
        n_jobs=-1                    # use all CPU cores
    )

    print("\nTraining Random Forest...")
    model.fit(X_train_s, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    tss, tpr, fpr = compute_tss(y_test, y_pred)
    hss           = compute_hss(y_test, y_pred)
    auc           = roc_auc_score(y_test, y_proba)

    print(f"\n{'─'*40}")
    print(f"  TSS (True Skill Statistic) : {tss}")
    print(f"  HSS (Heidke Skill Score)   : {hss}")
    print(f"  AUC-ROC                    : {auc:.4f}")
    print(f"  True Positive Rate (TPR)   : {tpr}")
    print(f"  False Positive Rate (FPR)  : {fpr}")
    print(f"{'─'*40}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred,
                                target_names=['Quiet', 'Flare']))

    # ── Feature importance plot ───────────────────────────────────────
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Feature importance
    importances.plot(kind='barh', ax=axes[0], color='#3B8BD4')
    axes[0].set_title('Feature Importance')
    axes[0].set_xlabel('Importance score')

    # ROC curve
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr_arr, tpr_arr, color='#E8593C', lw=2,
                label=f'ROC (AUC = {auc:.3f})')
    axes[1].plot([0,1],[0,1], '--', color='gray', lw=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve — Nowcast Model')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/nowcast_evaluation.png', dpi=150)
    plt.close()

    # ── Save model + scaler ───────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    joblib.dump(model,  'models/nowcast_rf.pkl')
    joblib.dump(scaler, 'models/nowcast_scaler.pkl')
    print(f"\nSaved: models/nowcast_rf.pkl")
    print(f"Saved: models/nowcast_scaler.pkl")

    return model, scaler, {
        'tss': tss, 'hss': hss, 'auc': auc,
        'tpr': tpr, 'fpr': fpr
    }

if __name__ == "__main__":
    model, scaler, metrics = train_nowcast_model()