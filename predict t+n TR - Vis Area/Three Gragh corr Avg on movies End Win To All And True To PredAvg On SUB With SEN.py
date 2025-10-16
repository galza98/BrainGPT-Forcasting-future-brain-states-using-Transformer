# -*- coding: utf-8 -*-
# השוואת מודל לעומת נאיבי: Last vs Pred, Pred vs True, ולבסוף Last vs True
# מצייר שלושה קווים עם error bars (mean ± SEM) עבור אופקים t+1...t+12

import os
import pickle
import numpy as np
import math
import torch
import torch.nn as nn
from torch import Tensor
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ————————————————————————————————————————————————
# 1) פונקציות קורלציה + עזר ל-SEM
# ————————————————————————————————————————————————
def corr_last_vs_pred(pred: np.ndarray, last: np.ndarray) -> float:
    """Pearson r בין חיזוי לבין הדגימה האחרונה בחלון."""
    return pearsonr(pred.ravel(), last.ravel())[0]

def corr_pred_vs_true(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson r בין חיזוי לבין הדגימה האמיתית באופק (או בין last ל-true בבסיס הנאיבי)."""
    return pearsonr(pred.ravel(), true.ravel())[0]

def sem(arr) -> float:
    """
    Standard Error of the Mean: SD/sqrt(n).
    אם יש רק דגימה אחת, מחזיר NaN (אין הגדרה ל-ddof=1).
    """
    n = len(arr)
    if n <= 1:
        return float('nan')
    return np.std(arr, ddof=1) / np.sqrt(n)

# ————————————————————————————————————————————————
# 2) PositionalEncoder ו־Transformer (כפי במקור)
# ————————————————————————————————————————————————
class PositionalEncoder(nn.Module):
    def __init__(
        self,
        dropout: float=0.1,
        max_seq_len: int=5000,
        d_model: int=512,
        batch_first: bool=True
        ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
             x = x + self.pe[:x.size(self.x_dim)].squeeze().unsqueeze(0)
        else:
            x = x + self.pe[:x.size(self.x_dim)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
        input_size: int,
        dec_seq_len: int,
        batch_first: bool=True,
        out_seq_len: int=58,
        max_seq_len: int=5000,
        dim_val: int=512,
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2,
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        num_predicted_features: int=1
        ):
        super().__init__()
        self.dec_seq_len = dec_seq_len

        # ------ ENCODER ------ #
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        )
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            batch_first=batch_first
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )

        # ------ DECODER ------ #
        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        )

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None, tgt_mask: Tensor=None) -> Tensor:
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src)

        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )
        decoder_output = self.linear_mapping(decoder_output)
        return decoder_output

def load_model(path: str, device: torch.device, model_cfg: dict) -> nn.Module:
    """טוען את המודל מה־state_dict או מהאובייקט, לפי מה שקיים בקובץ."""
    m = TimeSeriesTransformer(
        input_size=model_cfg['input_size'],
        dec_seq_len=model_cfg['dec_seq_len'],
        batch_first=True,
        out_seq_len=model_cfg.get('out_seq_len', 1),
        max_seq_len=model_cfg.get('max_seq_len', 5000),
        dim_val=model_cfg.get('dim_val', 512),
        n_encoder_layers=model_cfg.get('n_encoder_layers', 4),
        n_decoder_layers=model_cfg.get('n_decoder_layers', 4),
        n_heads=model_cfg.get('n_heads', 8),
        dropout_encoder=model_cfg.get('dropout_encoder', 0.2),
        dropout_decoder=model_cfg.get('dropout_decoder', 0.2),
        dropout_pos_enc=model_cfg.get('dropout_pos_enc', 0.1),
        dim_feedforward_encoder=model_cfg.get('dim_feedforward_encoder', 2048),
        dim_feedforward_decoder=model_cfg.get('dim_feedforward_decoder', 2048),
        num_predicted_features=model_cfg['num_predicted_features']
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"לא נמצא קובץ מודל: {path}")
    try:
        sd = torch.load(path, map_location=device)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        m.load_state_dict(sd)
    except Exception:
        # ייתכן שהקובץ הוא המודל עצמו
        m = torch.load(path, map_location=device)
    return m.float().to(device).eval()

# ————————————————————————————————————————————————
# 3) פרמטרים, נתיבים וטעינת מודלים
# ————————————————————————————————————————————————
data_dir         = r"D:\Final Project\Yuval_Gal\Processed_Matrices_Vis_Area"
test_dirs_pickle = r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\test_dirs.pickle"

# אופקים t+1 ... t+12 (בדיוק לפי מה ששיתפת)
model_paths = [
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+1\predict_TR+1_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+2\predict_TR+2_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+3\predict_TR+3_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+4\predict_TR+4_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+5\predict_TR+5_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+6\predict_TR+6_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+7\predict_TR+7_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+8\predict_TR+8_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+9\predict_TR+9_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+10\predict_TR+10_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+11\predict_TR+11_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+12\predict_TR+12_transformer_VIS_AREA.pth",
]

# בדיקות נתיבים בסיסיות כדי ליפול יפה אם משהו חסר
assert os.path.isdir(data_dir), f"data_dir לא קיים: {data_dir}"
assert os.path.isfile(test_dirs_pickle), f"pickle לרשימת נבדקים לא קיים: {test_dirs_pickle}"
_missing = [p for p in model_paths if not os.path.isfile(p)]
assert len(_missing) == 0, "קבצי מודלים חסרים:\n" + "\n".join(_missing)

# חלון קידוד
window_start = 0
window_size  = 30

# תצורת המודל (לפי מה ששלחת)
model_cfg = {
    'input_size': 47,
    'dec_seq_len': window_size,
    'num_predicted_features': 47,
}

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models  = [load_model(p, device, model_cfg) for p in model_paths]
horizons = len(models)  # מצופה 12

# מאגרים לקורלציות בכל אופק
overall_last         = {h: [] for h in range(1, horizons+1)}  # Last vs Pred
overall_true         = {h: [] for h in range(1, horizons+1)}  # Pred vs True
overall_last_vs_true = {h: [] for h in range(1, horizons+1)}  # ← חדש: Last vs True

# ————————————————————————————————————————————————
# 4) טעינת רשימת נבדקים ולולאת חישוב
# ————————————————————————————————————————————————
with open(test_dirs_pickle, 'rb') as f:
    subjects = pickle.load(f)
assert isinstance(subjects, (list, tuple)), "test_dirs.pickle צריך להכיל רשימת תיקיות נבדקים"

for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    if not os.path.isdir(subj_dir):
        # מדלגים על נבדק שחסר
        continue

    files = sorted(f for f in os.listdir(subj_dir) if f.endswith('.pkl'))
    for fname in files:
        fpath = os.path.join(subj_dir, fname)
        try:
            arr = pickle.load(open(fpath, 'rb'))
            data_np = np.array(arr)
        except Exception as e:
            # אם הקובץ לא תקין, מדלגים
            continue

        # התאמת ממדים: data בצורה [N_TRs, N_voxels]
        if data_np.ndim == 2 and data_np.shape[1] == model_cfg['input_size']:
            data = data_np
        elif data_np.ndim == 2 and data_np.shape[0] == model_cfg['input_size']:
            data = data_np.T
        else:
            # מידות לא תואמות
            continue

        # מספיק TRs לחלון? אם לא - דילוג
        if window_start + window_size > data.shape[0]:
            continue

        # חלון מקודד והדגימה האחרונה בחלון
        segment = data[window_start:window_start+window_size, :]
        last_sample = segment[-1, :]  # shape: [47]

        # כניסות למודל
        enc_in = torch.tensor(segment, dtype=torch.float32, device=device).unsqueeze(0)         # [1, 30, 47]
        last_t = torch.tensor(last_sample, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)  # [1, 1, 47]

        # לכל אופק (horizon) מחושבים שלושת המדדים
        for h, m in enumerate(models, start=1):  # h=1→t+1, h=12→t+12
            idx_true = window_start + window_size + (h - 1)
            print(idx_true)
            if idx_true >= data.shape[0]:
                # אין מספיק TRs כדי לקחת את הדגימה האמיתית באופק הזה
                continue

            with torch.no_grad():
                pred = m(enc_in, last_t).cpu().numpy().squeeze()  # shape: [47]

            # 1) Last vs Pred
            overall_last[h].append(corr_last_vs_pred(pred, last_sample))

            # 2) Pred vs True
            true_sample = data[idx_true, :]
            overall_true[h].append(corr_pred_vs_true(pred, true_sample))

            # 3) ← חדש: Last vs True (baseline נאיבי לאורך זמן)
            overall_last_vs_true[h].append(corr_pred_vs_true(last_sample, true_sample))

# ————————————————————————————————————————————————
# 5) ממוצעים ו-SEM לכל אופק
# ————————————————————————————————————————————————
horizons_sorted = sorted(overall_last.keys())  # [1..12]

mean_last         = [np.mean(overall_last[h])         if len(overall_last[h])         else np.nan for h in horizons_sorted]
mean_true         = [np.mean(overall_true[h])         if len(overall_true[h])         else np.nan for h in horizons_sorted]
mean_last_vs_true = [np.mean(overall_last_vs_true[h]) if len(overall_last_vs_true[h]) else np.nan for h in horizons_sorted]

sem_last         = [sem(overall_last[h])         for h in horizons_sorted]
sem_true         = [sem(overall_true[h])         for h in horizons_sorted]
sem_last_vs_true = [sem(overall_last_vs_true[h]) for h in horizons_sorted]

# ————————————————————————————————————————————————
# 6) ציור שלושת הקווים עם error bars
# ————————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(9, 5))

eb1 = ax.errorbar(
    horizons_sorted, mean_last, yerr=sem_last,
    fmt='-o', capsize=4, label='Last vs Pred (mean ± SEM)'
)
eb2 = ax.errorbar(
    horizons_sorted, mean_true, yerr=sem_true,
    fmt='--s', capsize=4, label='Pred vs True (mean ± SEM)'
)
eb3 = ax.errorbar(
    horizons_sorted, mean_last_vs_true, yerr=sem_last_vs_true,
    fmt='-.^', capsize=4, label='Last vs True (mean ± SEM)'  # ← הקו החדש
)

ax.set_title("Mean Pearson r Across All Subjects - Visual Network - Window From Movie Start", fontsize=16)
ax.set_xlabel("Horizon (t+…)", fontsize=14)
ax.set_ylabel("Mean Pearson r", fontsize=14)
ax.set_xticks(horizons_sorted)
ax.set_xticklabels([f"t+{h}" for h in horizons_sorted])  # כאן המודלים מתחילים מ־t+1
ax.grid(linestyle='--', alpha=0.3)
ax.legend(loc='best', fontsize=15)
fig.tight_layout()
plt.show()
