import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
import torch
import torch.nn as nn
from torch import nn, Tensor
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

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
        #------ ENCODER ------#
        # Creating the three linear layers needed for the model
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


        #------ DECODER ------#
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

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None,
                tgt_mask: Tensor=None) -> Tensor:
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
#
#
#
# import os
# import pickle
# import torch
# import numpy as np
# from scipy.stats import pearsonr
#
#
# def compute_subject_correlations(
#         data_dir,  # תיקיית השורש שבה יש תיקיות של כל נבדק
#         test_dirs_pickle,  # קובץ pickle עם רשימת התיקיות (test_sub_split)
#         model_paths,  # list באורך 8 של נתיבי המודלים (.pth)
#         window_start=0,  # אינדקס התחלה של החלון (כדוגמה 0)
#         window_size=30,  # אורך החלון (כדוגמה 30)
#         device=None
# ):
#     """
#     עבור כל נבדק ברשימת test_dirs_pickle:
#       1. טוענים את כל קבצי ה-.pkl שבתיקיית data_dir/<test_sub>
#       2. לכל קובץ – בונים segment = f_data[window_start:window_start+window_size, :]
#       3. כל מודל i מחזה את הדגימה ב-index = window_start + window_size + (i-1)
#       4. אוספים את כל הניבויים והערכים האמיתיים, ומחשבים קורלציית פירסון
#     מחזיר dict: { test_sub: corr }.
#     """
#     device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#
#     # 1. טען את רשימת התיקיות (folds)
#     with open(test_dirs_pickle, 'rb') as f:
#         test_sub_split = pickle.load(f)
#
#     # 2. טען את המודלים
#     models = []
#     for mp in model_paths:
#         m = torch.load(mp, map_location=device)
#         m.eval()
#         models.append(m)
#
#     results = {}
#     for test_sub in test_sub_split:
#         sub_dir = os.path.join(data_dir, test_sub)
#         pkl_files = sorted([f for f in os.listdir(sub_dir) if f.endswith('.pkl')])
#
#         all_preds = []
#         all_trues = []
#
#         for fname in pkl_files:
#             path = os.path.join(sub_dir, fname)
#             with open(path, 'rb') as f:
#                 data = pickle.load(f)
#
#             arr = np.array(data)
#             # הנחה: arr.shape = (timepoints, 47) או (47, timepoints)
#             if arr.ndim == 2 and arr.shape[0] == 47:
#                 # סיבוב אם הציר הראשון הוא 47
#                 f_data = arr.T
#             else:
#                 f_data = arr
#
#             # בדיקה שכל הדאטה מספיק ארוך
#             max_required = window_start + window_size + (len(models) - 1)
#             if f_data.shape[0] <= max_required:
#                 raise ValueError(
#                     f"עבור נבדק {test_sub}, הקובץ {fname} קצר מדי "
#                     f"(דורשים לפחות {max_required + 1} דגימות, יש {f_data.shape[0]})"
#                 )
#
#             # חלון קלט קבוע לכל המודלים
#             segment = f_data[window_start: window_start + window_size, :]  # shape (window_size, 47)
#             x = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)  # (1, window_size, 47)
#
#             encoder_input = torch.tensor(segment, device=device).unsqueeze(0).to(device=device, dtype = torch.float32)
#             decoder_input = torch.tensor(segment[-1], device=device).unsqueeze(0).unsqueeze(1).to(device=device, dtype = torch.float32)
#
#             # חיזוי לכל מודל
#             for i, model in enumerate(models, start=1):
#                 with torch.no_grad():
#                     # אם למודל דרוש decoder_input, הוסף כאן לפי הארכיטקטורה שלך
#                     pred = model(encoder_input, decoder_input)  # התאם לקריאות שלך: model(encoder_input) או (enc, dec)
#                 pred_np = pred.cpu().numpy().squeeze()  # (47,)
#
#                 all_preds.append(pred_np)
#
#                 # true vector: נקודת הזמן המתאימה
#                 target_idx = window_start + window_size + (i - 1)
#                 true_np = f_data[target_idx, :]
#                 all_trues.append(true_np)
#
#         # איחוד וקטורים וחשב קורלציה
#         pred_vector = np.concatenate([p.flatten() for p in all_preds])
#         true_vector = np.concatenate([t.flatten() for t in all_trues])
#
#         corr, _ = pearsonr(pred_vector, true_vector)
#         print(f'נבדק {test_sub}: קורלציית פירסון = {corr:.4f}')
#         results[test_sub] = corr
#
#     return results
#
#
# if __name__ == '__main__':
#     data_dir = r"D:\Final Project\Yuval_Gal\Processed_Matrices_Vis_Area"  # תיקייה שבה תיקיות של נבדקים
#     test_dirs_pickle = r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\test_dirs.pickle"  # pickle עם רשימת תיקיות (test_sub_split)
#     model_paths = [
#     # r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\VIS_ZSCORE_VS_NO_ZSCORE\small model no ZS\Small_NO_ZS_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+2\predict_TR+2_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+3\predict_TR+3_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+4\predict_TR+4_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+5\predict_TR+5_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+6\predict_TR+6_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+7\predict_TR+7_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+8\predict_TR+8_transformer_VIS_AREA.pth',
# ]
#
#     # דוגמה: החלון מתחיל ב-0, אורך 30 דגימות
#     corrs = compute_subject_correlations(
#         data_dir=data_dir,
#         test_dirs_pickle=test_dirs_pickle,
#         model_paths=model_paths,
#         window_start=20,
#         window_size=30
#     )















































# predict_tn_full.py

import os
import pickle
import numpy as np
import torch
from torch import nn
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# 1) ייבוא הקוד של המודל שלך
# -----------------------------------------------------------------------------
# from Transformer import TimeSeriesTransformer

# -----------------------------------------------------------------------------
# 2) פונקציית עזר לטעינת המודל (state_dict או מודל מלא), ו–cast ל-float32
# -----------------------------------------------------------------------------
def load_model(path, device, model_cfg):
    # 2.1) אתחול ארכיטקטורה זהה לזו שבאימון
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
    # 2.2) נסיון לטעון state_dict
    try:
        sd = torch.load(path, map_location=device)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        m.load_state_dict(sd)
    except Exception:
        # fallback: שמרת את המודל כולו
        m = torch.load(path, map_location=device)
    # 2.3) לוודא שבכל המשקלים הוא ב-float32 ובמכשיר הנכון
    m = m.float().to(device).eval()
    return m

# -----------------------------------------------------------------------------
# 3) הלב: חישוב קורלציה לכל נבדק
# -----------------------------------------------------------------------------
def compute_subject_correlations(
    data_dir,            # תיקיית השורש שבה יש תיקיות של נבדקים
    test_dirs_pickle,    # pickle עם רשימת שמות התיקיות
    model_paths,         # list ארוך 8 של נתיבי .pth
    window_start,        # אינדקס התחלה של החלון
    window_size,         # אורך החלון
    model_cfg
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3.1) טען רשימת נבדקים
    with open(test_dirs_pickle, 'rb') as f:
        test_subs = pickle.load(f)

    # 3.2) טען מראש את 8 המודלים (float32)
    models = [ load_model(p, device, model_cfg) for p in model_paths ]

    results = {}
    R = model_cfg['input_size']
    N = len(models)

    for subj in test_subs:
        subdir = os.path.join(data_dir, subj)
        pkls = sorted(f for f in os.listdir(subdir) if f.endswith('.pkl'))

        all_preds, all_trues = [], []
        for fn in pkls:
            arr = np.array(pickle.load(open(os.path.join(subdir, fn),'rb')))
            # לוודא (T, R)
            if arr.ndim==2 and arr.shape[1]==R:
                data = arr
            elif arr.ndim==2 and arr.shape[0]==R:
                data = arr.T
            else:
                raise RuntimeError(f"Unexpected shape {arr.shape} in {fn}")

            # דורש לפחות window_start+window_size+N דגימות
            needed = window_start + window_size + N
            if data.shape[0] < needed:
                raise RuntimeError(f"{fn} too short: need ≥{needed}, got {data.shape[0]}")

            # 3.3) החלון
            segment = data[window_start:window_start+window_size, :]  # (W, R)

            # 3.4) encoder_input ב־float32 על ה־device
            encoder_input = torch.tensor(
                segment, device=device, dtype=torch.float32
            ).unsqueeze(0)  # (1, W, R)

            # 3.5) לכל מודל: בונים decoder_input, ניבוי ואיסוף
            for i, m in enumerate(models):
                last = segment[-1, :]  # (R,)
                decoder_input = torch.tensor(
                    last, device=device, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(1)  # (1,1,R)

                with torch.no_grad():
                    pred = m(encoder_input, decoder_input)  # → (1,1,R) או (1,R)
                p_np = pred.cpu().numpy().squeeze()       # (R,)
                all_preds.append(p_np)

                t_idx = window_start + window_size + i
                all_trues.append(data[t_idx, :])         # (R,)

        # 3.6) איחוד וקטורים + קורלציה
        p_vec = np.concatenate(all_preds, axis=0).ravel()
        t_vec = np.concatenate(all_trues, axis=0).ravel()
        r = pearsonr(p_vec, t_vec)[0]
        print(f"{subj:15} → Pearson r = {r:.4f}")
        results[subj] = r

    return results

# -----------------------------------------------------------------------------
# 4) main: הגדרות והרצה
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    data_dir         = r"D:\Final Project\Yuval_Gal\Processed_Matrices_Vis_Area"
    test_dirs_pickle = r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\test_dirs.pickle"
    model_paths = [
        # r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\VIS_ZSCORE_VS_NO_ZSCORE\small model no ZS\Small_NO_ZS_transformer_VIS_AREA.pth',
        r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+2\predict_TR+2_transformer_VIS_AREA.pth',
        r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+3\predict_TR+3_transformer_VIS_AREA.pth',
        r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+4\predict_TR+4_transformer_VIS_AREA.pth',
        r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+5\predict_TR+5_transformer_VIS_AREA.pth',
        r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+6\predict_TR+6_transformer_VIS_AREA.pth',
        r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+7\predict_TR+7_transformer_VIS_AREA.pth',
        r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+8\predict_TR+8_transformer_VIS_AREA.pth',
    ]
    window_start = 20
    window_size  = 30

    # הגדרות הארכיטקטורה – חייבים להתאים למה שאימנת!
    model_cfg = {
        'input_size': 47,             # R
        'dec_seq_len': window_size,   # רק פורמלית
        'num_predicted_features': 47, # אם חוזה 47 אז 47; אם חוזה 1 – 1
        # שאר הפרמטרים כבר default בתוך __init__
    }

    corrs = compute_subject_correlations(
        data_dir, test_dirs_pickle, model_paths,
        window_start, window_size, model_cfg
    )
    print("\n=== All correlations ===")
    for s, r in corrs.items():
        print(f"  {s:15} → {r:.4f}")