# import os
# import pickle
# import numpy as np
# import math
# import torch
# import torch.nn as nn
# from torch import Tensor
# from scipy.stats import pearsonr
# import matplotlib.pyplot as plt
#
# # ————————————————————————————————————————————————
# # 1) פונקציות קורלציה + עזר ל-SEM
# # ————————————————————————————————————————————————
# def corr_last_vs_pred(pred: np.ndarray, last: np.ndarray) -> float:
#     """Pearson r בין חיזוי לבין הדגימה האחרונה בחלון."""
#     return pearsonr(pred.ravel(), last.ravel())[0]
#
# def corr_pred_vs_true(pred: np.ndarray, true: np.ndarray) -> float:
#     """Pearson r בין חיזוי לבין הדגימה האמיתית באופק."""
#     return pearsonr(pred.ravel(), true.ravel())[0]
#
# def sem(arr) -> float:
#     """
#     Standard Error of the Mean: SD/sqrt(n).
#     אם יש רק דגימה אחת, מחזיר NaN (אין הגדרה ל-ddof=1).
#     """
#     n = len(arr)
#     if n <= 1:
#         return float('nan')
#     return np.std(arr, ddof=1) / np.sqrt(n)
#
# # ————————————————————————————————————————————————
# # 2) PositionalEncoder ו־Transformer (כפי במקור)
# # ————————————————————————————————————————————————
# class PositionalEncoder(nn.Module):
#     def __init__(
#         self,
#         dropout: float=0.1,
#         max_seq_len: int=5000,
#         d_model: int=512,
#         batch_first: bool=True
#         ):
#         super().__init__()
#         self.d_model = d_model
#         self.dropout = nn.Dropout(p=dropout)
#         self.batch_first = batch_first
#         self.x_dim = 1 if batch_first else 0
#
#         # copy pasted from PyTorch tutorial
#         position = torch.arange(max_seq_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_seq_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: Tensor) -> Tensor:
#         if self.batch_first:
#              x = x + self.pe[:x.size(self.x_dim)].squeeze().unsqueeze(0)
#         else:
#             x = x + self.pe[:x.size(self.x_dim)]
#         return self.dropout(x)
#
# class TimeSeriesTransformer(nn.Module):
#
#     def __init__(self,
#         input_size: int,
#         dec_seq_len: int,
#         batch_first: bool=True,
#         out_seq_len: int=58,
#         max_seq_len: int=5000,
#         dim_val: int=512,
#         n_encoder_layers: int=4,
#         n_decoder_layers: int=4,
#         n_heads: int=8,
#         dropout_encoder: float=0.2,
#         dropout_decoder: float=0.2,
#         dropout_pos_enc: float=0.1,
#         dim_feedforward_encoder: int=2048,
#         dim_feedforward_decoder: int=2048,
#         num_predicted_features: int=1
#         ):
#
#         super().__init__()
#         self.dec_seq_len = dec_seq_len
#         #------ ENCODER ------#
#         # Creating the three linear layers needed for the model
#         self.encoder_input_layer = nn.Linear(
#             in_features=input_size,
#             out_features=dim_val
#             )
#
#         self.positional_encoding_layer = PositionalEncoder(
#             d_model=dim_val,
#             dropout=dropout_pos_enc,
#             batch_first=batch_first
#             )
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim_val,
#             nhead=n_heads,
#             dim_feedforward=dim_feedforward_encoder,
#             dropout=dropout_encoder,
#             batch_first=batch_first
#             )
#         self.encoder = nn.TransformerEncoder(
#             encoder_layer=encoder_layer,
#             num_layers=n_encoder_layers,
#             norm=None
#             )
#
#
#         #------ DECODER ------#
#         self.decoder_input_layer = nn.Linear(
#             in_features=num_predicted_features,
#             out_features=dim_val
#             )
#
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=dim_val,
#             nhead=n_heads,
#             dim_feedforward=dim_feedforward_decoder,
#             dropout=dropout_decoder,
#             batch_first=batch_first
#             )
#
#         self.decoder = nn.TransformerDecoder(
#             decoder_layer=decoder_layer,
#             num_layers=n_decoder_layers,
#             norm=None
#             )
#
#         self.linear_mapping = nn.Linear(
#             in_features=dim_val,
#             out_features=num_predicted_features
#             )
#
#     def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None,
#                 tgt_mask: Tensor=None) -> Tensor:
#         src = self.encoder_input_layer(src)
#         src = self.positional_encoding_layer(src)
#         src = self.encoder(src=src)
#         decoder_output = self.decoder_input_layer(tgt)
#         decoder_output = self.decoder(
#             tgt=decoder_output,
#             memory=src,
#             tgt_mask=tgt_mask,
#             memory_mask=src_mask
#             )
#         decoder_output = self.linear_mapping(decoder_output)
#         return decoder_output
#
# def load_model(path: str, device: torch.device, model_cfg: dict) -> nn.Module:
#     m = TimeSeriesTransformer(
#         input_size=model_cfg['input_size'],
#         dec_seq_len=model_cfg['dec_seq_len'],
#         batch_first=True,
#         out_seq_len=model_cfg.get('out_seq_len', 1),
#         max_seq_len=model_cfg.get('max_seq_len', 5000),
#         dim_val=model_cfg.get('dim_val', 512),
#         n_encoder_layers=model_cfg.get('n_encoder_layers', 4),
#         n_decoder_layers=model_cfg.get('n_decoder_layers', 4),
#         n_heads=model_cfg.get('n_heads', 8),
#         dropout_encoder=model_cfg.get('dropout_encoder', 0.2),
#         dropout_decoder=model_cfg.get('dropout_decoder', 0.2),
#         dropout_pos_enc=model_cfg.get('dropout_pos_enc', 0.1),
#         dim_feedforward_encoder=model_cfg.get('dim_feedforward_encoder', 2048),
#         dim_feedforward_decoder=model_cfg.get('dim_feedforward_decoder', 2048),
#         num_predicted_features=model_cfg['num_predicted_features']
#     )
#     try:
#         sd = torch.load(path, map_location=device)
#         if isinstance(sd, dict) and 'state_dict' in sd:
#             sd = sd['state_dict']
#         m.load_state_dict(sd)
#     except Exception:
#         m = torch.load(path, map_location=device)
#     return m.float().to(device).eval()
# # 4) חישוב ממוצע ו-SEM של קורלציות על כל הנבדקים + ציור
# # ————————————————————————————————————————————————
#
# # ————————————————————————————————————————————————
# # 3) פרמטרים, נתיבים וטעינת מודלים
# # ————————————————————————————————————————————————
# data_dir         = r"D:\Final Project\Yuval_Gal\Processed_Matrices_Vis_Area"
# test_dirs_pickle = r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\test_dirs.pickle"
# model_paths = [
#     # r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\VIS_ZSCORE_VS_NO_ZSCORE\small model no ZS\Small_NO_ZS_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+2\predict_TR+2_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+3\predict_TR+3_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+4\predict_TR+4_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+5\predict_TR+5_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+6\predict_TR+6_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+7\predict_TR+7_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+8\predict_TR+8_transformer_VIS_AREA.pth',
#     r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+9\predict_TR+9_transformer_VIS_AREA.pth",
#     r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+10\predict_TR+10_transformer_VIS_AREA.pth"
# ]
# window_start = 100
# window_size  = 30
#
# model_cfg = {
#     'input_size': 47,
#     'dec_seq_len': window_size,
#     'num_predicted_features': 47,
# }
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# models = [load_model(p, device, model_cfg) for p in model_paths]
# horizons = len(models)
#
# overall_last = {h: [] for h in range(1, horizons+1)}
# overall_true = {h: [] for h in range(1, horizons+1)}
#
# with open(test_dirs_pickle, 'rb') as f:
#     subjects = pickle.load(f)
#
# for subj in subjects:
#     subj_dir = os.path.join(data_dir, subj)
#     files = sorted(f for f in os.listdir(subj_dir) if f.endswith('.pkl'))
#
#     for fname in files:
#         arr = np.array(pickle.load(open(os.path.join(subj_dir, fname),'rb')))
#         if arr.ndim == 2 and arr.shape[1] == model_cfg['input_size']:
#             data = arr
#         elif arr.ndim == 2 and arr.shape[0] == model_cfg['input_size']:
#             data = arr.T
#         else:
#             continue
#
#         # בדיקת גבולות לקטע
#         if window_start + window_size > data.shape[0]:
#             continue
#
#         segment = data[window_start:window_start+window_size, :]
#         last_sample = segment[-1, :]
#
#         enc_in = torch.tensor(segment, dtype=torch.float32, device=device).unsqueeze(0)
#
#         for h, m in enumerate(models, start=1):
#             # בדיקה שהאינדקס של הדגימה האמיתית קיים
#             idx_true = window_start + window_size + (h - 1)
#             if idx_true >= data.shape[0]:
#                 continue
#
#             dec_in = torch.tensor(last_sample, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
#             with torch.no_grad():
#                 pred = m(enc_in, dec_in).cpu().numpy().squeeze()
#
#             overall_last[h].append(corr_last_vs_pred(pred, last_sample))
#             true_sample = data[idx_true, :]
#             overall_true[h].append(corr_pred_vs_true(pred, true_sample))
#
# # ממוצעים ו-SEM לכל אופק
# horizons_sorted = sorted(overall_last.keys())
# mean_last = [np.mean(overall_last[h]) if len(overall_last[h]) else np.nan for h in horizons_sorted]
# mean_true = [np.mean(overall_true[h]) if len(overall_true[h]) else np.nan for h in horizons_sorted]
# sem_last  = [sem(overall_last[h]) for h in horizons_sorted]
# sem_true  = [sem(overall_true[h]) for h in horizons_sorted]
#
# # ציור עם error bars (mean ± SEM)
# plt.figure(figsize=(8, 5))
# plt.errorbar(
#     horizons_sorted, mean_last, yerr=sem_last,
#     fmt='-o', capsize=4, label='Last vs Pred (mean ± SEM)'
# )
# plt.errorbar(
#     horizons_sorted, mean_true, yerr=sem_true,
#     fmt='--s', capsize=4, label='Pred vs True (mean ± SEM)'
# )
# plt.title("Mean Pearson r Across All Subjects")
# plt.xlabel("Horizon (t+…)")
# plt.ylabel("Mean Pearson r")
# # שמירה על ההטייה המקורית של הלייבלים (כי המודלים מתחילים מ־t+2)
# plt.xticks(horizons_sorted, [f"t+{h+1}" for h in horizons_sorted])
# plt.grid(linestyle='--', alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()




















# import os
# import pickle
# import numpy as np
# import math
# import torch
# import torch.nn as nn
# from torch import Tensor
# from scipy.stats import pearsonr
# import matplotlib.pyplot as plt
#
# # ————————————————————————————————————————————————
# # 1) פונקציות קורלציה + עזר ל-SEM
# # ————————————————————————————————————————————————
# def corr_last_vs_pred(pred: np.ndarray, last: np.ndarray) -> float:
#     """Pearson r בין חיזוי לבין הדגימה האחרונה בחלון."""
#     return pearsonr(pred.ravel(), last.ravel())[0]
#
# def corr_pred_vs_true(pred: np.ndarray, true: np.ndarray) -> float:
#     """Pearson r בין חיזוי לבין הדגימה האמיתית באופק."""
#     return pearsonr(pred.ravel(), true.ravel())[0]
#
# def sem(arr) -> float:
#     """
#     Standard Error of the Mean: SD/sqrt(n).
#     אם יש רק דגימה אחת, מחזיר NaN (אין הגדרה ל-ddof=1).
#     """
#     n = len(arr)
#     if n <= 1:
#         return float('nan')
#     return np.std(arr, ddof=1) / np.sqrt(n)
#
# # ————————————————————————————————————————————————
# # 2) PositionalEncoder ו־Transformer (כפי במקור)
# # ————————————————————————————————————————————————
# class PositionalEncoder(nn.Module):
#     def __init__(
#         self,
#         dropout: float=0.1,
#         max_seq_len: int=5000,
#         d_model: int=512,
#         batch_first: bool=True
#         ):
#         super().__init__()
#         self.d_model = d_model
#         self.dropout = nn.Dropout(p=dropout)
#         self.batch_first = batch_first
#         self.x_dim = 1 if batch_first else 0
#
#         # copy pasted from PyTorch tutorial
#         position = torch.arange(max_seq_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_seq_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: Tensor) -> Tensor:
#         if self.batch_first:
#              x = x + self.pe[:x.size(self.x_dim)].squeeze().unsqueeze(0)
#         else:
#             x = x + self.pe[:x.size(self.x_dim)]
#         return self.dropout(x)
#
# class TimeSeriesTransformer(nn.Module):
#
#     def __init__(self,
#         input_size: int,
#         dec_seq_len: int,
#         batch_first: bool=True,
#         out_seq_len: int=58,
#         max_seq_len: int=5000,
#         dim_val: int=512,
#         n_encoder_layers: int=4,
#         n_decoder_layers: int=4,
#         n_heads: int=8,
#         dropout_encoder: float=0.2,
#         dropout_decoder: float=0.2,
#         dropout_pos_enc: float=0.1,
#         dim_feedforward_encoder: int=2048,
#         dim_feedforward_decoder: int=2048,
#         num_predicted_features: int=1
#         ):
#
#         super().__init__()
#         self.dec_seq_len = dec_seq_len
#         #------ ENCODER ------#
#         self.encoder_input_layer = nn.Linear(
#             in_features=input_size,
#             out_features=dim_val
#             )
#
#         self.positional_encoding_layer = PositionalEncoder(
#             d_model=dim_val,
#             dropout=dropout_pos_enc,
#             batch_first=batch_first
#             )
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim_val,
#             nhead=n_heads,
#             dim_feedforward=dim_feedforward_encoder,
#             dropout=dropout_encoder,
#             batch_first=batch_first
#             )
#         self.encoder = nn.TransformerEncoder(
#             encoder_layer=encoder_layer,
#             num_layers=n_encoder_layers,
#             norm=None
#             )
#
#
#         #------ DECODER ------#
#         self.decoder_input_layer = nn.Linear(
#             in_features=num_predicted_features,
#             out_features=dim_val
#             )
#
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=dim_val,
#             nhead=n_heads,
#             dim_feedforward=dim_feedforward_decoder,
#             dropout=dropout_decoder,
#             batch_first=batch_first
#             )
#
#         self.decoder = nn.TransformerDecoder(
#             decoder_layer=decoder_layer,
#             num_layers=n_decoder_layers,
#             norm=None
#             )
#
#         self.linear_mapping = nn.Linear(
#             in_features=dim_val,
#             out_features=num_predicted_features
#             )
#
#     def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None,
#                 tgt_mask: Tensor=None) -> Tensor:
#         src = self.encoder_input_layer(src)
#         src = self.positional_encoding_layer(src)
#         src = self.encoder(src=src)
#         decoder_output = self.decoder_input_layer(tgt)
#         decoder_output = self.decoder(
#             tgt=decoder_output,
#             memory=src,
#             tgt_mask=tgt_mask,
#             memory_mask=src_mask
#             )
#         decoder_output = self.linear_mapping(decoder_output)
#         return decoder_output
#
# def load_model(path: str, device: torch.device, model_cfg: dict) -> nn.Module:
#     m = TimeSeriesTransformer(
#         input_size=model_cfg['input_size'],
#         dec_seq_len=model_cfg['dec_seq_len'],
#         batch_first=True,
#         out_seq_len=model_cfg.get('out_seq_len', 1),
#         max_seq_len=model_cfg.get('max_seq_len', 5000),
#         dim_val=model_cfg.get('dim_val', 512),
#         n_encoder_layers=model_cfg.get('n_encoder_layers', 4),
#         n_decoder_layers=model_cfg.get('n_decoder_layers', 4),
#         n_heads=model_cfg.get('n_heads', 8),
#         dropout_encoder=model_cfg.get('dropout_encoder', 0.2),
#         dropout_decoder=model_cfg.get('dropout_decoder', 0.2),
#         dropout_pos_enc=model_cfg.get('dropout_pos_enc', 0.1),
#         dim_feedforward_encoder=model_cfg.get('dim_feedforward_encoder', 2048),
#         dim_feedforward_decoder=model_cfg.get('dim_feedforward_decoder', 2048),
#         num_predicted_features=model_cfg['num_predicted_features']
#     )
#     try:
#         sd = torch.load(path, map_location=device)
#         if isinstance(sd, dict) and 'state_dict' in sd:
#             sd = sd['state_dict']
#         m.load_state_dict(sd)
#     except Exception:
#         m = torch.load(path, map_location=device)
#     return m.float().to(device).eval()
# # 4) חישוב ממוצע ו-SEM של קורלציות על כל הנבדקים + ציור
# # ————————————————————————————————————————————————
#
# # ————————————————————————————————————————————————
# # 3) פרמטרים, נתיבים וטעינת מודלים
# # ————————————————————————————————————————————————
# data_dir         = r"D:\Final Project\Yuval_Gal\Processed_Matrices_Vis_Area"
# test_dirs_pickle = r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\test_dirs.pickle"
# model_paths = [
#     # r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\VIS_ZSCORE_VS_NO_ZSCORE\small model no ZS\Small_NO_ZS_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+2\predict_TR+2_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+3\predict_TR+3_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+4\predict_TR+4_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+5\predict_TR+5_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+6\predict_TR+6_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+7\predict_TR+7_transformer_VIS_AREA.pth',
#     r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+8\predict_TR+8_transformer_VIS_AREA.pth',
#     r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+9\predict_TR+9_transformer_VIS_AREA.pth",
#     r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+10\predict_TR+10_transformer_VIS_AREA.pth"
# ]
# window_start = 100
# window_size  = 30
#
# model_cfg = {
#     'input_size': 47,
#     'dec_seq_len': window_size,
#     'num_predicted_features': 47,
# }
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# models = [load_model(p, device, model_cfg) for p in model_paths]
# horizons = len(models)
#
# overall_last = {h: [] for h in range(1, horizons+1)}
# overall_true = {h: [] for h in range(1, horizons+1)}
#
# with open(test_dirs_pickle, 'rb') as f:
#     subjects = pickle.load(f)
#
# for subj in subjects:
#     subj_dir = os.path.join(data_dir, subj)
#     files = sorted(f for f in os.listdir(subj_dir) if f.endswith('.pkl'))
#
#     for fname in files:
#         arr = np.array(pickle.load(open(os.path.join(subj_dir, fname),'rb')))
#         if arr.ndim == 2 and arr.shape[1] == model_cfg['input_size']:
#             data = arr
#         elif arr.ndim == 2 and arr.shape[0] == model_cfg['input_size']:
#             data = arr.T
#         else:
#             continue
#
#         # בדיקת גבולות לקטע
#         if window_start + window_size > data.shape[0]:
#             continue
#
#         segment = data[window_start:window_start+window_size, :]
#         last_sample = segment[-1, :]
#
#         enc_in = torch.tensor(segment, dtype=torch.float32, device=device).unsqueeze(0)
#
#         for h, m in enumerate(models, start=1):
#             # בדיקה שהאינדקס של הדגימה האמיתית קיים
#             idx_true = window_start + window_size + (h - 1)
#             if idx_true >= data.shape[0]:
#                 continue
#
#             dec_in = torch.tensor(last_sample, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
#             with torch.no_grad():
#                 pred = m(enc_in, dec_in).cpu().numpy().squeeze()
#
#             overall_last[h].append(corr_last_vs_pred(pred, last_sample))
#             true_sample = data[idx_true, :]
#             overall_true[h].append(corr_pred_vs_true(pred, true_sample))
#
# # ממוצעים ו-SEM לכל אופק
# horizons_sorted = sorted(overall_last.keys())
# mean_last = [np.mean(overall_last[h]) if len(overall_last[h]) else np.nan for h in horizons_sorted]
# mean_true = [np.mean(overall_true[h]) if len(overall_true[h]) else np.nan for h in horizons_sorted]
# sem_last  = [sem(overall_last[h]) for h in horizons_sorted]
# sem_true  = [sem(overall_true[h]) for h in horizons_sorted]
#
# # ציור עם error bars (mean ± SEM) + מספרים של ה-SEM על הגרף
# fig, ax = plt.subplots(figsize=(8, 5))
#
# eb1 = ax.errorbar(
#     horizons_sorted, mean_last, yerr=sem_last,
#     fmt='-o', capsize=4, label='Last vs Pred (mean ± SEM)'
# )
# eb2 = ax.errorbar(
#     horizons_sorted, mean_true, yerr=sem_true,
#     fmt='--s', capsize=4, label='Pred vs True (mean ± SEM)'
# )
#
# ax.set_title("Mean Pearson r Across All Subjects")
# ax.set_xlabel("Horizon (t+…)")
# ax.set_ylabel("Mean Pearson r")
# ax.set_xticks(horizons_sorted)
# ax.set_xticklabels([f"t+{h+1}" for h in horizons_sorted])  # שמירה על ההטייה המקורית
# ax.grid(linestyle='--', alpha=0.3)
# ax.legend()
# fig.tight_layout()
#
# # —— הוספת מספרים של SEM ליד כל נקודה ——
# ymin, ymax = ax.get_ylim()
# dy = ymax - ymin
# offset_up   = 0.02 * dy   # היסט קטן מעל ה-bar לכחול
# offset_down = 0.03 * dy   # היסט קטן מתחת ל-bar לכתום
#
# color_last = eb1[0].get_color()
# color_true = eb2[0].get_color()
#
# # טקסטים ל-Last vs Pred (מעל)
# for x, y, e in zip(horizons_sorted, mean_last, sem_last):
#     if np.isfinite(y) and np.isfinite(e):
#         ax.text(x, y + (e if np.isfinite(e) else 0) + offset_up,
#                 f"±{e:.5f}", ha='center', va='bottom', fontsize=9, color=color_last)
#
# # טקסטים ל-Pred vs True (מתחת)
# for x, y, e in zip(horizons_sorted, mean_true, sem_true):
#     if np.isfinite(y) and np.isfinite(e):
#         ax.text(x, y - (e if np.isfinite(e) else 0) - offset_down,
#                 f"±{e:.5f}", ha='center', va='top', fontsize=9, color=color_true)
#
# plt.show()
































import os
import pickle
import numpy as np
import math
import torch
import torch.nn as nn
from torch import Tensor
from scipy.stats import pearsonr, t   # ← הוספתי t
import matplotlib.pyplot as plt

# ————————————————————————————————————————————————
# 1) פונקציות קורלציה + עזר ל-SEM
# ————————————————————————————————————————————————
def corr_last_vs_pred(pred: np.ndarray, last: np.ndarray) -> float:
    """Pearson r בין חיזוי לבין הדגימה האחרונה בחלון."""
    return pearsonr(pred.ravel(), last.ravel())[0]

def corr_pred_vs_true(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson r בין חיזוי לבין הדגימה האמיתית באופק."""
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
        #------ ENCODER ------#
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

def load_model(path: str, device: torch.device, model_cfg: dict) -> nn.Module:
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
    try:
        sd = torch.load(path, map_location=device)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        m.load_state_dict(sd)
    except Exception:
        m = torch.load(path, map_location=device)
    return m.float().to(device).eval()
# 4) חישוב ממוצע ו-SEM של קורלציות על כל הנבדקים + ציור
# ————————————————————————————————————————————————

# ————————————————————————————————————————————————
# 3) פרמטרים, נתיבים וטעינת מודלים
# ————————————————————————————————————————————————
data_dir         = r"D:\Final Project\Yuval_Gal\Processed_Matrices_Vis_Area"
test_dirs_pickle = r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\test_dirs.pickle"
model_paths = [
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+1\predict_TR+1_transformer_VIS_AREA.pth",
    r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+2\predict_TR+2_transformer_VIS_AREA.pth',
    r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+3\predict_TR+3_transformer_VIS_AREA.pth',
    r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+4\predict_TR+4_transformer_VIS_AREA.pth',
    r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+5\predict_TR+5_transformer_VIS_AREA.pth',
    r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+6\predict_TR+6_transformer_VIS_AREA.pth',
    r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+7\predict_TR+7_transformer_VIS_AREA.pth',
    r'D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+8\predict_TR+8_transformer_VIS_AREA.pth',
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+9\predict_TR+9_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+10\predict_TR+10_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+11\predict_TR+11_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR - Vis Area\pred t+12\predict_TR+12_transformer_VIS_AREA.pth"
]
window_start = 100
window_size  = 30

model_cfg = {
    'input_size': 47,
    'dec_seq_len': window_size,
    'num_predicted_features': 47,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = [load_model(p, device, model_cfg) for p in model_paths]
horizons = len(models)

overall_last = {h: [] for h in range(1, horizons+1)}
overall_true = {h: [] for h in range(1, horizons+1)}

with open(test_dirs_pickle, 'rb') as f:
    subjects = pickle.load(f)

for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    files = sorted(f for f in os.listdir(subj_dir) if f.endswith('.pkl'))

    for fname in files:
        arr = np.array(pickle.load(open(os.path.join(subj_dir, fname),'rb')))
        if arr.ndim == 2 and arr.shape[1] == model_cfg['input_size']:
            data = arr
        elif arr.ndim == 2 and arr.shape[0] == model_cfg['input_size']:
            data = arr.T
        else:
            continue

        # בדיקת גבולות לקטע
        if window_start + window_size > data.shape[0]:
            continue

        segment = data[window_start:window_start+window_size, :]
        last_sample = segment[-1, :]

        enc_in = torch.tensor(segment, dtype=torch.float32, device=device).unsqueeze(0)

        for h, m in enumerate(models, start=1):
            # בדיקה שהאינדקס של הדגימה האמיתית קיים
            idx_true = window_start + window_size + (h - 1)
            if idx_true >= data.shape[0]:
                continue

            dec_in = torch.tensor(last_sample, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
            with torch.no_grad():
                pred = m(enc_in, dec_in).cpu().numpy().squeeze()

            overall_last[h].append(corr_last_vs_pred(pred, last_sample))
            true_sample = data[idx_true, :]
            overall_true[h].append(corr_pred_vs_true(pred, true_sample))

# ממוצעים ו-SEM לכל אופק
horizons_sorted = sorted(overall_last.keys())
mean_last = [np.mean(overall_last[h]) if len(overall_last[h]) else np.nan for h in horizons_sorted]
mean_true = [np.mean(overall_true[h]) if len(overall_true[h]) else np.nan for h in horizons_sorted]
sem_last  = [sem(overall_last[h]) for h in horizons_sorted]
sem_true  = [sem(overall_true[h]) for h in horizons_sorted]

# ====== תוספת: חישובי "כמה הפרש הוא טוב" והדפסות ======
delta = np.array(mean_true) - np.array(mean_last)
pooled_sem = np.sqrt(np.array(sem_true)**2 + np.array(sem_last)**2)
# להימנע מחלוקה ב-0
z_like = np.divide(delta, pooled_sem, out=np.full_like(delta, np.nan), where=pooled_sem > 0)

print("\n— Quick check (Δ / pooled SEM) —")
for h, d, psem, z in zip(horizons_sorted, delta, pooled_sem, z_like):
    print(f"t+{h}: Δ={d:.5f}, pooledSEM={psem:.5f}, Δ/pooledSEM={z:.2f}")

print("\n— Paired comparison (recommended) —")
for h in horizons_sorted:
    a = np.array(overall_true[h], dtype=float)
    b = np.array(overall_last[h], dtype=float)
    n = min(len(a), len(b))
    if n <= 1:
        print(f"t+{h}: not enough samples")
        continue
    a, b = a[:n], b[:n]
    diffs = a - b
    mean_diff = np.mean(diffs)
    sd_diff   = np.std(diffs, ddof=1)
    sem_diff  = sd_diff / np.sqrt(n)
    if sem_diff == 0 or np.isnan(sem_diff):
        t_stat = np.nan
        p_val = np.nan
        ci_low = ci_high = mean_diff
    else:
        t_stat = mean_diff / sem_diff
        p_val  = 2 * (1 - t.cdf(abs(t_stat), df=n-1))
        tcrit  = t.ppf(0.975, df=n-1)
        ci_low, ci_high = mean_diff - tcrit*sem_diff, mean_diff + tcrit*sem_diff
    print(f"t+{h}: meanΔ={mean_diff:.5f}, SEMΔ={sem_diff:.5f}, 95% CI=[{ci_low:.5f}, {ci_high:.5f}], p={p_val:.4f}, n={n}")
#
# # ====== ציור עם error bars (mean ± SEM) + מספרים של ה-SEM על הגרף ======
# fig, ax = plt.subplots(figsize=(8, 5))
#
# eb1 = ax.errorbar(
#     horizons_sorted, mean_last, yerr=sem_last,
#     fmt='-o', capsize=4, label='Last vs Pred (mean ± SEM)'
# )
# eb2 = ax.errorbar(
#     horizons_sorted, mean_true, yerr=sem_true,
#     fmt='--s', capsize=4, label='Pred vs True (mean ± SEM)'
# )
#
# ax.set_title("Mean Pearson r Across All Subjects")
# ax.set_xlabel("Horizon (t+…)")
# ax.set_ylabel("Mean Pearson r")
# ax.set_xticks(horizons_sorted)
# ax.set_xticklabels([f"t+{h+1}" for h in horizons_sorted])  # שמירה על ההטייה המקורית
# ax.grid(linestyle='--', alpha=0.3)
# ax.legend()
# fig.tight_layout()
#
# # —— הוספת מספרים של SEM ליד כל נקודה ——
# ymin, ymax = ax.get_ylim()
# dy = ymax - ymin
# offset_up   = 0.02 * dy   # היסט קטן מעל ה-bar לכחול
# offset_down = 0.03 * dy   # היסט קטן מתחת ל-bar לכתום
#
# color_last = eb1[0].get_color()
# color_true = eb2[0].get_color()
#
# # טקסטים ל-Last vs Pred (מעל)
# for x, y, e in zip(horizons_sorted, mean_last, sem_last):
#     if np.isfinite(y) and np.isfinite(e):
#         ax.text(x, y + (e if np.isfinite(e) else 0) + offset_up,
#                 f"±{e:.5f}", ha='center', va='bottom', fontsize=9, color=color_last)
#
# # טקסטים ל-Pred vs True (מתחת)
# for x, y, e in zip(horizons_sorted, mean_true, sem_true):
#     if np.isfinite(y) and np.isfinite(e):
#         ax.text(x, y - (e if np.isfinite(e) else 0) - offset_down,
#                 f"±{e:.5f}", ha='center', va='top', fontsize=9, color=color_true)
#
#
# # ====== ציור עם error bars (mean ± SEM) + מספרים של ה-SEM על הגרף ======
# fig, ax = plt.subplots(figsize=(8, 5))
#
# eb1 = ax.errorbar(
#     horizons_sorted, mean_last, yerr=sem_last,
#     fmt='-o', capsize=4, label='Last vs Pred (mean ± SEM)'
# )
# eb2 = ax.errorbar(
#     horizons_sorted, mean_true, yerr=sem_true,
#     fmt='--s', capsize=4, label='Pred vs True (mean ± SEM)'
# )
#
# ax.set_title("Mean Pearson r ± SEM Across Prediction Horizons")
# ax.set_xlabel("Horizon (t+…)")
# ax.set_ylabel("Mean Pearson r")
# ax.set_xticks(horizons_sorted)
# ax.set_xticklabels([f"t+{h+1}" for h in horizons_sorted])  # שמירה על ההטייה המקורית
# ax.grid(linestyle='--', alpha=0.3)
# ax.legend()
# fig.tight_layout()
#
# # —— הוספת מספרים של SEM ליד כל נקודה ——
# ymin, ymax = ax.get_ylim()
# dy = ymax - ymin
# offset_up   = 0.02 * dy   # היסט מעל ה-bar לכחול
# offset_down = 0.03 * dy   # היסט מתחת ל-bar לכתום
#
# color_last = eb1[0].get_color()
# color_true = eb2[0].get_color()
#
# for x, y, e in zip(horizons_sorted, mean_last, sem_last):
#     if np.isfinite(y) and np.isfinite(e):
#         ax.text(x, y + e + offset_up, f"±{e:.5f}",
#                 ha='center', va='bottom', fontsize=9, color=color_last)
#
# for x, y, e in zip(horizons_sorted, mean_true, sem_true):
#     if np.isfinite(y) and np.isfinite(e):
#         ax.text(x, y - e - offset_down, f"±{e:.5f}",
#                 ha='center', va='top', fontsize=9, color=color_true)

# ====== ציור עם error bars (mean ± SEM) + מספרים של ה-SEM על הגרף ======
import numpy as np
fig, ax = plt.subplots(figsize=(10, 5))  # הרחבתי לרוחב כדי לפנות מקום למקרא הצד

eb1 = ax.errorbar(
    horizons_sorted, mean_last, yerr=sem_last,
    fmt='-o', capsize=4, label='Last vs Pred (mean ± SEM)'
)
eb2 = ax.errorbar(
    horizons_sorted, mean_true, yerr=sem_true,
    fmt='--s', capsize=4, label='Pred vs True (mean ± SEM)'
)

# — שם גרף —
ax.set_title("Pred vs True vs Last — Mean Pearson r (±SEM) In VIS Area", fontsize=12, pad=10)

ax.set_xlabel("Horizon (t+…)")
ax.set_ylabel("Mean Pearson r")
ax.set_xticks(horizons_sorted)
ax.set_xticklabels([f"t+{h}" for h in horizons_sorted])  # שמירה על ההטייה המקורית
ax.grid(linestyle='--', alpha=0.3)
ax.legend()

# —— הוספת מספרים של SEM ליד כל נקודה ——
ymin, ymax = ax.get_ylim()
dy = ymax - ymin
offset_up   = 0.02 * dy   # היסט מעל ה-bar לכחול
offset_down = 0.03 * dy   # היסט מתחת ל-bar לכתום

color_last = eb1[0].get_color()
color_true = eb2[0].get_color()

for x, y, e in zip(horizons_sorted, mean_last, sem_last):
    if np.isfinite(y) and np.isfinite(e):
        ax.text(x, y + e + offset_up, f"±{e:.5f}",
                ha='center', va='bottom', fontsize=9, color=color_last)

for x, y, e in zip(horizons_sorted, mean_true, sem_true):
    if np.isfinite(y) and np.isfinite(e):
        ax.text(x, y - e - offset_down, f"±{e:.5f}",
                ha='center', va='top', fontsize=9, color=color_true)

# ====== "מקרא" סיכום בצד: טבלה של הערכים וההפרש Δ לכל אופק ======
# דואגים לרווח מימין לגרף
# fig.subplots_adjust(right=0.73)
#
# # חישוב ההפרשים (כתום-כחול)
# delta = np.array(mean_true) - np.array(mean_last)
#
# # ציר חדש לטבלת הסיכום
# summary_ax = fig.add_axes([0.75, 0.12, 0.23, 0.76])  # [left, bottom, width, height] ב־figure coords
# summary_ax.axis('off')
#
# rows = []
# for h, mt, ml, d in zip(horizons_sorted, mean_true, mean_last, delta):
#     rows.append([f"t+{h}",
#                  f"{mt:.3f}" if np.isfinite(mt) else "nan",
#                  f"{ml:.3f}" if np.isfinite(ml) else "nan",
#                  f"{d:.3f}"  if np.isfinite(d)  else "nan"])
#
# table = summary_ax.table(
#     cellText=rows,
#     colLabels=["Horizon", "Pred vs True", "Last vs Pred", "Δ (orange-blue)"],
#     loc='center',
#     cellLoc='center'
# )
# table.auto_set_font_size(False)
# table.set_fontsize(8)
# table.scale(1.05, 1.2)

# plt.tight_layout(rect=[0, 0, 0.73, 1])  # שומר מקום לטבלה מימין
plt.show()

# --- תצוגה + שמירה לקובץ ---
try:
    plt.show(block=True)   # יציג חלון אם יש GUI
except Exception:
    pass
