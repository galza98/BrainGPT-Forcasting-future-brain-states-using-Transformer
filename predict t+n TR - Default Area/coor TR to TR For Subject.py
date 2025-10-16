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
# import os
# import pickle
# import numpy as np
# import torch
# from torch import nn
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
#
# # Import your model class
# # from Transformer import TimeSeriesTransformer
#
# # Configuration
# data_dir = r"D:\Final Project\Yuval_Gal\Processed_Matrices_Vis_Area"
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
# ]
# window_start = 20
# window_size = 30
#
# # Model configuration (must match training)
# model_cfg = {
#     'input_size': 47,
#     'dec_seq_len': window_size,
#     'num_predicted_features': 47,
# }
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # # Load a single model (state_dict or full)
# def load_model(path):
#     m = TimeSeriesTransformer(
#         input_size=model_cfg['input_size'],
#         dec_seq_len=model_cfg['dec_seq_len'],
#         num_predicted_features=model_cfg['num_predicted_features']
#     )
#     sd = torch.load(path, map_location=device)
#     if isinstance(sd, dict) and 'state_dict' in sd:
#         sd = sd['state_dict']
#     m.load_state_dict(sd)
#     return m.float().to(device).eval()

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
#
# # Main processing: load subjects, models, predict and plot
# models = [ load_model(p, device, model_cfg) for p in model_paths ]
#
# with open(test_dirs_pickle, 'rb') as f:
#     subjects = pickle.load(f)
#
# for subj in subjects:
#     subdir = os.path.join(data_dir, subj)
#     pkl_files = sorted([f for f in os.listdir(subdir) if f.endswith('.pkl')])
#
#     # Prepare storage per horizon
#     data_horizons = {h: {'pred': [], 'true': []} for h in range(1, len(models)+1)}
#
#     for fname in pkl_files:
#         raw = pickle.load(open(os.path.join(subdir, fname), 'rb'))
#         arr = np.array(raw)
#         # Ensure shape (timepoints, regions)
#         if arr.ndim==2 and arr.shape[1]==model_cfg['input_size']:
#             data = arr
#         elif arr.ndim==2 and arr.shape[0]==model_cfg['input_size']:
#             data = arr.T
#         else:
#             continue
#
#         # Check length
#         required = window_start + window_size + len(models)
#         if data.shape[0] < required:
#             continue
#
#         segment = data[window_start:window_start+window_size, :]
#         encoder_input = torch.tensor(segment, device=device, dtype=torch.float32).unsqueeze(0)
#
#         for i, m in enumerate(models, start=1):
#             last = segment[-1, :]
#             decoder_input = torch.tensor(last, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
#             with torch.no_grad():
#                 pred = m(encoder_input, decoder_input).cpu().numpy().squeeze()
#             true = data[window_start+window_size+(i-1), :]
#             data_horizons[i]['pred'].append(pred)
#             data_horizons[i]['true'].append(true)
#
#     # After collecting, plot for each horizon
#     for h, d in data_horizons.items():
#         pred_vec = np.concatenate(d['pred']).ravel()
#         true_vec = np.concatenate(d['true']).ravel()
#         # Pearson correlation
#         r = pearsonr(pred_vec, true_vec)[0]
#         # Scatter plot
#         plt.figure(figsize=(6,6))
#         plt.scatter(true_vec, pred_vec, alpha=0.3)
#         plt.title(f"Subject {subj} – Horizon t+{h+1} (r={r:.2f})")
#         plt.xlabel("True values")
#         plt.ylabel("Predicted values")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()






























import os
import pickle
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Import your model class
# from Transformer import TimeSeriesTransformer

# Configuration
data_dir = r"D:\Final Project\Yuval_Gal\Processed_Matrices_Vis_Area"
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
window_size = 30

# Model configuration (must match training)
model_cfg = {
    'input_size': 47,
    'dec_seq_len': window_size,
    'num_predicted_features': 47,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a single model (state_dict or full)
# def load_model(path):
#     m = TimeSeriesTransformer(
#         input_size=model_cfg['input_size'],
#         dec_seq_len=model_cfg['dec_seq_len'],
#         num_predicted_features=model_cfg['num_predicted_features']
#     )
#     sd = torch.load(path, map_location=device)
#     if isinstance(sd, dict) and 'state_dict' in sd:
#         sd = sd['state_dict']
#     m.load_state_dict(sd)
#     return m.float().to(device).eval()

# Main processing: load subjects, models, predict and plot correlation over time
models = [ load_model(p, device, model_cfg) for p in model_paths ]

with open(test_dirs_pickle, 'rb') as f:
    subjects = pickle.load(f)

for subj in subjects:
    subdir = os.path.join(data_dir, subj)
    pkl_files = sorted([f for f in os.listdir(subdir) if f.endswith('.pkl')])

    # Prepare storage for correlations per horizon
    corrs_over_time = {h: [] for h in range(1, len(models) + 1)}

    for fname in pkl_files:
        raw = pickle.load(open(os.path.join(subdir, fname), 'rb'))
        arr = np.array(raw)
        # Ensure shape (timepoints, regions)
        if arr.ndim == 2 and arr.shape[1] == model_cfg['input_size']:
            data = arr
        elif arr.ndim == 2 and arr.shape[0] == model_cfg['input_size']:
            data = arr.T
        else:
            continue

        # Check enough length
        required = window_start + window_size + len(models)
        if data.shape[0] < required:
            continue

        # Extract input window
        segment = data[window_start:window_start + window_size, :]
        encoder_input = torch.tensor(
            segment,
            device=device,
            dtype=torch.float32
        ).unsqueeze(0)  # (1, window_size, regions)

        # For each model horizon, predict and compute correlation
        for i, m in enumerate(models, start=1):
            last = segment[-1, :]
            decoder_input = torch.tensor(
                last,
                device=device,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(1)  # (1,1,regions)

            with torch.no_grad():
                pred = m(encoder_input, decoder_input).cpu().numpy().squeeze()
            true = data[window_start + window_size + (i - 1), :]

            # Pearson correlation for this timepoint across regions
            r = pearsonr(pred.flatten(), true.flatten())[0]
            corrs_over_time[i].append(r)

    # Plot correlations over time for this subject
    plt.figure(figsize=(10, 6))
    for i, values in corrs_over_time.items():
        plt.plot(values, label=f't+{i+1}', marker='o')
    plt.title(f'Subject {subj} – Correlation over time - by movie num')
    plt.xlabel('Movie Num')
    plt.ylabel('Pearson correlation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




