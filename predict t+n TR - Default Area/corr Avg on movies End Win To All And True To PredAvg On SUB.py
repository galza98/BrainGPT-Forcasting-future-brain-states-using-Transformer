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
# 1) פונקציות קורלציה
# ————————————————————————————————————————————————
def corr_last_vs_pred(pred: np.ndarray, last: np.ndarray) -> float:
    """
    מחשבת Pearson r בין חיזוי לבין הדגימה האחרונה בחלון
    """
    return pearsonr(pred.ravel(), last.ravel())[0]

def corr_pred_vs_true(pred: np.ndarray, true: np.ndarray) -> float:
    """
    מחשבת Pearson r בין חיזוי לבין הדגימה האמיתית באופק
    """
    return pearsonr(pred.ravel(), true.ravel())[0]


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


# ————————————————————————————————————————————————
# 3) פרמטרים, נתיבים וטעינת מודלים
# ————————————————————————————————————————————————
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
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+9\predict_TR+9_transformer_VIS_AREA.pth",
    r"D:\Final Project\Yuval_Gal\brain_state_pred-main\נסיונות\predict t+n TR\pred t+10\predict_TR+10_transformer_VIS_AREA.pth"
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

with open(test_dirs_pickle, 'rb') as f:
    subjects = pickle.load(f)


# ————————————————————————————————————————————————
# 4) חישוב ממוצע קורלציות על כל הנבדקים + ציור PLOT יחיד
# ————————————————————————————————————————————————
# אגרגציה גלובלית
overall_last = {h: [] for h in range(1, horizons+1)}
overall_true = {h: [] for h in range(1, horizons+1)}

for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    files = sorted(f for f in os.listdir(subj_dir) if f.endswith('.pkl'))

    for fname in files:
        arr = np.array(pickle.load(open(os.path.join(subj_dir, fname),'rb')))
        if arr.ndim==2 and arr.shape[1]==model_cfg['input_size']:
            data = arr
        elif arr.ndim==2 and arr.shape[0]==model_cfg['input_size']:
            data = arr.T
        else:
            continue

        segment = data[window_start:window_start+window_size, :]
        last_sample = segment[-1, :]
        enc_in = torch.tensor(segment, dtype=torch.float32, device=device).unsqueeze(0)

        for h, m in enumerate(models, start=1):
            dec_in = torch.tensor(last_sample, dtype=torch.float32, device=device)\
                         .unsqueeze(0).unsqueeze(1)
            with torch.no_grad():
                pred = m(enc_in, dec_in).cpu().numpy().squeeze()

            overall_last[h].append(corr_last_vs_pred(pred, last_sample))
            true_sample = data[window_start + window_size + (h-1), :]
            overall_true[h].append(corr_pred_vs_true(pred, true_sample))

# ממוצעים לכל אופק
horizons_sorted = sorted(overall_last.keys())
mean_last = [np.mean(overall_last[h]) for h in horizons_sorted]
mean_true = [np.mean(overall_true[h]) for h in horizons_sorted]

# ציור גרף יחיד
plt.figure(figsize=(8,5))
plt.plot(horizons_sorted, mean_last, linestyle='-', marker='o', label='Last vs Pred')
plt.plot(horizons_sorted, mean_true, linestyle='--', marker='s', label='Pred vs True')
plt.title("Mean Pearson r Across All Subjects")
plt.xlabel("Horizon (t+…)")
plt.ylabel("Mean Pearson r")
plt.xticks(horizons_sorted, [f"t+{h+1}" for h in horizons_sorted])
# plt.ylim(0.0, 1.05)
plt.grid(linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

