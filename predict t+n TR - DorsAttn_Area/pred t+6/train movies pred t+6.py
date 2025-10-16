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
from tqdm import tqdm
import matplotlib.pyplot as plt

class MoviefMRIDataset(Dataset):
    def __init__(self, data_dir, sub_list, sample_size=30, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.data_dir = data_dir
        self.subjects = sub_list
        self.sample_size = sample_size
        self.device = device

        self.all_samples = []
        self.all_targets = []

        for sub in sub_list:
            sub_dir = os.path.join(data_dir, sub)
            pkl_files = [f for f in os.listdir(sub_dir) if f.endswith('.pkl')]
            if len(pkl_files) != 13:
                print(f"Warning: Found {len(pkl_files)} files in the folder {sub_dir}, expected 13")
                continue

            for pkl_file in pkl_files:
                file_path = os.path.join(sub_dir, pkl_file)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)  # [300, 143]
                data = data.T  # [143, 300]

                num_samples = data.shape[0] - sample_size - 5  # 110
                for i in range(num_samples):
                    window = data[i:i + sample_size, :]  # [30, 300] ; TRi - TRi + 29
                    target = data[i + sample_size + 5, :]  # [300] ; TR31
                    self.all_samples.append(window)
                    self.all_targets.append(target)

        self.all_samples = np.stack(self.all_samples, axis=0)  # [1469 * len(sub_list), 30, 300]
        self.all_targets = np.stack(self.all_targets, axis=0)  # [1469 * len(sub_list), 300]

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        time_series = torch.tensor(self.all_samples[idx], dtype=torch.float64, device=self.device)  # ישירות על ה-device
        time_point = torch.tensor(self.all_targets[idx], dtype=torch.float64, device=self.device)  # ישירות על ה-device
        return time_series, time_point


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


def mse_calc(x, x_hat):
    reproduction_loss = nn.functional.mse_loss(x_hat, x)
    return reproduction_loss

def se_calc(x, x_hat):
    x1 = x.cpu().numpy()
    x2 = x_hat.cpu().numpy()
    se = (x1 - x2)**2
    return se

data_dir = r"D:\Final Project\Yuval_Gal\Processed_Matrices_DorsAttn_Area"
window_size = 30
dim_val = 256
n_heads = 4
n_decoder_layers = 2
n_encoder_layers = 2
input_size = 34
dec_seq_len = 1
output_sequence_length = 1
num_predicted_features = 34
batch_first = True
lr = 1e-5
epochs = 100
batch_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Early stopping
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
best_epoch = 0

subject_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
if len(subject_ids) != 176:
    print(f"אזהרה: נמצאו {len(subject_ids)} נבדקים, צפוי ל-176")

with open("D:\Final Project\Yuval_Gal\\brain_state_pred-main\נסיונות\\test_dirs.pickle", 'rb') as file:
    test_subs = pickle.load(file)

train_subs = [sub for sub in subject_ids if sub not in test_subs]

print(f"Train subjects length: {len(train_subs)}")
print(f"Test subjects length: {len(test_subs)}")
print(f"Train subjects: {train_subs}")
print(f"Test subjects: {test_subs}")

train_dataset = MoviefMRIDataset(data_dir, train_subs, sample_size=window_size, device=device)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MoviefMRIDataset(data_dir, test_subs, sample_size=window_size, device=device)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)  # הסרת pin_memory=True

enc_seq_len = window_size
max_seq_len = enc_seq_len
model = TimeSeriesTransformer(
                                dim_val=dim_val,
                                input_size=input_size,
                                n_heads=n_heads,
                                dec_seq_len=dec_seq_len,
                                max_seq_len=max_seq_len,
                                out_seq_len=output_sequence_length,
                                n_decoder_layers=n_decoder_layers,
                                n_encoder_layers=n_encoder_layers,
                                batch_first=batch_first,
                                num_predicted_features=num_predicted_features)
model.to(device)
model = model.double()


loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_hist = []
val_loss_hist = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} Is In Training")
    model.train()
    single_epo_loss = []
    progress_bar = tqdm(range(len(train_dataloader)))
    for batch_idx, (data, target) in enumerate(train_dataloader):
        encoder_input = data
        decoder_input = data[:, -1, :].unsqueeze(1)  # [batch_size, 1, 300]
        pred = model(encoder_input, decoder_input)  # [batch_size, 1, 300]
        trg = target.unsqueeze(1)
        loss = loss_func(trg, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        single_epo_loss.append(loss.cpu().detach().numpy())
        progress_bar.update(1)
    progress_bar.close()
    single_epo_loss = np.array(single_epo_loss)
    loss_hist.append(single_epo_loss)

    model.eval()
    test_mse = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_dataloader)):
            encoder_input = data
            decoder_input = data[:, -1, :].unsqueeze(1)
            pred = model(encoder_input, decoder_input)
            target = target.unsqueeze(1)
            error = mse_calc(target, pred)
            test_mse.append(error.item())
    test_mse = np.array(test_mse)
    val_loss_hist.append(test_mse)
    print(f"Epoch {epoch + 1} complete! \tAverage Loss(MSE): {np.mean(single_epo_loss):.6f} \tValidation Loss (MSE): {np.mean(test_mse):.6f}")

    print(f"np.mean(test_mse) {np.mean(test_mse)} < {best_val_loss} best_val_loss")
    # Early stopping check
    if np.mean(test_mse) < best_val_loss:
        best_val_loss = np.mean(test_mse)
        best_epoch = epoch
        epochs_no_improve = 0
        print(f"Improvement. best_epoch is epoch {best_epoch}.\n")
        torch.save(model, f"predict_TR+6_transformer_DorsAttn_AREA.pth")
    else:
        epochs_no_improve += 1
        print(f"No improvement. Early stopping at epoch {epochs_no_improve}.\n")
        if epochs_no_improve >= patience:
            print(f"\n\nNo improvement for {patience} epochs. Early stopping at epoch {epoch+1}.")
            break

loss_hist = np.array(loss_hist)
val_loss_hist = np.array(val_loss_hist)

torch.save(model, f"predict_TR+6_transformer_DorsAttn_AREA.pth")
np.save(f'predict_TR+6_transformer_DorsAttn_AREA_train_loss.npy', loss_hist)
np.save(f'predict_TR+6_transformer_DorsAttn_AREA_valid_loss.npy', val_loss_hist)

x_values = list(range(1, best_epoch + 2))
plt.plot(x_values, loss_hist.mean(axis=1), label='Train Loss')
plt.plot(x_values, val_loss_hist.mean(axis=1), label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.savefig(f'predict_TR+6_transformer_DorsAttn_AREA.png')
plt.clf()
plt.show()
