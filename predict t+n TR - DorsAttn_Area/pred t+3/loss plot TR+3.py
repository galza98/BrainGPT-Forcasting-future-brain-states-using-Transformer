import numpy as np
import matplotlib.pyplot as plt

loss_hist = np.load('predict_TR+3_transformer_VIS_AREA_train_loss.npy')
val_loss_hist = np.load('predict_TR+3_transformer_VIS_AREA_valid_loss.npy')

x_values = list(range(1, loss_hist.shape[0] + 1))

plt.plot(x_values, loss_hist.mean(axis=1), marker='o', label='Train Loss')
plt.plot(x_values, val_loss_hist.mean(axis=1), marker='o', label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training & Validation Loss For Predicting TR+3 VIS AREA')
plt.grid(True)
plt.legend()
plt.savefig(f'predict_TR+3_transformer_VIS_AREA.png')
plt.show()