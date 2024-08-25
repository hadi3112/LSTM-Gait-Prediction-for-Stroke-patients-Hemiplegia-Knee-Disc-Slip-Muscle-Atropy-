import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out)
        return out

def load_data(path):
    exclude_indices = {1, 3, 5, 7, 9, 10, 12, 15, 16, 17}
    all_data = []
    for i in range(54):
        if i not in exclude_indices:
            mat_data = scipy.io.loadmat(f'{path}final_resultant{i}.mat')
            resultant_waveform = mat_data['final_resultant'].flatten()
            resultant_waveform = np.pad(resultant_waveform, (0, max(0, 150 - len(resultant_waveform))), 'constant')
            all_data.append(resultant_waveform.tolist())
    return all_data

def data_to_X_and_y(data, sequence_length):
    X_train, y_train = [], []
    for wave in data:
        sequences = [wave[i:i + sequence_length] for i in range(len(wave) - sequence_length)]
        labels = [wave[i + sequence_length] for i in range(len(wave) - sequence_length)]
        X_train.append(sequences)
        y_train.append(labels)
    return convert_to_tensors(X_train, y_train, sequence_length)

def convert_to_tensors(X_train, y_train, sequence_length):
    X_train_tensors = [torch.tensor(sequence, dtype=torch.float32) for sequence in X_train]
    y_train_tensors = [torch.tensor(labels, dtype=torch.float32) for labels in y_train]
    X_train_padded = pad_sequence(X_train_tensors, batch_first=True, padding_value=0.0)
    y_train_padded = pad_sequence(y_train_tensors, batch_first=True, padding_value=0.0)
    # Ensure the tensors have the correct dimensions
    max_seq_count = max([tensor.shape[1] for tensor in X_train_tensors])
    X_train_final = X_train_padded[:, :max_seq_count, :sequence_length]
    y_train_final = y_train_padded[:, :max_seq_count]
    return X_train_final, y_train_final

# Normalize and scale the data
def scale_data(data):
    max_val = data.max()
    min_val = data.min()
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data, max_val, min_val


# MAIN
sequence_length = 12  # User-defined sequence length
directory_path = 'AFIRM_DATA/Final_Resultants/'
data = load_data(directory_path)
X_train_final, y_train_final = data_to_X_and_y(data, sequence_length)

# Normalize inputs and outputs
X_train_normalized, max_val_X, min_val_X = scale_data(X_train_final.float())
Y_train_normalized, max_val_Y, min_val_Y = scale_data(y_train_final.float())

# Extract data for a specific waveform
index = 19
X_train_wave1 = X_train_normalized[index]
Y_train_wave1 = Y_train_normalized[index]

model = LSTMModel(sequence_length, 50, 1, 1)                # modify hyperparameters here
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30

# Train the model
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_wave1.unsqueeze(0))
    loss = criterion(outputs.squeeze(), Y_train_wave1)
    loss.backward()
    optimizer.step()

# Predicting after training
predicted_waveform = model(X_train_wave1.unsqueeze(0)).squeeze().detach()
predicted_waveform = (predicted_waveform + 1) / 2 * (max_val_Y - min_val_Y) + min_val_Y
Y_train_wave1 = (Y_train_wave1 + 1) / 2 * (max_val_Y - min_val_Y) + min_val_Y

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(predicted_waveform.numpy(), label='Predicted', marker='o')
plt.plot(Y_train_wave1.numpy(), label='Labelled', marker='x')
plt.title('Comparison of Predicted and Labelled Values')
plt.xlabel('Sequence Index')
plt.ylabel('Waveform Value')
plt.legend()
plt.grid(True)
plt.show()